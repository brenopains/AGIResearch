import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchaudio
import cv2
import numpy as np
import os
import re
import torchhd
from encodec import EncodecModel
from core.coglang import Manifold
from core.vocal_synthesizer import synthesize, formant_for_vowel

# ─── SpaCy Semantic Encoder ─────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_md")
except Exception:
    _nlp = None

HDC_DIM = 10000
SEMANTIC_DIM = 300
_PROJ_SEMANTIC = None  # lazy init on first use

class GroundingEngine:
    """
    Motor de Grounding Multimodal: Converte Visão, Áudio e Texto em Manifolds da CogLang.
    """
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Visual Encoder: ResNet18 congelada
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.resnet.eval()
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Transformações para os frames de vídeo
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Matrizes de projeção fixa aleatória (geradas com sementes fixas para consistência)
        torch.manual_seed(42)
        self.proj_visual = torch.randn(Manifold.DIMENSIONALITY, 512, device=self.device) / np.sqrt(512)
        self.proj_audio = torch.randn(Manifold.DIMENSIONALITY, 128 * 80, device=self.device) / np.sqrt(128 * 80)
        self.proj_text = torch.randn(Manifold.DIMENSIONALITY, 5000, device=self.device) / np.sqrt(5000)
        
        # Audio transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        ).to(self.device)

        # 2. EnCodec para representação auditiva discreta nativa
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(1.5) # 1.5 kbps -> 2 quantizadores
        self.encodec_model.eval()
        self.encodec_model.to(self.device)

        # Hipervetores de Binding para chaves de EnCodec
        # Q (Quantizadores: 0, 1), T (Frames: 0..74), V (Valores: 0..1023)
        self.quant_tensors = torchhd.random(2, Manifold.DIMENSIONALITY, "MAP", device=self.device)
        self.time_tensors = torchhd.random(75, Manifold.DIMENSIONALITY, "MAP", device=self.device)
        self.code_value_tensors = torchhd.random(1024, Manifold.DIMENSIONALITY, "MAP", device=self.device)

        # 3. Hipervetores de Posição Temporal para sequenciamento visual (Sonho Acordado)
        self.position_manifolds = [
            Manifold(torchhd.random(1, Manifold.DIMENSIONALITY, "MAP", device=self.device).squeeze(0), device=self.device)
            for _ in range(6)
        ]

        # Resampler de Áudio (22050 Hz -> 24000 Hz)
        self.resample_22_to_24 = torchaudio.transforms.Resample(orig_freq=22050, new_freq=24000).to(self.device)

        
    def _inject_nt_state(self, manifold, concept_key: str, contexts=None):
        """
        Injeta a nt_signature e causal_modifiers no Manifold de acordo com o conceito e contextos ativos.
        """
        try:
            from .neuromodulator_state import NeuromodulatorState, NT_NAMES, CAUSAL_MODIFIERS
            nts = NeuromodulatorState()
            if contexts:
                nts.set_contexts(contexts)
            sig = nts.resolve_concept_nt(concept_key, contexts)
            manifold.nt_signature = {name: float(val) for name, val in zip(NT_NAMES, sig)}
            
            if concept_key in CAUSAL_MODIFIERS:
                manifold.causal_modifiers = []
                for mod in CAUSAL_MODIFIERS[concept_key]:
                    delta_dict = {name: float(val) for name, val in zip(NT_NAMES, mod["delta"])}
                    manifold.causal_modifiers.append({
                        "context": mod["context"],
                        "delta": delta_dict
                    })
        except Exception:
            pass
        return manifold

    def encode_video(self, video_path, num_frames=8, contexts=None):
        """
        Lê frames uniformemente do vídeo, extrai features visuais e agrupa em um Manifold.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado em: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return Manifold(device=self.device)
            
        # Determina os índices dos frames a serem extraídos
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frame_manifolds = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Converte de BGR para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Transforma e adiciona dimensão de batch
            img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Extrai features (1, 512, 1, 1) -> flatten para (512)
                features = self.feature_extractor(img_tensor).view(-1)
                # Projeta no espaço HDC (10000)
                hdc_vec = torch.matmul(self.proj_visual, features)
                # Converte para bipolar esparso
                hdc_vec = torch.sign(hdc_vec)
                frame_manifolds.append(Manifold(hdc_vec, device=self.device))
                
        cap.release()
        
        if not frame_manifolds:
            return Manifold(device=self.device)
            
        # Bundling: Superposição de todos os frames (representação temporal condensada com positional binding)
        bundled = frame_manifolds[0] * self.position_manifolds[0]
        for i, fm in enumerate(frame_manifolds[1:]):
            if i + 1 < len(self.position_manifolds):
                bundled = bundled + (fm * self.position_manifolds[i + 1])
            else:
                bundled = bundled + fm
            
        return self._inject_nt_state(bundled, "unknown", contexts)

    def encode_video_stream(self, video_path: str, num_frames: int = 8, contexts=None):
        """
        Generator: processa vídeo frame a frame sem acumular em RAM.
        Para CADA frame: extrai features -> projeta HDC -> descarta o raw.
        No final, faz bundle temporal e YIELD o manifold do vídeo inteiro.
        """
        if not os.path.exists(video_path):
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        bundled = None
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(img_tensor).view(-1)
                hdc_vec = torch.sign(torch.matmul(self.proj_visual, features))

            mf = Manifold(hdc_vec, device=self.device)

            # Binding posicional e bundling incremental (sem lista)
            if bundled is None:
                if i < len(self.position_manifolds):
                    bundled = mf * self.position_manifolds[i]
                else:
                    bundled = mf
            else:
                if i < len(self.position_manifolds):
                    bundled = bundled + (mf * self.position_manifolds[i])
                else:
                    bundled = bundled + mf

            # Libera tensores intermediários para o GC
            del img_tensor, features, hdc_vec, frame

        cap.release()

        if bundled is not None:
            yield self._inject_nt_state(bundled, "unknown", contexts)

    def synthesize_dog_bark(self):
        """
        Sintetiza um som realista de latido de cachorro usando o vocal synthesizer nativo.
        Gera uma sequência de 3 latidos curtos com envelopes de amplitude rápidos e formantes abertos.
        """
        f1, f2, f3 = 350, 1000, 2200
        waveforms = []
        pitches = [280, 260, 245]
        for pitch in pitches:
            wav_bytes = synthesize(pitch=pitch, f1=f1, f2=f2, f3=f3, duration=0.15, voiced=True)
            raw_data = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
            waveforms.append(torch.tensor(raw_data, device=self.device))
            # Silêncio curto entre latidos (0.08 s)
            silence = torch.zeros(int(22050 * 0.08), device=self.device)
            waveforms.append(silence)
            
        return torch.cat(waveforms[:-1])

    def synthesize_caption_audio(self, caption):
        """
        Sintetiza a legenda como áudio vocal usando o sintetizador nativo de formantes.
        Mapeia palavras em sons de vogais. Se for cachorro latindo, gera latidos nativos.
        """
        words = caption.lower().split()
        if "dog" in words and any(b in caption.lower() for b in ["bark", "barks", "barking"]):
            # Gera latidos nativos de cachorro
            return self.synthesize_dog_bark()

        waveforms = []
        
        # Mapeamento simples de vogais e ritmos baseados na semântica
        vowel_map = {
            'a': 'aa', 'e': 'ee', 'i': 'ii', 'o': 'oo', 'u': 'uu',
            'm': 'mmm', 'n': 'mmm'
        }
        
        for word in words:
            # Encontra a primeira vogal ou usa 'aa' como padrão
            vowel = 'aa'
            for char in word:
                if char in vowel_map:
                    vowel = vowel_map[char]
                    break
            
            # Sintetiza a vogal
            f1, f2, f3 = formant_for_vowel(vowel)
            pitch = 120 + (len(word) % 5) * 10 # Pitch varia levemente com o tamanho da palavra
            duration = 0.15 + (len(word) % 3) * 0.05 # Duração proporcional
            
            wav_bytes = synthesize(pitch=pitch, f1=f1, f2=f2, f3=f3, duration=duration)
            # Converte os bytes do WAV para tensor 1D (pulando os 44 bytes de cabeçalho WAV)
            raw_data = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
            waveforms.append(torch.tensor(raw_data, device=self.device))
            
        # Junta todas as palavras com silêncio curto de intervalo
        silence = torch.zeros(int(22050 * 0.05), device=self.device)
        full_audio_list = []
        for w in waveforms:
            full_audio_list.append(w)
            full_audio_list.append(silence)
            
        if not full_audio_list:
            return torch.zeros(22050, device=self.device)
            
        return torch.cat(full_audio_list[:-1])
        
    def encode_audio(self, audio_tensor, contexts=None):
        """
        Recebe a waveform de áudio, calcula o MelSpectrogram e projeta em um Manifold.
        """
        # Garante que seja 1D
        audio_tensor = audio_tensor.reshape(-1)
        if audio_tensor.shape[0] < 22050:
            # Preenche com zeros se for muito curto
            pad = torch.zeros(22050 - audio_tensor.shape[0], device=self.device)
            audio_tensor = torch.cat([audio_tensor, pad])
            
        # Computa MelSpectrogram
        with torch.no_grad():
            mel_spec = self.mel_transform(audio_tensor)
            # Redimensiona/achata para corresponder à dimensão de projeção de áudio
            # Para manter leve, achatamos e cortamos ou preenchemos
            mel_flat = mel_spec.reshape(-1)
            target_size = self.proj_audio.shape[1]
            if mel_flat.shape[0] < target_size:
                pad = torch.zeros(target_size - mel_flat.shape[0], device=self.device)
                mel_flat = torch.cat([mel_flat, pad])
            else:
                mel_flat = mel_flat[:target_size]
                
            # Projeta no espaço HDC
            hdc_vec = torch.matmul(self.proj_audio, mel_flat)
            hdc_vec = torch.sign(hdc_vec)
            
        manifold = Manifold(hdc_vec, device=self.device)
        return self._inject_nt_state(manifold, "unknown", contexts)

    def encode_audio_encodec(self, audio_tensor, contexts=None):
        """
        Converte a waveform de áudio em Manifold usando EnCodec e binding HDC.
        Recebe waveform de 22kHz, resampleia para 24kHz e codifica.
        """
        # Garante formato 1D no dispositivo correto
        audio_tensor = audio_tensor.to(self.device).reshape(-1)
        
        # Resampleia para 24kHz se necessário
        audio_24k = self.resample_22_to_24(audio_tensor)
        
        # Reshape para [batch, channels, samples] = [1, 1, 24000]
        audio_24k = audio_24k.reshape(1, 1, -1)
        if audio_24k.shape[-1] < 24000:
            pad = torch.zeros(1, 1, 24000 - audio_24k.shape[-1], device=self.device)
            audio_24k = torch.cat([audio_24k, pad], dim=-1)
        else:
            audio_24k = audio_24k[:, :, :24000]

        with torch.no_grad():
            # Executa codificador EnCodec
            encoded_frames = self.encodec_model.encode(audio_24k)
            codes = encoded_frames[0][0].squeeze(0) # Shape: [2, 75]
            
            # Faz o binding de cada código: Q_q * T_t * V_val
            summed_vector = torch.zeros(Manifold.DIMENSIONALITY, device=self.device)
            for q in range(2):
                for t in range(75):
                    val_idx = codes[q, t].item()
                    term = self.quant_tensors[q] * self.time_tensors[t] * self.code_value_tensors[val_idx]
                    summed_vector = summed_vector + term
                    
            # Bipolariza
            summed_vector = torch.sign(summed_vector)
            
        manifold = Manifold(summed_vector, device=self.device)
        return self._inject_nt_state(manifold, "unknown", contexts)

    def decode_audio_encodec(self, audio_manifold):
        """
        Decodifica o Manifold de áudio de volta em uma waveform de 24kHz usando unbinding HDC e EnCodec.
        """
        with torch.no_grad():
            codes = torch.zeros(1, 2, 75, dtype=torch.int64, device=self.device)
            
            # Unbinding para extrair cada código discreto
            for q in range(2):
                for t in range(75):
                    query = audio_manifold.tensor * self.quant_tensors[q] * self.time_tensors[t]
                    # Busca pelo maior casamento no dicionário de valores (V)
                    sims = torch.matmul(self.code_value_tensors, query)
                    best_idx = torch.argmax(sims).item()
                    codes[0, q, t] = best_idx
                    
            # Decodifica usando EnCodec
            decoded_frames = [(codes, None)]
            decoded_audio = self.encodec_model.decode(decoded_frames)
            return decoded_audio.squeeze(0).squeeze(0) # Retorna tensor 1D
        
    def encode_text_sync(self, text, contexts=None):
        """
        Versão síncrona (CPU, sem await).

        Projeção Johnson-Lindenstrauss com Dithering:
        1. SpaCy 300D → L2 normalize
        2. Matriz de projeção N(0,1) SEM divisão por sqrt(300) —
           preserva variância ~1 na saída (antes: 1/300 ≈ 0.003)
        3. Dithering (N(0,0.01)) para quebrar empates exatos
        4. sign() → bipolar {-1, +1}

        Isso elimina o colapso semântico onde palavras diferentes
        (ex: gravity vs mechanics) viravam vetores idênticos.
        """
        global _PROJ_SEMANTIC
        if _PROJ_SEMANTIC is None:
            torch.manual_seed(42)
            _PROJ_SEMANTIC = torch.randn(SEMANTIC_DIM, HDC_DIM,
                                          device=self.device)

        if _nlp is None:
            raise RuntimeError("spaCy en_core_web_md não disponível")

        with torch.no_grad():
            doc = _nlp(text)
            semantic_vec = torch.tensor(doc.vector, dtype=torch.float32, device=self.device)
            if semantic_vec.norm() < 1e-8:
                return Manifold(torch.zeros(HDC_DIM, device=self.device), device=self.device)
            semantic_vec = semantic_vec / semantic_vec.norm()
            hdc_float = torch.matmul(semantic_vec, _PROJ_SEMANTIC)
            noise = torch.randn_like(hdc_float) * 0.01
            hdc_bipolar = torch.where(hdc_float + noise >= 0, 1.0, -1.0)
            
        manifold = Manifold(hdc_bipolar, device=self.device)
        clean_text = text.strip().lower()
        return self._inject_nt_state(manifold, clean_text, contexts)

    async def encode_text_async(self, text, contexts=None):
        """Versão assíncrona: roda spaCy num thread executor para não travar o event loop."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_text_sync, text, contexts)

    def encode_text(self, text, contexts=None):
        """Versão síncrona (compatível com código existente)."""
        return self.encode_text_sync(text, contexts)

    def encode_texts_batch(self, texts, contexts=None):
        """Codifica múltiplos textos em lote usando spaCy nlp.pipe (mais rápido)."""
        global _PROJ_SEMANTIC
        if _PROJ_SEMANTIC is None:
            torch.manual_seed(42)
            _PROJ_SEMANTIC = torch.randn(SEMANTIC_DIM, HDC_DIM, device=self.device)

        if _nlp is None:
            raise RuntimeError("spaCy en_core_web_md não disponível")

        results = []
        with torch.no_grad():
            for doc in _nlp.pipe(texts, batch_size=256):
                semantic_vec = torch.tensor(doc.vector, dtype=torch.float32, device=self.device)
                if semantic_vec.norm() < 1e-8:
                    results.append(Manifold(torch.zeros(HDC_DIM, device=self.device), device=self.device))
                    continue
                semantic_vec = semantic_vec / semantic_vec.norm()
                hdc_float = torch.matmul(semantic_vec, _PROJ_SEMANTIC)
                noise = torch.randn_like(hdc_float) * 0.01
                hdc_bipolar = torch.where(hdc_float + noise >= 0, 1.0, -1.0)
                manifold = Manifold(hdc_bipolar, device=self.device)
                results.append(manifold)

        return results

    # Palavras funcionais (stopwords) ignoradas na extração SVO
    STOPWORDS_SVO = frozenset({
        'the','a','an','is','are','was','were','be','been','has','have','had',
        'do','does','did','will','would','can','could','may','might','shall',
        'to','of','in','on','for','with','by','as','at','from','into','through',
        'and','or','but','if','because','so','than','that','this','these','those',
        'i','you','he','she','it','we','they','me','him','her','us','them',
        'my','your','his','its','our','their','no','not','very','just',
        'o','a','os','as','um','uma','uns','umas','e','ou','mas','se','por',
        'para','com','sem','de','da','do','das','dos','no','na','nos','nas',
        'em','ele','ela','eles','elas','eu','tu','voce','nos','isto','isso',
        'aquele','aquela','seu','sua','seus','suas','meu','minha','teu','tua',
        'sao','tem','esta','estao','era','foram','ser','ter','estar','poder',
        'fazer','mais','menos','muito','pouco','como','quando','onde','porque',
        'ja','ainda','tambem','so','nao','sim',
    })

    @staticmethod
    def extract_svo(text: str):
        """
        Extrai Sujeito, Verbo, Objeto de uma frase curta.
        Usa heurística simples: ignora stopwords, pega as 3 primeiras
        palavras significativas como [suj, ver, obj].
        """
        import unicodedata
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        content = [w for w in words if w not in GroundingEngine.STOPWORDS_SVO]

        suj = content[0] if len(content) > 0 else ""
        ver = content[1] if len(content) > 1 else ""
        obj = content[2] if len(content) > 2 else ""

        return suj, ver, obj

    def encode_phrase(self, text: str, contexts=None) -> Manifold:
        """
        Codifica uma frase como Role-Filler Structure no espaço HDC.

        Frase_Manifold = ROLE_SUBJECT * word_suj + ROLE_VERB * word_ver
                         + ROLE_OBJECT * word_obj

        Permite que o BrocaDecoder faça unbinding por papel sintático
        e reconstrua a frase com ordem correta (suj -> ver -> obj).
        """
        from .coglang import bind_roles
        suj, ver, obj = self.extract_svo(text)

        pairs = []
        if suj:
            pairs.append(("suj", self.encode_text(suj, contexts)))
        if ver:
            pairs.append(("ver", self.encode_text(ver, contexts)))
        if obj:
            pairs.append(("obj", self.encode_text(obj, contexts)))

        if len(pairs) < 2:
            return self.encode_text(text, contexts)

        manifold = bind_roles(*pairs)
        clean_text = text.strip().lower()
        return self._inject_nt_state(manifold, clean_text, contexts)

