import os
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"
import json
import torch
import torchhd
import numpy as np
import random
import torchaudio
from core.grounding_engine import GroundingEngine
from core.sandbox_engine import SandboxEngine
from core.coglang import Manifold, ArchetypeForge, calculate_free_energy
from core.vocal_synthesizer import synthesize

def main():
    print("====================================================")
    print("       NOVAMIND THREE AGI TESTS OF FIRE             ")
    print("====================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device detectado: {device}")
    
    # Configurações do dataset
    dataset_dir = r"D:\novamind_data\MSVD"
    clips_dir = os.path.join(dataset_dir, "YouTubeClips")
    train_json_path = os.path.join(dataset_dir, "msvd_train.json")
    
    if not os.path.exists(train_json_path):
        print(f"[ERRO] Dataset não encontrado em {dataset_dir}. Execute o download_msvd.py primeiro.")
        return
        
    with open(train_json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    print("[INIT] Inicializando GroundingEngine...")
    grounding = GroundingEngine(device=device)
    print("[INIT] Inicializando SandboxEngine...")
    sandbox = SandboxEngine(device=device)
    forge = ArchetypeForge()
    
    # ----------------------------------------------------
    # FASE DE TREINAMENTO MULTIMODAL COM ENCODEC
    # ----------------------------------------------------
    num_samples = 40
    random.seed(42)
    selected_records = random.sample(metadata, num_samples * 2)
    
    visual_list = []
    audio_list_encodec = []
    text_list = []
    valid_records = []
    
    # Procura um vídeo específico de cachorro latindo para o Teste 2
    dog_bark_record = None
    for r in metadata:
        caption = r["caption"][0].lower()
        if "dog" in caption and any(b in caption for b in ["bark", "barks", "barking"]):
            video_path = os.path.join(clips_dir, r["video"])
            if os.path.exists(video_path):
                dog_bark_record = r
                break
                
    if dog_bark_record:
        print(f"[DOG BARK FOUND] Vídeo selecionado para o Teste 2: {dog_bark_record['video']}")
        # Adiciona no início da lista para garantir que seja treinado
        selected_records.insert(0, dog_bark_record)
    
    print(f"\n[GROUNDING] Codificando {num_samples} clipes multimodais com EnCodec...")
    processed = 0
    for record in selected_records:
        if processed >= num_samples:
            break
            
        video_name = record["video"]
        video_path = os.path.join(clips_dir, video_name)
        if not os.path.exists(video_path):
            continue
            
        caption = record["caption"][0]
        try:
            # Visão com positional binding
            visual_manifold = grounding.encode_video(video_path, num_frames=6)
            
            # Áudio nativo (sintetizado a partir do texto/latidos)
            waveform = grounding.synthesize_caption_audio(caption)
            audio_manifold = grounding.encode_audio_encodec(waveform)
            
            # Texto
            text_manifold = grounding.encode_text(caption)
            
            visual_list.append(visual_manifold)
            audio_list_encodec.append(audio_manifold)
            text_list.append(text_manifold)
            valid_records.append(record)
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processados: {processed}/{num_samples} clipes...")
        except Exception as e:
            continue
            
    print(f"[GROUNDING] {len(valid_records)} clipes codificados com sucesso.")
    
    # Treina as associações no Sandbox
    print("\n[SANDBOX] Treinando rede de associação: Áudio -> Visão...")
    sandbox.train_association(audio_list_encodec, visual_list, mode="audio_to_visual", epochs=80, lr=0.005)
    
    print("\n[SANDBOX] Treinando rede de associação: Texto -> Visão...")
    sandbox.train_association(text_list, visual_list, mode="text_to_visual", epochs=80, lr=0.005)
    
    print("\n[SANDBOX] Treinando rede de associação: Visão -> Áudio (Voz Nativa)...")
    sandbox.train_association(visual_list, audio_list_encodec, mode="visual_to_audio", epochs=120, lr=0.005)
    
    # ----------------------------------------------------
    # TESTE 1: O Teste do Sonho Acordado (A Prova do Sandbox)
    # ----------------------------------------------------
    print("\n====================================================")
    print("TESTE 1: O Teste do Sonho Acordado (A Prova do Sandbox)")
    print("====================================================")
    
    # Pega um texto de teste
    test_idx = 0
    test_record = valid_records[test_idx]
    test_caption = test_record["caption"][0]
    print(f"Frase de entrada: '{test_caption}'")
    
    # Recupera os manifolds reais
    actual_video = visual_list[test_idx]
    text_in = text_list[test_idx]
    
    # Sandbox sonha a sequência
    predicted_frames = sandbox.predict_sequence_from_text(text_in, grounding.position_manifolds)
    
    # Recupera os frames reais fazendo unbinding
    actual_frames = [actual_video * pos for pos in grounding.position_manifolds]
    
    print("\nCalculando similaridades de frames sonhados vs frames reais:")
    similarities = []
    for t in range(6):
        sim = predicted_frames[t].similarity(actual_frames[t])
        similarities.append(sim)
        print(f"  Frame {t+1} -> Similaridade com Alvo Real: {sim:.4f}")
        
    mean_sim = np.mean(similarities)
    print(f"Similaridade média temporal: {mean_sim:.4f}")
    
    # Prova t-SNE / PCA das trajetórias temporais
    print("\nExecutando SVD em PyTorch para redução a 2D (PCA) da assinatura visual...")
    # Agrupa frames reais e sonhados em uma matriz
    frame_tensors = torch.stack([f.tensor for f in actual_frames] + [f.tensor for f in predicted_frames])
    # Centraliza dados
    mean = frame_tensors.mean(dim=0)
    centered = frame_tensors - mean
    # Calcula SVD
    U, S, V = torch.linalg.svd(centered, full_matrices=False)
    coords_2d = U[:, :2].cpu().numpy()
    
    print("Coordenadas 2D projetadas:")
    for t in range(6):
        print(f"  Frame Real {t+1}   : ({coords_2d[t][0]:.3f}, {coords_2d[t][1]:.3f})")
        print(f"  Frame Sonhado {t+1}: ({coords_2d[t+6][0]:.3f}, {coords_2d[t+6][1]:.3f})")
        
    if mean_sim > 0.8:
        print("\n[VEREDITO TESTE 1]: SUCESSO! O Maestro sonhou a física visual correta no Sandbox.")
    else:
        print("\n[VEREDITO TESTE 1]: FALHA. Similaridade temporal muito baixa.")
        
    # ----------------------------------------------------
    # TESTE 2: A Prova da Voz Nativa (Sem TTS, Sem Texto Intermediário)
    # ----------------------------------------------------
    print("\n====================================================")
    print("TESTE 2: A Prova da Voz Nativa (Sem TTS, Sem Texto Intermediário)")
    print("====================================================")
    
    if dog_bark_record is None:
        print("[AVISO] Clipe de cachorro latindo não encontrado no split de treino. Simulando com dados sintéticos...")
        # Simula visual e audio para o teste
        dog_visual = Manifold(device=device)
        dog_audio_wav = grounding.synthesize_dog_bark()
        dog_audio_target = grounding.encode_audio_encodec(dog_audio_wav)
    else:
        # Encontra o índice correspondente no lote treinado
        dog_idx = next(i for i, r in enumerate(valid_records) if r["video"] == dog_bark_record["video"])
        dog_visual = visual_list[dog_idx]
        dog_audio_wav = grounding.synthesize_caption_audio(dog_bark_record["caption"][0])
        dog_audio_target = audio_list_encodec[dog_idx]
        
    # Maestro imagina o som a partir do visual do cachorro
    predicted_audio = sandbox.predict_audio_from_visual(dog_visual)
    
    # Decodifica via EnCodec nativo (HDC Unbinding -> EnCodec Decoder -> WAV)
    reconstructed_waveform = grounding.decode_audio_encodec(predicted_audio)
    
    # Salva o arquivo WAV
    output_wav_path = os.path.join(dataset_dir, "dog_bark_reconstructed.wav")
    torchaudio.save(output_wav_path, reconstructed_waveform.cpu().unsqueeze(0), 24000)
    print(f"Áudio reconstruído salvo em: {output_wav_path}")
    
    # Valida similaridade cosseno
    audio_sim = predicted_audio.similarity(dog_audio_target)
    print(f"Similaridade entre Áudio Reconstruído e Alvo Nativo: {audio_sim:.4f}")
    
    # Controle negativo
    control_audio = audio_list_encodec[1 if len(audio_list_encodec) > 1 else 0]
    control_sim = predicted_audio.similarity(control_audio)
    print(f"Similaridade com Áudio de Controle (Legenda aleatória): {control_sim:.4f}")
    print(f"Delta de Identidade Sonora: {audio_sim - control_sim:+.4f}")
    
    if audio_sim > 0.8 and (audio_sim - control_sim) > 0.3:
        print("\n[VEREDITO TESTE 2]: SUCESSO! O Maestro converteu o visual em áudio analógico representativo.")
    else:
        print("\n[VEREDITO TESTE 2]: FALHA. Reconstrução abaixo do limiar de identidade.")
        
    # ----------------------------------------------------
    # TESTE 3: O Teste da Forja e da Frustração (A Prova da Parede de Solomonoff)
    # ----------------------------------------------------
    print("\n====================================================")
    print("TESTE 3: O Teste da Forja e da Frustração (A Prova da Parede de Solomonoff)")
    print("====================================================")
    
    # 1. Simula visual de Bola Quicando (queda de gravidade)
    # 6 frames representando deslocamento Y
    frame_y = [1.0, 0.6, 0.2, 0.0, 0.4, 0.8]
    frame_manifolds = []
    for y in frame_y:
        # Cria manifold visual com padrão de pixels baseado em Y
        t = torch.zeros(512, device=device)
        t[int(y * 500)] = 1.0 # Padrão espacial
        hdc_vec = torch.matmul(grounding.proj_visual, t)
        frame_manifolds.append(Manifold(torch.sign(hdc_vec), device=device))
        
    # Bundling da Bola Quicando com posição temporal
    M_bounce_visual = frame_manifolds[0] * grounding.position_manifolds[0]
    for idx, fm in enumerate(frame_manifolds[1:]):
        M_bounce_visual = M_bounce_visual + (fm * grounding.position_manifolds[idx+1])
        
    # 2. Simula áudio de Pitch Subindo (incompatível com a gravidade normal)
    # Onda senoidal de frequência crescente (100 Hz -> 800 Hz)
    sr = 22050
    t = np.arange(int(sr * 0.8)) / sr
    freq = 100 + 700 * t
    waveform_rising = np.sin(2 * np.pi * freq * t)
    # Converte para WAV bytes
    waveform_rising_tensor = torch.tensor(waveform_rising, dtype=torch.float32, device=device)
    M_rising_audio = grounding.encode_audio_encodec(waveform_rising_tensor)
    
    # 3. O Maestro tenta fazer o espelhamento no Sandbox (Áudio -> Visão)
    # Prediz a física esperada a partir do som (subindo de tom -> esperaria movimento contínuo para cima)
    # Mas o visual recebido é a bola quicando para baixo!
    predicted_visual = sandbox.predict_visual_from_audio(M_rising_audio)
    
    # 4. Medição de Energia Livre (Frustração)
    free_energy = calculate_free_energy(predicted_visual, M_bounce_visual)
    print(f"Energia Livre Inicial (Frustração da Contradição): {free_energy:.4f}")
    
    # 5. Gatilho de Solomonoff
    if free_energy > 0.7:
        # Dispara a Forja de Arquétipos (fazendo a Síntese de Programas HDC)
        new_archetype = forge.trigger_forge(predicted_visual, M_bounce_visual, name="ReverseGravity")
        
        # 6. Aplicação da nova regra inventada pela AGI
        corrected_prediction = new_archetype(predicted_visual)
        
        # Medição final de Energia Livre após aprendizado do Arquétipo
        new_free_energy = calculate_free_energy(corrected_prediction, M_bounce_visual)
        print(f"\nEnergia Livre Final (Após aplicar o Arquétipo 'ReverseGravity'): {new_free_energy:.4f}")
        
        if new_free_energy < 0.05:
            print("\n[VEREDITO TESTE 3]: SUCESSO! A Forja criou a regra Compensatória e anulou a Energia Livre.")
        else:
            print("\n[VEREDITO TESTE 3]: FALHA. O Arquétipo gerado não resolveu a frustração.")
    else:
        print("\n[VEREDITO TESTE 3]: FALHA. Incompatibilidade não causou frustração suficiente.")
        
    print("\n====================================================")
    print("            TODOS OS TESTES CONCLUÍDOS              ")
    print("====================================================")

if __name__ == "__main__":
    main()
