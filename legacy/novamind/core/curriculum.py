"""
Curriculum Trainer — Desenvolvimento por etapas da AGI.
"""
import os, sys, time, json, math, random, csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchaudio
from PIL import Image
import numpy as np

from .memory import MemorySystem
from .memory_persistence import save_full_backup, load_full_backup, restore_memory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COCO_ROOT = r"D:\AGI2\datasets\coco2017"
FEATURE_DIR = r"D:\AGI2\treino"
MEMORY_DIR = r"D:\AGI2\memories"
PROGRESS_FILE = os.path.join(MEMORY_DIR, "curriculum_progress.json")
SAVE_INTERVAL = 1000
os.makedirs(MEMORY_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(os.path.join(MEMORY_DIR, "curriculum_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


STAGES = [
    {"id": 0, "name": "Sensorial - Visao", "desc": "Exposicao a imagens COCO sem rotulos.",
     "datasets": ["coco_vision"], "batch_size": 64, "total": 50000},
    {"id": 1, "name": "Sensorial - Video", "desc": "Exposicao a videos curtos (movimento, objetos em acao).",
     "datasets": ["ucf101_raw"], "batch_size": 8, "total": 5000},
    {"id": 2, "name": "Sensorial - Audio", "desc": "Exposicao a sons do ESC-50.",
     "datasets": ["esc50"], "batch_size": 32, "total": 2000},
    {"id": 3, "name": "Sensorial - Fala", "desc": "Exposicao a fala do LibriSpeech.",
     "datasets": ["librispeech_sample"], "batch_size": 16, "total": 500},
    {"id": 4, "name": "Sensorial - Video+Audio", "desc": "Videos com audio sincrono.",
     "datasets": ["audio_video"], "batch_size": 4, "total": 1000},
    {"id": 5, "name": "Infantil - Objetos+Nomes", "desc": "COCO com legendas. Objeto -> nome.",
     "datasets": ["coco_captions"], "batch_size": 64, "total": 50000},
    {"id": 6, "name": "Linguagem - Alfabeto", "desc": "Letras, numeros, simbolos.",
     "datasets": ["mnist", "hasy", "alphabet_texts"], "batch_size": 64, "total": 10000},
    {"id": 7, "name": "Linguagem - Palavras e Frases", "desc": "Vocabulario infantil, frases simples.",
     "datasets": ["childes", "bnc_child"], "batch_size": 32, "total": 50000},
    {"id": 8, "name": "Fundamental - Ciencias", "desc": "Causalidade, matematica basica, ciencia.",
     "datasets": ["cifar10", "sci_questions"], "batch_size": 64, "total": 60000},
    {"id": 9, "name": "Medio - Abstração", "desc": "Fisica, quimica, algebra.",
     "datasets": ["textbooks_middle"], "batch_size": 16, "total": 20000},
    {"id": 10, "name": "Superior - Especializacao", "desc": "Papers, codigo, conhecimento avancado.",
     "datasets": ["arxiv", "github", "textbooks_advanced"], "batch_size": 8, "total": 50000},
]


class CurriculumTrainer:
    def __init__(self):
        self.memory = MemorySystem()
        self.stage_progress: Dict[str, int] = {}
        self.current_stage = 0
        self._resnet = None
        self._feat_ext = None
        self._proj = None
        self._save_counter = 0

    def resume_or_start(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
            self.stage_progress = {str(k): v for k, v in data.get("stage_progress", {}).items()}
            self.current_stage = int(data.get("current_stage", 0))
            state = load_full_backup()
            if state:
                restore_memory(self.memory, state[0], state[1])
                log(f"Restaurado: {len(self.memory.episodic)} episodios, "
                    f"{len(self.memory.semantic)} regras semanticas")
        else:
            log("Comecando do zero.")
        log(f"Curriculum na etapa {self.current_stage}")

    def save_progress(self, *, items_done=0, total=0, force=False):
        """Salva a cada 5000 itens ou no final."""
        if not force and (items_done % 5000 != 0 or items_done == 0) and items_done < total:
            return
        data = {"current_stage": self.current_stage,
                "stage_progress": self.stage_progress, "timestamp": time.time()}
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=1)
        save_full_backup(self.memory)

    def _init_vision(self):
        if self._resnet is None:
            import torchvision.models as models
            self._resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
            self._resnet.eval()
            self._feat_ext = nn.Sequential(*list(self._resnet.children())[:-1])
            torch.manual_seed(42)
            self._proj = torch.randn(10000, 512, device=DEVICE) / np.sqrt(512)
            log("ResNet18 + projecao fixa carregadas")

    def _hdv_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() > 1 and feats.shape[0] == 1:
            feats = feats.squeeze(0)
        bipolar = torch.sign(self._proj @ feats)
        bipolar[bipolar == 0] = 1
        return bipolar

    def run_stage(self, stage_id: int):
        stage = STAGES[stage_id]
        sid = str(stage_id)
        processed = self.stage_progress.get(sid, 0)
        if processed >= stage["total"]:
            log(f"  Etapa {stage_id} ja completa ({processed}/{stage['total']})")
            return True

        self.current_stage = stage_id
        log(f"\n{'='*60}")
        log(f"ETAPA {stage_id}: {stage['name']}")
        log(f"  {stage['desc']}")
        log(f"  {stage['total']} itens | processados: {processed}")
        log(f"{'='*60}")

        for ds_name in stage["datasets"]:
            handler = getattr(self, f"_process_{ds_name}", None)
            if handler is None:
                log(f"  [PLACEHOLDER] {ds_name} - dataset nao disponivel ainda")
                continue
            handler(batch_size=stage["batch_size"])

        self.stage_progress[sid] = stage["total"]
        self.save_progress(force=True)
        log(f"ETAPA {stage_id} CONCLUIDA!\n")

    def run_all(self):
        self.resume_or_start()
        for s in STAGES:
            self.run_stage(s["id"])

    # ─── Handlers ────────────────────────────────────────────

    def _process_coco_vision(self, batch_size=64):
        st = STAGES[0]
        sid = "0"
        processed = self.stage_progress.get(sid, 0)
        remaining = st["total"] - processed
        if remaining <= 0:
            return

        img_dir = os.path.join(COCO_ROOT, "train2017")
        files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        files = files[processed:processed + remaining]
        log(f"  Processando {len(files)} imagens...")

        self._init_vision()
        tf = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        batch, paths = [], []
        count = processed
        start = time.time()
        last_log = 0

        for idx, fname in enumerate(files):
            path = os.path.join(img_dir, fname)
            img = Image.open(path).convert("RGB")
            batch.append(tf(img))
            paths.append(path)

            if len(batch) == batch_size or idx == len(files) - 1:
                x = torch.stack(batch).to(DEVICE)
                with torch.no_grad():
                    feats = self._feat_ext(x).squeeze(-1).squeeze(-1)

                for j in range(len(batch)):
                    hdv = self._hdv_from_features(feats[j])
                    ep = self.memory.store_episode(
                        {"input": f"imagem_{fname}", "modality": "vision",
                         "path": paths[j], "features": feats[j].cpu().tolist()},
                        "perception_vision", emotional_valence=0.1, surprise=0.05)
                    ep.hdc_vector = hdv.cpu().numpy()
                    count += 1

                batch, paths = [], []
                done = count - processed
                if done - last_log >= 500 or count >= st["total"]:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (st["total"] - count) / rate if rate > 0 else 0
                    vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    log(f"  {count}/{st['total']} ({rate:.0f} img/s, ~{remaining:.0f}s rest, VRAM={vram:.1f}GB)")
                    self.stage_progress[sid] = count
                    self.save_progress(items_done=count, total=st["total"])
                    last_log = done

        self.stage_progress[sid] = st["total"]
        self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios")

    # ─── Audio (ESC-50) ─────────────────────────────────────

    def _process_esc50(self, batch_size=32):
        st = STAGES[2]
        sid = "2"
        processed = self.stage_progress.get(sid, 0)
        remaining = st["total"] - processed
        if remaining <= 0:
            return

        audio_dir = r"D:\AGI2\datasets\ESC-50-master\audio"
        meta_path = r"D:\AGI2\datasets\ESC-50-master\meta\esc50.csv"
        files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        files = files[processed:processed + remaining]

        log(f"  Processando {len(files)} sons ESC-50...")

        # Mel transform
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=64
        ).to(DEVICE)

        # Projecao audio: 64 × 87 (media temporal) = 5568D
        torch.manual_seed(43)
        proj_audio = torch.randn(10000, 64, device=DEVICE) / 8  # media temporal reduz pra 64D

        count = processed
        start = time.time()
        last_log = 0

        for idx, fname in enumerate(files):
            try:
                path = os.path.join(audio_dir, fname)
                wav, sr = torchaudio.load(path)
                if sr != 22050:
                    wav = torchaudio.functional.resample(wav, sr, 22050)
                # Mono
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)

                # Mel spectrogram
                mel = mel_tf(wav.to(DEVICE))  # (1, 64, T)
                # Media temporal
                feat = mel.mean(dim=-1).squeeze(0)  # (64,)

                # HDV
                hdv = torch.sign(proj_audio @ feat)
                hdv[hdv == 0] = 1

                ep = self.memory.store_episode(
                    {"input": f"som_{fname}", "modality": "audio",
                     "path": path, "file": fname},
                    "perception_audio", emotional_valence=0.15, surprise=0.1)
                ep.hdc_vector = hdv.cpu().numpy()
                count += 1
            except Exception as e:
                log(f"  AVISO: pulando {fname}: {e}")
                continue

            if count % 100 == 0 or count >= st["total"]:
                elapsed = time.time() - start
                rate = (count - processed) / elapsed if elapsed > 0 else 0
                log(f"  {count}/{st['total']} ({rate:.0f} aud/s)")
                self.stage_progress[sid] = count
                self.save_progress(items_done=count, total=st["total"])

        self.stage_progress[sid] = st["total"]
        self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios audio")

    # ─── Video (UCF-101 real) ──────────────────────────────

    def _process_ucf101_raw(self, batch_size=4):
        """Video real: frames do UCF-101 processados em sequencia."""
        st = STAGES[1]
        sid = "1"
        processed = self.stage_progress.get(sid, 0)
        remaining = st["total"] - processed
        if remaining <= 0:
            return

        ucf_dir = r"D:\AGI2\datasets\UCF101\UCF101\UCF-101"
        categories = sorted([d for d in os.listdir(ucf_dir) if os.path.isdir(os.path.join(ucf_dir, d))])
        all_videos = []
        for cat in categories:
            cat_dir = os.path.join(ucf_dir, cat)
            for f in os.listdir(cat_dir):
                if f.lower().endswith((".avi", ".mp4")):
                    all_videos.append(os.path.join(cat_dir, f))
        random.shuffle(all_videos)
        all_videos = all_videos[:remaining]
        log(f"  Processando {len(all_videos)} videos UCF-101...")

        self._init_vision()
        tf = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        count = processed
        start = time.time()
        last_log = 0

        for vid_path in all_videos:
            try:
                import cv2
                cap = cv2.VideoCapture(vid_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 5:
                    cap.release()
                    continue

                frame_idxs = [int(i * total_frames / 5) for i in range(5)]
                frames = []
                for fi in frame_idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        frames.append(tf(img))
                cap.release()

                if len(frames) < 3:
                    continue

                x = torch.stack(frames).to(DEVICE)
                with torch.no_grad():
                    feats = self._feat_ext(x).squeeze(-1).squeeze(-1)

                frame_hdvs = []
                for j in range(len(feats)):
                    hdv = self._hdv_from_features(feats[j])
                    frame_hdvs.append(hdv.cpu().numpy())

                avg_hdv = np.sign(np.mean(frame_hdvs, axis=0))
                avg_hdv[avg_hdv == 0] = 1

                rel_path = os.path.relpath(vid_path, ucf_dir)
                ep = self.memory.store_episode(
                    {"input": f"video_{rel_path}", "modality": "video",
                     "path": vid_path, "category": os.path.basename(os.path.dirname(vid_path)),
                     "n_frames": len(frame_hdvs), "total_frames": total_frames},
                    "perception_video", emotional_valence=0.15, surprise=0.2)
                ep.hdc_vector = avg_hdv
                count += 1

            except Exception as e:
                log(f"  AVISO: pulando {vid_path}: {e}")
                continue

            if count % 100 == 0 or count >= st["total"]:
                elapsed = time.time() - start
                rate = (count - processed) / elapsed if elapsed > 0 else 0
                vram = torch.cuda.memory_allocated() / 1024**3
                log(f"  {count}/{st['total']} videos ({rate:.0f} vid/s, VRAM={vram:.1f}GB)")
                self.stage_progress[sid] = count
                self.save_progress(items_done=count, total=st["total"])

            if count >= st["total"]:
                break

        self.stage_progress[sid] = st["total"]
        self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios video")

    # ─── Fala (LibriSpeech) ────────────────────────────────

    def _process_librispeech_sample(self, batch_size=16):
        """Fala: amostras do LibriSpeech via torchaudio."""
        st = STAGES[3]
        sid = "3"
        processed = self.stage_progress.get(sid, 0)
        remaining = st["total"] - processed
        if remaining <= 0:
            return

        libri_root = r"D:\AGI2\datasets\LibriSpeech"
        log(f"  Carregando LibriSpeech dev-clean em {libri_root}...")
        try:
            dataset = torchaudio.datasets.LIBRISPEECH(
                libri_root, url="dev-clean", download=True)
        except Exception as e:
            log(f"  ERRO ao carregar LibriSpeech: {e}. Pulando.")
            return

        # Pular ja processados
        idxs = list(range(processed, min(processed + remaining, len(dataset))))
        log(f"  Processando {len(idxs)} amostras de fala...")

        # Mel transform para features
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=128, n_mels=64
        ).to(DEVICE)
        torch.manual_seed(44)
        proj_audio = torch.randn(10000, 64, device=DEVICE) / 8

        count = processed
        start = time.time()
        last_log = 0

        for i, idx in enumerate(idxs):
            try:
                waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[idx]
                # Resample to 16k if needed
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                    sample_rate = 16000

                # Mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)

                # Mel
                mel = mel_tf(waveform.to(DEVICE))  # (1, 64, T)
                feat = mel.mean(dim=-1).squeeze(0)  # (64,)

                # HDV
                hdv = torch.sign(proj_audio @ feat)
                hdv[hdv == 0] = 1

                ep = self.memory.store_episode(
                    {"input": transcript, "modality": "speech",
                     "transcript": transcript, "speaker": speaker_id,
                     "utterance": f"{speaker_id}-{chapter_id}-{utterance_id}"},
                    "perception_speech", emotional_valence=0.15, surprise=0.15)
                ep.hdc_vector = hdv.cpu().numpy()
                count += 1

            except Exception as e:
                log(f"  AVISO: pulando {idx}: {e}")
                continue

            if (i + 1) % 50 == 0 or i == len(idxs) - 1:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                log(f"  {count}/{st['total']} ({rate:.0f} aud/s)")
                self.stage_progress[sid] = count
                self.save_progress(items_done=count, total=st["total"])

        self.stage_progress[sid] = st["total"]
        self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios fala")
        """Video real: frames do UCF-101 processados em sequencia."""
        st = STAGES[1]
        sid = "1"
        processed = self.stage_progress.get(sid, 0)
        remaining = st["total"] - processed
        if remaining <= 0:
            return

        ucf_dir = r"D:\AGI2\datasets\UCF101\UCF101\UCF-101"
        categories = sorted([d for d in os.listdir(ucf_dir) if os.path.isdir(os.path.join(ucf_dir, d))])
        all_videos = []
        for cat in categories:
            cat_dir = os.path.join(ucf_dir, cat)
            for f in os.listdir(cat_dir):
                if f.lower().endswith((".avi", ".mp4")):
                    all_videos.append(os.path.join(cat_dir, f))
        random.shuffle(all_videos)
        all_videos = all_videos[:remaining]
        log(f"  Processando {len(all_videos)} videos UCF-101...")

        self._init_vision()
        tf = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        count = processed
        start = time.time()
        last_log = 0

        for vid_path in all_videos:
            try:
                import cv2
                cap = cv2.VideoCapture(vid_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 5:
                    cap.release()
                    continue

                frame_idxs = [int(i * total_frames / 5) for i in range(5)]
                frames = []
                for fi in frame_idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        frames.append(tf(img))
                cap.release()

                if len(frames) < 3:
                    continue

                x = torch.stack(frames).to(DEVICE)
                with torch.no_grad():
                    feats = self._feat_ext(x).squeeze(-1).squeeze(-1)

                frame_hdvs = []
                for j in range(len(feats)):
                    hdv = self._hdv_from_features(feats[j])
                    frame_hdvs.append(hdv.cpu().numpy())

                avg_hdv = np.sign(np.mean(frame_hdvs, axis=0))
                avg_hdv[avg_hdv == 0] = 1

                rel_path = os.path.relpath(vid_path, ucf_dir)
                ep = self.memory.store_episode(
                    {"input": f"video_{rel_path}", "modality": "video",
                     "path": vid_path, "category": os.path.basename(os.path.dirname(vid_path)),
                     "n_frames": len(frame_hdvs), "total_frames": total_frames},
                    "perception_video", emotional_valence=0.15, surprise=0.2)
                ep.hdc_vector = avg_hdv
                count += 1

            except Exception as e:
                log(f"  AVISO: pulando {vid_path}: {e}")
                continue

            if count % 100 == 0 or count >= st["total"]:
                elapsed = time.time() - start
                rate = (count - processed) / elapsed if elapsed > 0 else 0
                vram = torch.cuda.memory_allocated() / 1024**3
                log(f"  {count}/{st['total']} videos ({rate:.0f} vid/s, VRAM={vram:.1f}GB)")
                self.stage_progress[sid] = count
                self.save_progress(items_done=count, total=st["total"])

            if count >= st["total"]:
                break

        self.stage_progress[sid] = st["total"]
        self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios video")

    def _process_coco_captions(self, batch_size=64):
        st = STAGES[5]
        sid = "5"
        processed = self.stage_progress.get(sid, 0)
        remaining = st["total"] - processed
        if remaining <= 0:
            return

        caps_path = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")
        with open(caps_path) as f:
            caps_data = json.load(f)
        cap_map = {}
        for ann in caps_data["annotations"]:
            iid = ann["image_id"]
            cap_map.setdefault(iid, []).append(ann["caption"])
        img_map = {img["file_name"]: img["id"] for img in caps_data["images"]}

        img_dir = os.path.join(COCO_ROOT, "train2017")
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        files = files[processed:processed + remaining]
        log(f"  Processando {len(files)} imagens com legendas...")

        self._init_vision()
        tf = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        batch, paths, caps = [], [], []
        count = processed
        start = time.time()
        last_log = 0

        for idx, fname in enumerate(files):
            path = os.path.join(img_dir, fname)
            img = Image.open(path).convert("RGB")
            batch.append(tf(img))
            paths.append(path)
            iid = img_map.get(fname)
            c = cap_map.get(iid, ["sem legenda"])[0]
            caps.append(c)

            if len(batch) == batch_size or idx == len(files) - 1:
                x = torch.stack(batch).to(DEVICE)
                with torch.no_grad():
                    feats = self._feat_ext(x).squeeze(-1).squeeze(-1)
                for j in range(len(batch)):
                    hdv = self._hdv_from_features(feats[j])
                    ep = self.memory.store_episode(
                        {"input": caps[j], "modality": "vision_language",
                         "path": paths[j], "caption": caps[j],
                         "features": feats[j].cpu().tolist()},
                        "perception_caption", emotional_valence=0.2, surprise=0.1)
                    ep.hdc_vector = hdv.cpu().numpy()
                    count += 1

                batch, paths, caps = [], [], []
                done = count - processed
                if done - last_log >= 500 or count >= st["total"]:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (st["total"] - count) / rate if rate > 0 else 0
                    vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    log(f"  {count}/{st['total']} ({rate:.0f} img/s, ~{remaining:.0f}s rest, VRAM={vram:.1f}GB)")
                    self.stage_progress[sid] = count
                    self.save_progress(items_done=count, total=st["total"])
                    last_log = done

        self.stage_progress[sid] = st["total"]
        self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios")
    def _text_to_hdv(self, text):
        torch.manual_seed(45)
        vs = 5000
        if not hasattr(self, "_proj_text"):
            self._proj_text = torch.randn(10000, vs, device=DEVICE) / np.sqrt(vs)
        words = str(text).lower().split()
        bow = torch.zeros(vs, device=DEVICE)
        for w in words:
            bow[hash(w) % vs] += 1.0
        hdv = torch.sign(self._proj_text @ (bow / (bow.norm() + 1e-8)))
        hdv[hdv == 0] = 1
        return hdv.cpu().numpy()

    def _process_mnist(self, batch_size=64):
        st = STAGES[6]; sid = "6"
        if self.stage_progress.get(sid, 0) >= st["total"]: return
        import torchvision.datasets as dset
        ds = dset.MNIST(r"D:\AGI2\datasets\mnist", train=True, download=True,
                        transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        log(f"  Processando {st['total']} digitos MNIST...")
        self._init_vision(); c = self.stage_progress.get(sid, 0); start = time.time()
        for i in range(min(st["total"], len(ds))):
            img, lb = ds[i]
            t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img.repeat(3,1,1).unsqueeze(0).to(DEVICE))
            with torch.no_grad():
                feats = self._feat_ext(t).flatten(1)
            h = self._hdv_from_features(feats).cpu().numpy()
            th = self._text_to_hdv(f"digito {lb}")
            h = np.sign(h + th); h[h == 0] = 1
            ep = self.memory.store_episode({"input": f"digito {lb}", "modality": "vision_text", "label": int(lb)}, "perception_digit")
            ep.hdc_vector = h; c += 1
            if c % 1000 == 0 or c >= st["total"]:
                log(f"  {c}/{st['total']} ({(c-self.stage_progress.get(sid,0))/(time.time()-start):.0f} img/s)")
                self.stage_progress[sid] = c; self.save_progress(items_done=c, total=st["total"])
        self.stage_progress[sid] = st["total"]; self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios MNIST")

    def _process_childes(self, batch_size=32):
        st = STAGES[7]; sid = "7"
        if self.stage_progress.get(sid, 0) >= st["total"]: return
        sf = r"D:\AGI2\datasets\stories\cleaned_merged_fairy_tales_without_eos.txt"
        if not os.path.exists(sf): log(f"  ERRO: {sf}"); return
        with open(sf, encoding="utf-8", errors="ignore") as f:
            sents = [s.strip() for s in f.read().replace("\n"," ").split(".") if len(s.strip()) > 10]
        sents = sents[:st["total"]]
        log(f"  Processando {len(sents)} frases...")
        c = 0; start = time.time()
        for s in sents:
            ep = self.memory.store_episode({"input": s, "modality": "text", "source": "fairy_tales"}, "perception_text")
            ep.hdc_vector = self._text_to_hdv(s); c += 1
            if c % 1000 == 0 or c >= st["total"]:
                log(f"  {c}/{st['total']} ({c/(time.time()-start):.0f} txt/s)")
        self.stage_progress[sid] = st["total"]; self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios texto")
    def _process_bnc_child(self, batch_size=32): self._process_childes(batch_size)

    def _process_alphabet_texts(self, batch_size=64):
        st = STAGES[6]; sid = "6"
        if self.stage_progress.get(sid, 0) >= st["total"]: return
        sym = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.,:;")
        texts = sym + [f"numero {d}" for d in "0123456789"] + [f"letra {l}" for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        log(f"  Processando {len(texts)} simbolos...")
        c = self.stage_progress.get(sid, 0)
        for t in texts:
            ep = self.memory.store_episode({"input": t, "modality": "text", "source": "alphabet"}, "perception_symbol")
            ep.hdc_vector = self._text_to_hdv(t); c += 1
        self.stage_progress[sid] = st["total"]; self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios simbolos")

    def _process_sci_questions(self, batch_size=32):
        st = STAGES[8]; sid = "8"
        if self.stage_progress.get(sid, 0) >= st["total"]: return
        facts = ["agua ferve a 100 graus","agua congela a 0 graus","terra gira em torno do sol","lua gira em torno da terra",
                 "plantas precisam de luz","gravidade atrai objetos","eletricidade e fluxo de eletrons","atomo tem nucleo e eletrons",
                 "dna e o codigo genetico","celulas sao unidade basica da vida","energia nao se cria nem se destroe",
                 "velocidade da luz e 300 mil km/s","iman tem dois polos","toda acao tem uma reacao igual e oposta"]
        log(f"  Processando {len(facts)} fatos...")
        c = self.stage_progress.get(sid, 0)
        for t in facts:
            ep = self.memory.store_episode({"input": t, "modality": "text", "source": "science"}, "perception_science")
            ep.hdc_vector = self._text_to_hdv(t); c += 1
        self.stage_progress[sid] = st["total"]; self.save_progress(force=True)
        log(f"  OK: {st['total']} episodios ciencia")

