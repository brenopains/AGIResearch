import subprocess
import tempfile
import os
import json
import random
import numpy as np
import traceback

YT_DLP_PATH = ('C:\\Users\\usuario\\AppData\\Roaming\\Python\\Python312\\Scripts\\yt-dlp.exe').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))


class YoutubeAdapter:
    """
    Adapter REAL para YouTube via canais da whitelist.
    
    NÃO busca por keyword. NAVEGA os canais curados e assiste o que encontra,
    como um bebê faria — sem saber o que é "gravidade" de antemão.
    
    Pipeline:
      1. Usa yt-dlp para listar vídeos recentes de um canal curado
      2. Escolhe um vídeo
      3. Baixa áudio → Whisper transcreve
      4. Extrai frames → modelo de visão descreve
      5. Retorna grounding multimodal intercalado
    """

    # Mapeamento de nomes amigáveis para URLs/handles dos canais
    CHANNEL_MAP = {
        "Manual do Mundo": "https://www.youtube.com/@manualdomundo/videos",
        "Ciência Todo Dia": "https://www.youtube.com/@CienciaTodoDia/videos",
        "Curso em Vídeo": "https://www.youtube.com/@CursoemVideo/videos",
        "Khan Academy Brasil": "https://www.youtube.com/@khanacademyportugues/videos",
        "MIT OpenCourseWare": "https://www.youtube.com/@mitocw/videos",
        "Khan Academy": "https://www.youtube.com/@khanacademy/videos",
        "3Blue1Brown": "https://www.youtube.com/@3blue1brown/videos",
        "Veritasium": "https://www.youtube.com/@veritasium/videos",
        "Computerphile": "https://www.youtube.com/@Computerphile/videos",
        "AuthenticGames": "https://www.youtube.com/@AuthenticGames/videos",
        "Mogswamp": "https://www.youtube.com/@Mogswamp/videos",
        "ilmango": "https://www.youtube.com/@ilmango/videos",
        "Xisumavoid": "https://www.youtube.com/@Xisumavoid/videos",
    }

    def __init__(self):
        self.tmp_dir = r"D:\novamind_data\temp\novamind_yt"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._whisper_model = None
        self._vision_model = None
        self._vision_processor = None
        self.last_video_title = ""
        self.last_video_url = ""

    def _get_whisper(self):
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            self._whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")
            print("[YT] Whisper loaded (tiny, GPU)")
        return self._whisper_model

    def _get_vision(self):
        if self._vision_model is None:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self._vision_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self._vision_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to("cuda")
                print("[YT] BLIP vision loaded (GPU)")
            except Exception as e:
                print(f"[YT] Vision unavailable: {e}")
                self._vision_model = False
        return self._vision_model

    def _list_channel_videos(self, channel_url: str, max_videos: int = 15) -> list:
        """Lista os vídeos mais recentes de um canal usando yt-dlp."""
        try:
            result = subprocess.run(
                [
                    YT_DLP_PATH, "--flat-playlist",
                    "--playlist-end", str(max_videos),
                    "--print", "%(id)s|%(title)s|%(duration)s",
                    channel_url,
                ],
                capture_output=True, text=True, timeout=30, encoding="utf-8", errors="ignore"
            )
            videos = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 2)
                    if len(parts) >= 2:
                        vid_id = parts[0].strip()
                        title = parts[1].strip()
                        duration = parts[2].strip() if len(parts) > 2 else "?"
                        videos.append({
                            "id": vid_id,
                            "title": title,
                            "duration": duration,
                            "url": f"https://www.youtube.com/watch?v={vid_id}",
                        })
            return videos
        except Exception as e:
            print(f"[YT] Erro listando canal: {e}")
            return []

    def _download_audio(self, url: str) -> str:
        audio_path = os.path.join(self.tmp_dir, "audio")
        # Remove old files
        for f in os.listdir(self.tmp_dir):
            if f.startswith("audio"):
                try: os.remove(os.path.join(self.tmp_dir, f))
                except: pass
        try:
            subprocess.run(
                [
                    YT_DLP_PATH, "-f", "worstaudio",
                    "--extract-audio", "--audio-format", "wav",
                    "--download-sections", "*0:00-3:00",
                    "-o", audio_path,
                    "--force-overwrites", "--no-warnings",
                    url,
                ],
                capture_output=True, timeout=120, text=True, encoding="utf-8", errors="ignore"
            )
            final_path = audio_path + ".wav"
            for candidate in [audio_path + ".wav", audio_path, audio_path + ".wav.wav"]:
                if os.path.exists(candidate):
                    if candidate != final_path:
                        try:
                            if os.path.exists(final_path):
                                os.remove(final_path)
                            os.rename(candidate, final_path)
                        except:
                            pass
                    return final_path
            # Check for any wav in tmp_dir
            for f in os.listdir(self.tmp_dir):
                if f.endswith(".wav"):
                    p = os.path.join(self.tmp_dir, f)
                    if p != final_path:
                        try:
                            if os.path.exists(final_path):
                                os.remove(final_path)
                            os.rename(p, final_path)
                        except:
                            pass
                    return final_path
            return None
        except Exception as e:
            print(f"[YT] Audio download failed: {e}")
            return None

    def _transcribe(self, audio_path: str) -> dict:
        model = self._get_whisper()
        segments, info = model.transcribe(audio_path, language=None)
        text_parts = []
        logprobs = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            logprobs.append(seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0)
        full_text = " ".join(text_parts)[:5000]
        avg_logprob = float(np.mean(logprobs)) if logprobs else 0
        # Convert logprob to confidence [0,1]: higher logprob = higher confidence
        confidence = max(0.1, min(1.0, 1.0 / (1.0 + np.exp(-avg_logprob - 1.0))))
        return {
            "text": full_text,
            "confidence": confidence,
            "avg_logprob": avg_logprob,
            "segments": len(logprobs),
        }

    def _extract_frames(self, url: str, interval_sec: int = 20, max_frames: int = 8) -> list:
        video_path = os.path.join(self.tmp_dir, "video.mp4")
        frames_dir = os.path.join(self.tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for f in os.listdir(frames_dir):
            try: os.remove(os.path.join(frames_dir, f))
            except: pass
        try:
            subprocess.run(
                [
                    YT_DLP_PATH, "-f", "worst[ext=mp4]/worst",
                    "--download-sections", "*0:00-3:00",
                    "-o", video_path, "--force-overwrites", "--no-warnings",
                    url,
                ],
                capture_output=True, timeout=180, text=True, encoding="utf-8", errors="ignore"
            )
            if not os.path.exists(video_path):
                return []
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"fps=1/{interval_sec}",
                    "-frames:v", str(max_frames),
                    os.path.join(frames_dir, "frame_%03d.jpg"),
                ],
                capture_output=True, timeout=60
            )
            frames = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith(".jpg")
            ])
            # Cleanup video
            try: os.remove(video_path)
            except: pass
            return frames[:max_frames]
        except Exception as e:
            print(f"[YT] Frame extraction failed: {e}")
            return []

    def _describe_frames(self, frame_paths: list) -> list:
        model = self._get_vision()
        if model is False or model is None:
            return [f"[Visual t={i*20}s]: (vision unavailable)" for i in range(len(frame_paths))]
        from PIL import Image
        descriptions = []
        for i, path in enumerate(frame_paths):
            try:
                img = Image.open(path).convert("RGB").resize((384, 384))
                inputs = self._vision_processor(images=img, return_tensors="pt").to("cuda")
                out = model.generate(**inputs, max_new_tokens=50)
                caption = self._vision_processor.decode(out[0], skip_special_tokens=True)
                descriptions.append(f"[Visual t={i*20}s]: {caption}")
            except Exception as e:
                traceback.print_exc()
                descriptions.append(f"[Visual t={i*20}s]: (erro: {e})")
        return descriptions

    def query(self, topic: str, source: str = "Manual do Mundo") -> dict:
        """
        NAVEGA um canal curado e assiste um vídeo.
        'topic' é ignorado para a seleção — a AGI assiste o que encontra,
        como um bebê que não sabe pesquisar.

        Retorna dict com:
          - content: str (texto combinado Whisper+BLIP — muleta)
          - frames_raw: list[bytes] (JPEG bruto de cada frame)
          - audio_raw: np.ndarray (waveform mono)
          - audio_sample_rate: int
        """
        channel_url = self.CHANNEL_MAP.get(source)
        if not channel_url:
            channel_url = self.CHANNEL_MAP.get("Manual do Mundo")

        print(f"[YT] Navegando canal: {source}")
        videos = self._list_channel_videos(channel_url, max_videos=15)
        if not videos:
            return {"content": f"[YouTube] Nenhum vídeo encontrado no canal {source}.",
                    "frames_raw": [], "audio_raw": None, "audio_sample_rate": 16000}

        # Pick a random video (like a baby channel-surfing)
        video = random.choice(videos)
        url = video["url"]
        title = video["title"]
        self.last_video_title = title
        self.last_video_url = url

        print(f"[YT] Assistindo: {title}")
        result_parts = [f"[Assistindo Vídeo: {title}] (Canal: {source})"]

        # RAW AUDIO (para o córtex auditivo da AGI)
        audio_raw = None
        audio_sr = 16000
        audio_path = None

        # AUDIO → Whisper → Text (muleta)
        print(f"[YT] Baixando e transcrevendo áudio...")
        audio_path = self._download_audio(url)
        transcription_data = {"text": "", "confidence": 0.0}
        if audio_path:
            transcription_data = self._transcribe(audio_path)
            try:
                import librosa
                audio_raw, audio_sr = librosa.load(audio_path, sr=16000, mono=True)
            except:
                pass

        transcription = transcription_data.get("text", "")
        confidence = transcription_data.get("confidence", 0.0)

        # RAW FRAMES (para o córtex visual da AGI)
        frames_raw = []
        frame_paths = []

        # VIDEO → Frames → BLIP → Text (muleta)
        print(f"[YT] Extraindo e descrevendo frames...")
        frame_paths = self._extract_frames(url)
        frame_descriptions = []
        if frame_paths:
            for fp in frame_paths:
                try:
                    with open(fp, 'rb') as f:
                        frames_raw.append(f.read())
                except:
                    pass
            frame_descriptions = self._describe_frames(frame_paths)

        # Include confidence info in result (for uncertainty quantification)
        uncertainty_tag = f"[AudioConfidence: {confidence:.2f}]"
        result_parts.append(uncertainty_tag)

        # INTERLEAVE: visual + audio descriptions together
        if transcription:
            words = transcription.split()
            chunk_size = max(1, len(words) // max(1, len(frame_descriptions)))
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i+chunk_size]))

            for i, desc in enumerate(frame_descriptions):
                result_parts.append(desc)
                if i < len(chunks):
                    result_parts.append(f"[Audio t={i*20}s conf={confidence:.2f}]: {chunks[i]}")

            for j in range(len(frame_descriptions), len(chunks)):
                result_parts.append(f"[Audio conf={confidence:.2f}]: {chunks[j]}")
        else:
            result_parts.extend(frame_descriptions)
            if not frame_descriptions:
                result_parts.append("[Sem áudio ou vídeo extraível]")

        return {
            "content": "\n".join(result_parts),
            "frames_raw": frames_raw,
            "audio_raw": audio_raw,
            "audio_sample_rate": audio_sr,
        }

    def cleanup_temp_files(self):
        """Remove all downloaded temp files to avoid storage buildup. Call AFTER video processing is complete."""
        try:
            import shutil
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
            print(f"[YT] Temp files cleaned up")
        except Exception as e:
            print(f"[YT] Cleanup error: {e}")
