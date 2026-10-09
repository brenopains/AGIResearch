import numpy as np
import traceback
from typing import Dict, Optional, List, Tuple


class AuditoryCortex:
    """
    Córtex Auditivo — processa WAVEFORM BRUTO com via DUAL (dorsal + ventral).

    Baseado no modelo de dois fluxos (Hickok & Poeppel 2007):
      ─ Via Ventral (what):  som → significado (compreensão)
      ─ Via Dorsal (where/how): som → articulação (produção)

    A via DORSAL é o que permite à AGI aprender a imitar sons:
    "Ouvi 'ba' → que articuladores preciso mover para fazer 'ba'?"
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.last_features: Dict = {}
        self.last_self_description: str = ""
        self.last_dorsal_description: str = ""
        self.total_segments_processed = 0
        self._prev_spectrum: Optional[np.ndarray] = None
        self.stream_buffer: List[np.ndarray] = []
        self.max_buffer_duration = 3.0  # segundos de buffer contínuo

        # Corollary Discharge: cópia do comando motor para suprimir auto-audição
        self._expected_self_sound: Optional[Dict] = None

    def expect_self_sound(self, f0: float, f1: float, f2: float, f3: float):
        """
        Recebe a cópia de eferência do comando motor ANTES da execução.

        No cérebro humano, o córtex motor envia uma cópia do comando
        para o córtex auditivo que PREDIZ o som esperado.
        O som REAL é então comparado com a predição, e apenas
        o RESÍDUO (erro de predição) é processado como informação nova.

        Isso permite distinguir sons auto-gerados de sons externos.
        (Khalilian-Gourtani et al. 2024, PMC — evidência direta em humanos)
        """
        self._expected_self_sound = {
            "f0": f0, "f1": f1, "f2": f2, "f3": f3,
            "timestamp": __import__("time").time(),
        }

    def clear_self_sound_expectation(self):
        """Limpa a expectativa (som não era auto-gerado)."""
        self._expected_self_sound = None

    def process(self, audio_samples: np.ndarray,
                sample_rate: Optional[int] = None) -> Dict:
        """
        Processa waveform bruto e retorna features auditivas.

        Args:
            audio_samples: array 1D float32 [-1, 1]
            sample_rate: taxa do áudio (default: 16000)
        """
        try:
            sr = sample_rate or self.sample_rate
            if len(audio_samples) == 0:
                return self._empty_result()

            if sr != sr:
                sr = self.sample_rate

            audio = audio_samples.astype(np.float64)
            self.total_segments_processed += 1

            features = {}

            # 1. Loudness (RMS envelope)
            frame_len = int(sr * 0.025)
            if frame_len < 1:
                frame_len = 1
            hop = frame_len // 2
            frames = []
            for start in range(0, len(audio) - frame_len + 1, hop):
                frames.append(np.sqrt(np.mean(audio[start:start+frame_len]**2)))
            if frames:
                rms = np.array(frames)
            else:
                rms = np.array([0.0])
            features['loudness'] = {
                'mean': float(np.mean(rms)),
                'peak': float(np.max(rms)),
                'dynamic_range': float(np.std(rms) / (np.mean(rms) + 1e-10)),
            }

            # Silêncio?
            is_silent = features['loudness']['mean'] < 0.01

            # 2. Espectro de frequências (FFT)
            n_fft = min(2048, len(audio))
            if len(audio) > n_fft:
                spectrum = np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft)))
            else:
                spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
            freqs = np.fft.rfftfreq(n_fft, d=1/sr)

            if len(spectrum) > 1:
                # Bandas: baixa (0-300Hz), média (300-3000Hz), alta (3000+)
                low_mask = freqs < 300
                mid_mask = (freqs >= 300) & (freqs < 3000)
                high_mask = freqs >= 3000

                # Evitar bandas vazias
                low_energy = float(np.sum(spectrum[low_mask])) if np.any(low_mask) else 0
                mid_energy = float(np.sum(spectrum[mid_mask])) if np.any(mid_mask) else 0
                high_energy = float(np.sum(spectrum[high_mask])) if np.any(high_mask) else 0
                total = low_energy + mid_energy + high_energy + 1e-10

                features['bands'] = {
                    'low': low_energy / total,
                    'mid': mid_energy / total,
                    'high': high_energy / total,
                }

                # Centróide espectral
                centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
                features['spectral_centroid'] = float(centroid)

                # Rolloff (frequência abaixo da qual está 85% da energia)
                cumsum = np.cumsum(spectrum)
                total_e = cumsum[-1] + 1e-10
                rolloff_idx = np.searchsorted(cumsum, 0.85 * total_e)
                features['spectral_rolloff'] = float(freqs[min(rolloff_idx, len(freqs)-1)])

                # Mudança espectral (onset espectral)
                if self._prev_spectrum is not None and len(self._prev_spectrum) == len(spectrum):
                    spec_diff = np.mean(np.abs(spectrum - self._prev_spectrum))
                    features['spectral_flux'] = float(spec_diff / (np.mean(spectrum) + 1e-10))
                else:
                    features['spectral_flux'] = 0.0

                self._prev_spectrum = spectrum.copy()
            else:
                features['bands'] = {'low': 0.33, 'mid': 0.34, 'high': 0.33}
                features['spectral_centroid'] = 1000.0
                features['spectral_rolloff'] = 2000.0
                features['spectral_flux'] = 0.0

            # 3. Pitch (F0) via autocorrelação
            if not is_silent and len(audio) > sr * 0.02:
                y = audio - np.mean(audio)
                corr = np.correlate(y, y, mode='full')
                mid = len(corr) // 2
                search_start = int(sr / 500)
                search_end = int(sr / 50)
                segment = corr[mid+search_start:mid+search_end]
                if len(segment) > 0:
                    peak_idx = np.argmax(segment)
                    confidence = float(segment[peak_idx] / (segment[0] + 1e-10))
                    f0 = sr / (search_start + peak_idx + 1)
                    if 50 < f0 < 500 and confidence > 0.1:
                        features['pitch'] = {
                            'detected': True,
                            'f0': float(f0),
                            'confidence': min(1.0, confidence),
                        }
                    else:
                        features['pitch'] = {'detected': False, 'f0': 0, 'confidence': 0}
                else:
                    features['pitch'] = {'detected': False, 'f0': 0, 'confidence': 0}
            else:
                features['pitch'] = {'detected': False, 'f0': 0, 'confidence': 0}

            # 4. Presença de voz (energia em banda de voz: 80-4000Hz + modulação rítmica)
            voice_mask = (freqs >= 80) & (freqs <= 4000)
            if np.any(voice_mask) and 'bands' in features:
                voice_ratio = features['bands']['mid'] + features['bands']['low'] * 0.5
                has_harmonics = features.get('pitch', {}).get('detected', False)
                features['voice_presence'] = min(1.0, voice_ratio * 1.5 + (0.4 if has_harmonics else 0))
            else:
                features['voice_presence'] = 0.5

            # 5. Onsets (transientes)
            if len(audio) > sr * 0.1:
                diff_env = np.abs(np.diff(rms)) if len(rms) > 1 else np.array([0])
                onset_thresh = np.mean(diff_env) + 2 * np.std(diff_env)
                onsets = np.where(diff_env > onset_thresh)[0]
                features['onsets'] = {
                    'count': int(len(onsets)),
                    'density': float(len(onsets) / (len(audio) / sr + 0.01)),
                }
            else:
                features['onsets'] = {'count': 0, 'density': 0}

            # 6. Complexidade espectral (entropia)
            if len(spectrum) > 1:
                spec_norm = spectrum / (np.sum(spectrum) + 1e-10)
                spec_entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10))
                max_ent = np.log2(len(spec_norm))
                features['spectral_entropy'] = min(1.0, spec_entropy / max_ent)
            else:
                features['spectral_entropy'] = 0.5

            self.last_features = features

            # ─── Corollary Discharge ────────────────────────────────
            # Subtrai a predição do comando motor do som real
            cd_result = None
            if self._expected_self_sound is not None:
                exp = self._expected_self_sound
                actual_pitch = features.get("pitch", {}).get("f0", 0)
                actual_f1 = dorsal_features.get("estimated_f1", 0) if 'dorsal_features' in dir() else 0
                actual_f2 = dorsal_features.get("estimated_f2", 0) if 'dorsal_features' in dir() else 0

                residual_f0 = abs(actual_pitch - exp["f0"]) if actual_pitch > 0 and exp["f0"] > 0 else 0
                residual_f1 = abs(actual_f1 - exp["f1"]) if actual_f1 > 0 and exp["f1"] > 0 else 0
                residual_f2 = abs(actual_f2 - exp["f2"]) if actual_f2 > 0 and exp["f2"] > 0 else 0

                is_self = (residual_f0 < 50 and residual_f1 < 300 and residual_f2 < 500)
                cd_result = {
                    "expected": [exp["f0"], exp["f1"], exp["f2"], exp["f3"]],
                    "residual": [residual_f0, residual_f1, residual_f2],
                    "is_self_generated": bool(is_self),
                    "prediction_error": float((residual_f0 + residual_f1 + residual_f2) / 3 / 1000),
                }
                self._expected_self_sound = None

            features["corollary_discharge"] = cd_result
            # ─── Fim Corollary Discharge ────────────────────────────

            # Via Ventral (what): descrição do que foi ouvido
            ventral_desc = self._generate_ventral_description(features, is_silent)
            self.last_self_description = ventral_desc

            # Via Dorsal (how): extrai features acústicas para guiar articulação
            dorsal_features = self._extract_dorsal(features, audio, sr)
            dorsal_desc = self._generate_dorsal_description(dorsal_features)
            self.last_dorsal_description = dorsal_desc

            return {
                'features': features,
                'dorsal': dorsal_features,
                'self_description': ventral_desc,
                'dorsal_description': dorsal_desc,
                'duration_seconds': len(audio) / sr,
                'is_silent': is_silent,
            }

        except Exception as e:
            traceback.print_exc()
            return self._empty_result(str(e))

    def _generate_ventral_description(self, f: Dict, silent: bool = False) -> str:
        if silent:
            return "[Áudio: silêncio]"

        parts = []

        bands = f.get('bands', {})
        if bands.get('low', 0) > 0.6:
            parts.append("sons graves")
        elif bands.get('high', 0) > 0.6:
            parts.append("sons agudos")
        elif bands.get('mid', 0) > 0.5:
            parts.append("sons de frequência média")
        else:
            parts.append("espectro misto")

        centroid = f.get('spectral_centroid', 1000)
        if centroid < 300:
            parts.append("som abafado/grave")
        elif centroid > 3000:
            parts.append("som brilhante/agudo")

        pitch = f.get('pitch', {})
        if pitch.get('detected'):
            parts.append(f"tom fundamental ~{pitch.get('f0', 0):.0f}Hz")

        voice = f.get('voice_presence', 0)
        if voice > 0.7:
            parts.append("presença de voz detectada")
        elif voice > 0.4:
            parts.append("possível voz ao fundo")

        onsets = f.get('onsets', {})
        if onsets.get('count', 0) > 5:
            parts.append("ritmo rápido/percussivo")
        elif onsets.get('count', 0) > 2:
            parts.append("eventos sonoros espaçados")
        else:
            parts.append("som contínuo")

        loud = f.get('loudness', {})
        if loud.get('dynamic_range', 0) > 1.0:
            parts.append("dinâmica variável")
        else:
            parts.append("volume estável")

        return f"[Áudio: {', '.join(parts)}]"

    def _extract_dorsal(self, features: Dict, audio: np.ndarray, sr: int) -> Dict:
        """
        Via Dorsal: extrai parâmetros acústicos precisos para guiar ARTICULAÇÃO.
        Responde: "que configuração articulatória produziria este som?"
        """
        n_fft = min(2048, len(audio))
        if len(audio) > n_fft:
            spectrum = np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft)))
        else:
            spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(n_fft, d=1/sr)

        # Encontra picos espectrais (formantes)
        if len(spectrum) > 10:
            smoothed = np.convolve(spectrum, np.ones(5)/5, mode='same')
            peaks = []
            for i in range(2, len(smoothed)-2):
                if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > np.mean(smoothed) * 0.3:
                    peaks.append(float(freqs[i]))
            peaks = sorted(peaks)[:4]
            while len(peaks) < 4:
                peaks.append(0.0)
        else:
            peaks = [0.0, 0.0, 0.0, 0.0]

        pitch_data = features.get('pitch', {})
        f0_est = pitch_data.get('f0', 0) if pitch_data.get('detected') else 0

        # Estima F1/F2/F3 dos picos (ignorando f0 e harmônicos)
        estimated_formants = []
        for p in peaks:
            if p > 0 and (not estimated_formants or p > estimated_formants[-1] * 1.1):
                estimated_formants.append(p)
        while len(estimated_formants) < 3:
            estimated_formants.append(0)
        f1_e = estimated_formants[0] if len(estimated_formants) > 0 else 500
        f2_e = estimated_formants[1] if len(estimated_formants) > 1 else 1500
        f3_e = estimated_formants[2] if len(estimated_formants) > 2 else 2500

        # Energia relativa em bandas de consoantes
        if len(spectrum) > 1:
            high_freq_energy = float(np.sum(spectrum[freqs > 4000])) / (np.sum(spectrum) + 1e-10)
        else:
            high_freq_energy = 0.0

        spectral_slope = features.get('spectral_centroid', 1000) / 3000.0

        return {
            "estimated_f0": float(f0_est),
            "estimated_f1": float(f1_e),
            "estimated_f2": float(f2_e),
            "estimated_f3": float(f3_e),
            "spectral_peaks": peaks[:4],
            "high_freq_energy": float(high_freq_energy),
            "spectral_slope": float(spectral_slope),
            "likely_vowel": self._estimate_vowel(f1_e, f2_e),
            "likely_consonant": self._estimate_consonant(high_freq_energy, features.get('spectral_flux', 0)),
        }

    def _estimate_vowel(self, f1: float, f2: float) -> str:
        """Estima a vogal mais provável baseado em F1 e F2."""
        if f1 < 50 or f2 < 50:
            return "desconhecida"
        vowels = {
            "aa": (700, 1100), "ee": (500, 1800),
            "ii": (300, 2200), "oo": (500, 900),
            "uu": (300, 900),
        }
        best = "desconhecida"
        best_dist = float("inf")
        for name, (vf1, vf2) in vowels.items():
            dist = abs(f1 - vf1) / 500 + abs(f2 - vf2) / 1500
            if dist < best_dist:
                best_dist = dist
                best = name
        return best if best_dist < 0.8 else "desconhecida"

    def _estimate_consonant(self, high_energy: float, flux: float) -> str:
        """Estima tipo de consoante baseado em energia de alta frequência."""
        if high_energy > 0.3 and flux > 0.5:
            return "fricativa"
        elif high_energy > 0.1 and flux > 0.3:
            return "plosiva"
        elif high_energy < 0.05:
            return "nasal"
        return "vogal"

    def _generate_dorsal_description(self, dorsal: Dict) -> str:
        """Descrição da via dorsal: o que a AGI precisa fazer para reproduzir."""
        parts = []
        vowel = dorsal.get("likely_vowel", "?")
        consonant = dorsal.get("likely_consonant", "vogal")
        f1 = dorsal.get("estimated_f1", 0)
        f2 = dorsal.get("estimated_f2", 0)
        parts.append(f"vogal={vowel}")
        if consonant != "vogal":
            parts.append(f"consoante={consonant}")
        parts.append(f"f1~{f1:.0f}")
        parts.append(f"f2~{f2:.0f}")
        return f"[Dorsal: {' '.join(parts)}]"

    def _empty_result(self, error: str = "") -> Dict:
        return {
            'features': {
                'loudness': {'mean': 0, 'peak': 0, 'dynamic_range': 0},
                'bands': {'low': 0.33, 'mid': 0.34, 'high': 0.33},
                'pitch': {'detected': False, 'f0': 0, 'confidence': 0},
                'voice_presence': 0,
                'onsets': {'count': 0, 'density': 0},
                'error': error or 'empty_audio',
            },
            'dorsal': {
                'estimated_f0': 0, 'estimated_f1': 0,
                'estimated_f2': 0, 'estimated_f3': 0,
                'likely_vowel': 'desconhecida', 'likely_consonant': 'vogal',
            },
            'self_description': f"[Áudio: erro ou silêncio]",
            'dorsal_description': "[Dorsal: silêncio]",
            'duration_seconds': 0,
            'is_silent': True,
        }

    def feed_stream(self, audio_chunk: np.ndarray, sr: int):
        """
        Alimenta áudio contínuo no buffer de streaming.

        A AGI ouve EM TEMPO REAL, acumulando até max_buffer_duration
        segundos de áudio para análise contínua.
        """
        if sr != self.sample_rate:
            from scipy import signal
            target_len = int(len(audio_chunk) * self.sample_rate / sr)
            audio_chunk = signal.resample(audio_chunk, target_len)
        self.stream_buffer.append(audio_chunk)
        total_samples = sum(len(c) for c in self.stream_buffer)
        max_samples = int(self.max_buffer_duration * self.sample_rate)
        while total_samples > max_samples and self.stream_buffer:
            removed = self.stream_buffer.pop(0)
            total_samples -= len(removed)

    def get_stream_state(self) -> str:
        """Descrição do que está no buffer auditivo contínuo."""
        if not self.stream_buffer:
            return "[Stream: silêncio]"
        combined = np.concatenate(self.stream_buffer)
        if len(combined) < self.sample_rate * 0.1:
            return "[Stream: buffer curto]"
        result = self.process(combined, self.sample_rate)
        return result.get("self_description", "[Stream: desconhecido]")

    def get_state(self) -> Dict:
        return {
            'total_segments': self.total_segments_processed,
            'last_description': self.last_self_description,
            'last_dorsal_description': self.last_dorsal_description,
            'features': self.last_features,
            'stream_buffer_seconds': sum(len(c) for c in self.stream_buffer) / self.sample_rate if self.stream_buffer else 0,
        }
