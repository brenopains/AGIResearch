import numpy as np
import time
import random
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from .vocal_synthesizer import synthesize, formant_for_vowel, save_wav_to

PHONEMES = {
    "aa": (700, 1100, 2500),
    "ee": (500, 1800, 2500),
    "ii": (300, 2200, 3000),
    "oo": (500, 900, 2500),
    "uu": (300, 900, 2300),
    "mmm": (200, 1000, 2000),
    "p": "plosive",
    "b": "plosive",
    "t": "plosive",
    "d": "plosive",
    "k": "plosive",
    "g": "plosive",
    "f": "fricative",
    "v": "fricative",
    "s": "fricative",
    "z": "fricative",
    "n": "nasal",
}

SYLLABLE_PATTERNS = ["V", "CV", "VC", "CVC", "CVV"]


class EmergentLanguage:
    def __init__(self):
        self.vowels = ["aa", "ee", "ii", "oo", "uu"]
        self.consonants = ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "n", "mmm"]
        self.phoneme_counts: Dict[str, int] = defaultdict(int)
        self.transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.self_play_history: deque = deque(maxlen=200)
        self.lexicon: Dict[str, str] = {}
        self._vocal_dir = r"D:\novamind_data\temp\novamind_vocal"
        os.makedirs(self._vocal_dir, exist_ok=True)
        self._wav_count = 0

    def generate_syllable(self) -> str:
        pattern = random.choice(SYLLABLE_PATTERNS)
        result = []
        for p in pattern:
            if p == "V":
                result.append(random.choice(self.vowels))
            elif p == "C":
                result.append(random.choice(self.consonants))
        return "+".join(result)

    def phoneme_to_wav(self, phoneme: str, pitch: float = 0,
                       duration: float = 0.3) -> Optional[str]:
        if phoneme in PHONEMES and isinstance(PHONEMES[phoneme], tuple):
            f1, f2, f3 = PHONEMES[phoneme]
            if pitch == 0:
                pitch = 120 + hash(phoneme) % 120
            wav = synthesize(pitch=pitch, f1=f1, f2=f2, f3=f3,
                             amplitude=0.6, duration=duration, voiced=True)
        elif phoneme in PHONEMES and PHONEMES[phoneme] in ("plosive", "fricative"):
            pitch_val = pitch if pitch else 150
            noise_amp = 0.3 if PHONEMES[phoneme] == "fricative" else 0.8
            wav = synthesize(pitch=pitch_val, f1=400, f2=1500, f3=2500,
                             amplitude=0.4, duration=max(0.05, duration * 0.3),
                             voiced=PHONEMES[phoneme] != "fricative")
        else:
            return None
        self._wav_count += 1
        fname = f"lang_{self._wav_count:04d}.wav"
        fpath = os.path.join(self._vocal_dir, fname)
        save_wav_to(wav, fpath)
        return fpath

    def syllable_to_wav(self, syllable: str) -> Optional[List[str]]:
        phonemes = syllable.split("+")
        paths = []
        for ph in phonemes:
            p = self.phoneme_to_wav(ph, duration=0.15)
            if p:
                paths.append(p)
        return paths if paths else None

    def babble_sequence(self, length: int = 3) -> str:
        syllables = []
        prev = None
        for _ in range(length):
            syl = self.generate_syllable()
            first_ph = syl.split("+")[0]
            if prev and (prev, first_ph) in self.transition_counts:
                if random.random() > 0.7:
                    weighted = []
                    for (p, n), c in self.transition_counts.items():
                        if p == prev:
                            weighted.extend([n] * c)
                    if weighted:
                        target = random.choice(weighted)
                        syl = target + "+" + random.choice(self.vowels)
            syllables.append(syl)
            for ph in syl.split("+"):
                self.phoneme_counts[ph] += 1
                if prev:
                    self.transition_counts[(prev, ph)] += 1
                prev = ph
        return " ".join(syllables)

    def learn_from_self_play(self, sequence: str):
        self.self_play_history.append({
            "time": time.time(),
            "sequence": sequence,
            "phonemes_used": list(set(sequence.replace(" ", "+").split("+"))),
        })
        total = sum(self.phoneme_counts.values())
        if total > 20:
            most_common = sorted(self.phoneme_counts.items(), key=lambda x: -x[1])[:5]
            if len(most_common) >= 3:
                label = "".join(ph for ph, _ in most_common[:3])
                label_clean = label.replace("mm", "m").replace("aa", "a").replace("ee", "e").replace("ii", "i").replace("oo", "o").replace("uu", "u")
                if label_clean not in self.lexicon.values():
                    concept = f"proto_{len(self.lexicon)}"
                    self.lexicon[concept] = label_clean

    def vocalize_sequence(self, sequence: str) -> Optional[str]:
        parts = sequence.split()
        all_paths = []
        for part in parts:
            paths = self.syllable_to_wav(part)
            if paths:
                all_paths.extend(paths)
        if all_paths:
            return all_paths[-1]
        return None

    def get_state(self) -> Dict:
        return {
            "phonemes_known": len(self.phoneme_counts),
            "transitions_learned": len(self.transition_counts),
            "lexicon_size": len(self.lexicon),
            "self_play_cycles": len(self.self_play_history),
            "most_common": sorted(self.phoneme_counts.items(), key=lambda x: -x[1])[:5],
            "lexicon": dict(self.lexicon),
        }
