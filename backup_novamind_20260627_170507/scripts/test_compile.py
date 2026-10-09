"""Quick test: encode existing books into test_brain.hdv"""
import sys, os, json, struct, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.grounding_engine import GroundingEngine

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
engine = GroundingEngine(device=device)
print(f"Engine loaded on {device}")

txt_dir = Path("D:/novamind_data/text_datasets")
splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

sentences = []
for f in sorted(txt_dir.iterdir()):
    if f.suffix == '.txt':
        text = f.read_text(encoding='utf-8', errors='replace')
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 20 and not line.startswith('#'):
                for s in splitter.split(line):
                    s = s.strip()
                    if 10 < len(s) < 500:
                        sentences.append((s, f.stem))

print(f"{len(sentences)} sentences from existing texts")

out = Path("D:/novamind_data/hdv/test_brain.hdv")
out.parent.mkdir(parents=True, exist_ok=True)

item_count = 0
with open(out, 'wb') as f:
    f.write(b'\x00' * 256)
    for text, src in sentences[:5000]:
        try:
            mf = engine.encode_text(text)
            vec = mf.tensor.cpu().numpy().astype(np.float32)
            meta = json.dumps({"id": f"{src}_{item_count}", "text": text, "source": src}).encode('utf-8')
            data = vec.tobytes()
            f.write(struct.pack("<I", len(data)))
            f.write(data)
            f.write(struct.pack("<I", len(meta)))
            f.write(meta)
            item_count += 1
        except Exception as e:
            pass

    offset_meta = f.tell()
    f.write(struct.pack("<Q", offset_meta))
    header = b"HDV1" + struct.pack("<IQ", item_count, offset_meta)
    f.seek(0)
    f.write(header + b'\x00' * (256 - len(header)))

print(f"Saved {item_count} items to test_brain.hdv ({out.stat().st_size / 1e6:.1f} MB)")
