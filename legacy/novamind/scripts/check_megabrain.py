"""Check contents of mega_brain.hdv"""
import sys, os, struct, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

path = Path("D:/novamind_data/hdv/mega_brain.hdv")
if not path.exists():
    print("mega_brain.hdv not found!")
    sys.exit(1)

with open(path, "rb") as f:
    magic = f.read(4)
    count, offset = struct.unpack("<IQ", f.read(12))
    print(f"Magic: {magic}, Items: {count}, Offset: {offset}")

    if count == 0:
        print("No items!")
        sys.exit(0)

    # Read first 5 items
    f.seek(256)
    for i in range(min(5, count)):
        data_len = struct.unpack("<I", f.read(4))[0]
        f.read(data_len)
        meta_len = struct.unpack("<I", f.read(4))[0]
        meta = json.loads(f.read(meta_len).decode("utf-8"))
        text = meta.get("text", "?")
        print(f"[{i}] id={meta.get('id','?')} text={text[:100]}")

    # Read last few items
    if count > 5:
        items_per_vec = data_len + 8 + meta_len  # data_len(4) + data + meta_len(4) + meta
        f.seek(256 + (count - 3) * items_per_vec)
        for i in range(count - 3, count):
            data_len = struct.unpack("<I", f.read(4))[0]
            f.read(data_len)
            meta_len = struct.unpack("<I", f.read(4))[0]
            meta = json.loads(f.read(meta_len).decode("utf-8"))
            text = meta.get("text", "?")
            print(f"[{i}] id={meta.get('id','?')} text={text[:100]}")

    print(f"\nTotal file size: {path.stat().st_size / 1e6:.1f} MB")
    print(f"File name: {path.name}")
