"""
Experimento 3: Multimodal (imagem + texto) bundle.

Testa se bundle de HDV(imagem_cachorro) + HDV(texto_cachorro_correndo)
produz algo coerente via decoder.

Tambem testa:
  - cross-modal retrieval: texto -> HDV -> busca no CGE
  - consistency: imagem_cachorro e texto_cachorro tem HDVs similares?
"""
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision, torchvision.transforms as T
import numpy as np
from PIL import Image
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline
from core.bge_encoder import BGEEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)
print("EXPERIMENTO 3: MULTIMODAL (IMAGEM + TEXTO) BUNDLE")
print("=" * 60)

eng = GroundingEngine(device=device)
pipe = VisualPipeline(eng.proj_visual, device=device)
pipe.decoder.load_state_dict(torch.load("save/visual_decoder.pt", map_location=device))
pipe.decoder.eval()

# BGE encoder para texto
bge = BGEEncoder(device=device)

norm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])

def encode_img(img_pil):
    n = norm(img_pil).unsqueeze(0).to(device)
    f = eng.feature_extractor(n).flatten(start_dim=1)
    hdv, _, _ = pipe.encode_frame(f[0])
    return hdv.tensor

def encode_text(text: str) -> torch.Tensor:
    """Codifica texto como HDV 10000D via BGE."""
    v = bge.encode(text)  # (768,)
    if v is None:
        return None
    # Projeta para 10000D
    proj = eng.proj_semantic.to(device) if hasattr(eng, 'proj_semantic') else None
    if proj is None:
        # Fallback: hash
        h = hashlib.sha256(text.encode()).hexdigest()
        g = torch.Generator(device=device)
        g.manual_seed(int(h[:8], 16))
        return torch.sign(torch.randn(10000, generator=g, device=device))
    presign = torch.matmul(proj, v.to(device))
    return torch.sign(presign)

def decode(hdv_vec):
    with torch.no_grad():
        return pipe.decode_bipolar(hdv_vec).clamp(0, 1)

# Teste 1: Similaridade cross-modal
print("\n1. SIMILARIDADE CROSS-MODAL (imagem vs texto)")
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
for idx, texto in [(3, "gato"), (0, "aviao"), (5, "cachorro"), (1, "automovel")]:
    img, lab = testset[idx]
    hdv_img = encode_img(img)
    hdv_text = encode_text(texto)
    if hdv_text is not None:
        sim = torch.nn.functional.cosine_similarity(
            hdv_img.flatten().unsqueeze(0), hdv_text.flatten().unsqueeze(0)
        ).item()
        print(f"  imagem({texto:12s}) x texto({texto:12s}): sim={sim:.3f}")

# Teste 2: Bundle imagem + texto
print("\n2. BUNDLE IMAGEM + TEXTO -> DECODER")
for idx, texto in [(3, "gato voador"), (5, "cachorro correndo")]:
    img, lab = testset[idx]
    hdv_img = encode_img(img)
    hdv_text = encode_text(texto)
    if hdv_text is not None:
        bundle = torch.sign(hdv_img + hdv_text)
        rec = decode(bundle)
        rec_img = T.ToPILImage()(rec.cpu())
        comp = Image.new("RGB", (448, 224))
        comp.paste(img.resize((224, 224)), (0, 0))
        comp.paste(rec_img, (224, 0))
        path = f"experiments/multimodal_{texto.replace(' ','_')}.png"
        comp.save(path)
        print(f"  bundle(img, '{texto}') salvo em {path}")

# Teste 3: Contraste (textos diferentes, mesma imagem)
print("\n3. CONTRASTE: mesma imagem, textos diferentes")
img, lab = testset[3]  # gato
hdv_img = encode_img(img)
for texto in ["gato", "cachorro", "aviao", "carro"]:
    hdv_text = encode_text(texto)
    if hdv_text is not None:
        sim = torch.nn.functional.cosine_similarity(
            hdv_img.flatten().unsqueeze(0), hdv_text.flatten().unsqueeze(0)
        ).item()
        print(f"  imagem(gato) x texto({texto:12s}): sim={sim:.3f}")

print("\nConcluido.")
