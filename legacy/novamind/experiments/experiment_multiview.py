"""
Experimento 1: Bundle multi-angulo -> prototipo.

Testa se bundle de HDVs de diferentes ângulos do mesmo objeto
produz um prototipo coerente via decoder.
Para este teste, usamos CIFAR-10 como proxy (multiplas instancias da mesma classe).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision, torchvision.transforms as T
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline
from core.scene_decoder import MentalState, SceneRepresentation
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)
print("EXPERIMENTO 1: PROTOTIPO POR BUNDLE MULTI-INSTANCIA")
print("=" * 60)

eng = GroundingEngine(device=device)
pipe = VisualPipeline(eng.proj_visual, device=device)
pipe.decoder.load_state_dict(torch.load("save/visual_decoder.pt", map_location=device))
pipe.decoder.eval()

norm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])

def encode(img_pil):
    n = norm(img_pil).unsqueeze(0).to(device)
    f = eng.feature_extractor(n).flatten(start_dim=1)
    hdv, _, _ = pipe.encode_frame(f[0])
    return hdv.tensor

def decode(hdv_vec):
    with torch.no_grad():
        return pipe.decode_bipolar(hdv_vec).clamp(0, 1)

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

# Para cada classe, faz bundle de N imagens
CLASSES = ["aviao", "auto", "passaro", "gato", "veado", "cachorro", "sapo", "cavalo", "navio", "cam"]
N_BUNDLE = [3, 5, 10]

for classe, nome in enumerate(CLASSES[:4]):
    # Pega N imagens da classe
    imgs = []
    for i in range(len(trainset)):
        if len(imgs) >= max(N_BUNDLE):
            break
        img, lab = trainset[i]
        if lab == classe:
            imgs.append(img)

    if len(imgs) < min(N_BUNDLE):
        continue

    # Codifica
    hdvs = [encode(img) for img in imgs]

    for n in N_BUNDLE:
        if n > len(hdvs):
            continue
        bundle = torch.sign(sum(hdvs[:n]))
        rec = decode(bundle)
        rec_img = T.ToPILImage()(rec.cpu())

        # Lado a lado
        comp = Image.new("RGB", (224 * (n + 1), 224))
        for i in range(n):
            comp.paste(imgs[i].resize((224, 224)), (i * 224, 0))
        comp.paste(rec_img, (n * 224, 0))
        path = f"experiments/prototipo_{nome}_N{n}.png"
        comp.save(path)

        # Similaridade com a primeira imagem
        sim_ref = torch.nn.functional.cosine_similarity(
            bundle.flatten().unsqueeze(0), hdvs[0].flatten().unsqueeze(0)
        ).item()
        print(f"  {nome:12s} N={n:2d}: similaridade_media={sim_ref:.3f} | salvo={path}")

print("\nConcluido. Verifique os arquivos em experiments/prototipo_*.png")
