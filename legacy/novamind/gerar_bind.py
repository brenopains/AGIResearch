"""Gera imagem composta: gato × aviao (bind) e mostra resultado."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import torchvision.transforms as T
import torchvision
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

eng = GroundingEngine(device=device)
pipe = VisualPipeline(eng.proj_visual, device=device)
pipe.decoder.load_state_dict(torch.load("save/visual_decoder_best.pt", map_location=device))
pipe.decoder.eval()

norm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def encode(img_pil):
    n = norm(img_pil).unsqueeze(0).to(device)
    f = eng.feature_extractor(n).flatten(start_dim=1)
    hdv, _, _ = pipe.encode_frame(f[0])
    return hdv.tensor

def decode(hdv_vec):
    with torch.no_grad():
        return pipe.decode_bipolar(hdv_vec).clamp(0, 1)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

# Pega gato (classe 3) e aviao (classe 0)
gato_img = aviao_img = None
for i in range(10000):
    img, lab = testset[i]
    if lab == 3 and gato_img is None: gato_img = img
    if lab == 0 and aviao_img is None: aviao_img = img
    if gato_img and aviao_img: break

hdv_gato = encode(gato_img)
hdv_aviao = encode(aviao_img)

print(f"\n--- RECONSTRUCAO BASELINE ---")
rec_gato = decode(hdv_gato)
rec_aviao = decode(hdv_aviao)
print(f"Gato OK | Aviao OK")

print(f"\n--- BIND: gato X aviao ('gato alado') ---")
hdv_bind = hdv_gato * hdv_aviao
rec_bind = decode(hdv_bind)

print(f"\n--- BUNDLE: gato + aviao ---")
hdv_bundle = torch.sign(hdv_gato + hdv_aviao)
rec_bundle = decode(hdv_bundle)

print(f"\n--- INTERPOLACAO: 0.5*gato + 0.5*aviao ---")
hdv_interp = torch.sign(hdv_gato * 0.5 + hdv_aviao * 0.5)
rec_interp = decode(hdv_interp)

# Salva grid 2x3
grid = Image.new("RGB", (224*3, 224*2))
to_pil = T.ToPILImage()

cell = [
    (gato_img.resize((224,224)), "GATO original"),
    (aviao_img.resize((224,224)), "AVIAO original"),
    (to_pil(rec_gato.cpu()), "GATO reconstruido"),
    (to_pil(rec_aviao.cpu()), "AVIAO reconstruido"),
    (to_pil(rec_bind.cpu()), "BIND: gato X aviao"),
    (to_pil(rec_bundle.cpu()), "BUNDLE: gato + aviao"),
]

DESC = """GATO original       | AVIAO original      | GATO reconstruido
AVIAO reconstruido  | BIND (gato X aviao) | BUNDLE (gato + aviao)"""

for i, (img, label) in enumerate(cell):
    x = (i % 3) * 224
    y = (i // 3) * 224
    grid.paste(img, (x, y))

grid.save("exp_composicao_hdv.png")
print(f"Imagem salva: exp_composicao_hdv.png ({grid.size})")
print(f"\n{DESC}")
print(f"\n--- METRICAS ---")
t_gato = T.ToTensor()(gato_img.resize((224,224))).unsqueeze(0).to(device)
t_aviao = T.ToTensor()(aviao_img.resize((224,224))).unsqueeze(0).to(device)
def psnr(rec, orig):
    mse = ((rec - orig)**2).mean().item()
    return 20 * __import__('numpy').log10(1.0 / max(1e-10, mse**0.5))

print(f"Gato:              PSNR={psnr(rec_gato, t_gato[0]):.1f}dB")
print(f"Aviao:             PSNR={psnr(rec_aviao, t_aviao[0]):.1f}dB")
print(f"BIND(gato,aviao):  similaridade com gato={1-((rec_bind-t_gato[0])**2).mean().item():.3f}")
print(f"BIND(gato,aviao):  similaridade com aviao={1-((rec_bind-t_aviao[0])**2).mean().item():.3f}")
print(f"BUNDLE(gato+aviao):similaridade com gato={1-((rec_bundle-t_gato[0])**2).mean().item():.3f}")
