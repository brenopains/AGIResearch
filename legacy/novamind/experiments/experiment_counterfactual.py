"""
Experimento 4: World Model + Simulation -> novo HDV -> decoder.

Testa se o SimulationEngine pode gerar novos HDVs (contrafactuais)
que o decoder traduz em imagens coerentes.

Pipeline:
  1. Codifica imagem real para HDV
  2. SimulationEngine cria variacao (ex: "gato com asas")
  3. A variacao produz um NOVO HDV
  4. Decoder traduz em imagem

Como o SimEngine atual opera em conceitos, nao em pixels,
simulamos a manipulacao semantica via algebra HDV:
  - Bundle: adiciona conceito
  - Bind: funde conceitos
  - Interpolacao: transicao suave
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision, torchvision.transforms as T
from PIL import Image
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)
print("EXPERIMENTO 4: WORLD MODEL + SIMULACAO -> NOVO HDV -> DECODER")
print("=" * 60)

eng = GroundingEngine(device=device)
pipe = VisualPipeline(eng.proj_visual, device=device)
pipe.decoder.load_state_dict(torch.load("save/visual_decoder.pt", map_location=device))
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

# Helper: HDV hash para conceitos abstratos
def concept_hdv(text: str) -> torch.Tensor:
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    g = torch.Generator(device=device)
    g.manual_seed(int(h[:8], 16))
    return torch.sign(torch.randn(10000, generator=g, device=device))

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

# Encontra imagens
gato_img = aviao_img = carro_img = None
for i in range(len(testset)):
    img, lab = testset[i]
    if lab == 3 and gato_img is None: gato_img = img
    if lab == 0 and aviao_img is None: aviao_img = img
    if lab == 1 and carro_img is None: carro_img = img
    if all(x is not None for x in [gato_img, aviao_img, carro_img]):
        break

# 4a. CONTRAFACTUAL: gato com asas (gato + aviao)
print("\n4a. CONTRAFACTUAL: gato + asas = gato alado")
hdv_gato = encode(gato_img)
hdv_aviao = encode(aviao_img)
hdv_asas = concept_hdv("asas")

# Simulation: adiciona asas ao gato
hdv_gato_alado = torch.sign(hdv_gato + hdv_asas * concept_hdv("asa"))
rec_gato_alado = decode(hdv_gato_alado)
comp = Image.new("RGB", (448, 224))
comp.paste(gato_img.resize((224, 224)), (0, 0))
comp.paste(T.ToPILImage()(rec_gato_alado.cpu()), (224, 0))
comp.save("experiments/contrafactual_gato_alado.png")
print("  Gato alado: experiments/contrafactual_gato_alado.png")

# 4b. INTERPOLACAO: transicao suave de gato para aviao
print("\n4b. INTERPOLACAO: gato -> aviao")
n_frames = 5
for i, peso_gato in enumerate([1.0, 0.8, 0.6, 0.4, 0.2]):
    peso_aviao = 1.0 - peso_gato
    hdv_interp = torch.sign(hdv_gato * peso_gato + hdv_aviao * peso_aviao)
    rec = decode(hdv_interp)
    comp = Image.new("RGB", (448, 224))
    comp.paste(gato_img.resize((224, 224)), (0, 0))
    comp.paste(T.ToPILImage()(rec.cpu()), (224, 0))
    comp.save(f"experiments/interpolacao_{i}.png")
print("  Frames salvos: experiments/interpolacao_*.png")

# 4c. VIDEO IMAGINADO: sequencia de interpolacao
print("\n4c. VIDEO IMAGINADO (sequencia interpolada)")
try:
    frames = []
    for i in range(8):
        t = i / 7.0
        hdv_frame = torch.sign(hdv_gato * (1 - t) + hdv_aviao * t)
        rec = decode(hdv_frame)
        frames.append(T.ToPILImage()(rec.cpu()))
    frames[0].save("experiments/video_imaginado.gif",
                  save_all=True, append_images=frames[1:],
                  duration=200, loop=0)
    print("  GIF salvo: experiments/video_imaginado.gif (gato -> aviao)")
except Exception as e:
    print(f"  Erro ao criar GIF: {e}")

print("\nConcluido!")
