"""
Experimento 2: Sequencia temporal -> decoder -> video.

Testa se uma sequencia de HDVs (frames de video) pode ser decodificada
em uma sequencia de frames, simulando um video.

Usa CIFAR-10 como proxy: seleciona imagens da mesma classe em sequencia
crescente de complexidade.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision, torchvision.transforms as T
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)
print("EXPERIMENTO 2: SEQUENCIA TEMPORAL -> DECODER -> VIDEO")
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

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

# Pega 8 imagens da classe "cavalo" em ordem crescente de complexidade
# (proxies para frames de um video de cavalo)
classe = 7  # cavalo
imgs = []
for i in range(len(testset)):
    if len(imgs) >= 8:
        break
    img, lab = testset[i]
    if lab == classe:
        imgs.append(img)

if len(imgs) < 4:
    print("Poucas imagens de cavalo encontradas")
else:
    # Reconstroi cada frame
    frames = []
    for i, img in enumerate(imgs):
        hdv = encode(img)
        rec = decode(hdv)
        frame = (rec.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(frame)

    # Salva como grid animado (strip horizontal)
    strip = Image.new("RGB", (224 * len(frames), 224))
    for i, rec_img in enumerate([T.ToPILImage()(rec.cpu()) for rec in [decode(encode(img)) for img in imgs]]):
        orig_img = imgs[i].resize((224, 224))
        comp = Image.new("RGB", (448, 224))
        comp.paste(orig_img, (0, 0))
        comp.paste(rec_img, (224, 0))
        comp.save(f"experiments/video_frame_{i}.png")

    # Cria GIF (se PIL suportar)
    try:
        gif_frames = []
        for img in imgs:
            gif_frames.append(img.resize((224, 224)))
        gif_frames[0].save("experiments/video_original.gif",
                          save_all=True, append_images=gif_frames[1:],
                          duration=500, loop=0)
        rec_frames = []
        for img in imgs:
            rec_img = T.ToPILImage()(decode(encode(img)).cpu())
            rec_frames.append(rec_img)
        rec_frames[0].save("experiments/video_reconstruido.gif",
                          save_all=True, append_images=rec_frames[1:],
                          duration=500, loop=0)
        print("  GIFs salvos: experiments/video_original.gif e video_reconstruido.gif")
    except Exception as e:
        print(f"  GIF nao disponivel: {e}")

    print(f"  {len(imgs)} frames processados")
    print("  Verifique experiments/video_frame_*.png")

print("\nConcluido.")
