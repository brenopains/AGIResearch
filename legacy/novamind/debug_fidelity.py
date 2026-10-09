"""Debug: compare bipolar vs presign vs direct features reconstruction quality."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"
import torch, cv2, numpy as np
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline

eng = GroundingEngine(device="cuda")
vp = VisualPipeline(eng.proj_visual, device="cuda")
vp.decoder.load_state_dict(torch.load(r"D:\novamind_data\models\visual_decoder.pt"))
vp.decoder.eval()

# Frame do video do cachorro
cap = cv2.VideoCapture(r"D:\novamind_data\MSVD\YouTubeClips\IWhrWLOAin0_1_4.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
ret, frame = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_t = eng.transform(frame_rgb).unsqueeze(0).to("cuda")
with torch.no_grad():
    feats = eng.feature_extractor(img_t).view(-1)

mf, presign, _ = vp.encode_frame(feats)

with torch.no_grad():
    f_bip = torch.matmul(vp.pinv, mf.tensor.float())
    img_bip = vp.decoder(f_bip.unsqueeze(0)).squeeze(0)
    f_pre = torch.matmul(vp.pinv, presign.float())
    img_pre = vp.decoder(f_pre.unsqueeze(0)).squeeze(0)
    img_feat = vp.decoder(feats.unsqueeze(0)).squeeze(0)

img_orig = torch.from_numpy(cv2.resize(frame_rgb, (224,224)).astype(np.float32)/255.0).permute(2,0,1)

m_bip = vp.verify_fidelity(img_orig, img_bip)
m_pre = vp.verify_fidelity(img_orig, img_pre)
m_feat = vp.verify_fidelity(img_orig, img_feat)

print(f"De bipolar:       PSNR={m_bip['psnr']:.1f} dB  SSIM={m_bip['ssim']:.4f}")
print(f"De presign:       PSNR={m_pre['psnr']:.1f} dB  SSIM={m_pre['ssim']:.4f}")
print(f"De features direto: PSNR={m_feat['psnr']:.1f} dB  SSIM={m_feat['ssim']:.4f}")

sim_bip_pre = float(torch.cosine_similarity(mf.tensor.float().unsqueeze(0), presign.unsqueeze(0))[0])
sim_feat_bip = float(torch.cosine_similarity(feats.unsqueeze(0), f_bip.unsqueeze(0))[0])
sim_feat_pre = float(torch.cosine_similarity(feats.unsqueeze(0), f_pre.unsqueeze(0))[0])
print(f"\nCOSSIM bipolar vs presign:    {sim_bip_pre:.4f}")
print(f"COSSIM feats vs rec_bipolar:  {sim_feat_bip:.4f}")
print(f"COSSIM feats vs rec_presign:   {sim_feat_pre:.4f}")

# Salvar imagens pra comparacao
out = r"D:\novamind_data\debug_fidelity"
os.makedirs(out, exist_ok=True)
def save(t, name):
    arr = (t.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, name), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
save(img_orig, "original.png")
save(img_bip, "from_bipolar.png")
save(img_pre, "from_presign.png")
save(img_feat, "from_features.png")
print(f"\nImagens salvas em {out}/")
