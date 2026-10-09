"""Test feature extraction — checks GPU memory and finds the crash point."""
import sys, os, time
sys.path.insert(0, '.')
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

coco = r'D:\AGI2\datasets\coco2017\train2017'
files = sorted([f for f in os.listdir(coco) if f.lower().endswith(('.jpg','.jpeg','.png'))])[:50000]
device = 'cuda'

resnet = models.resnet18(weights=models.Resnet18_Weights.DEFAULT).to(device)
resnet.eval()
feat_ext = nn.Sequential(*list(resnet.children())[:-1])

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

all_feats, all_imgs = [], []
batch_f, batch_i = [], []
batch_size = 32

for i, fname in enumerate(files):
    try:
        path = os.path.join(coco, fname)
        img = Image.open(path).convert('RGB')
        t = transform(img)
        batch_i.append(t)
        batch_f.append(t)
    except Exception as e:
        print(f'ERROR {i} {fname}: {e}')
        continue

    if len(batch_f) == batch_size or i == len(files) - 1:
        x = torch.stack(batch_f).to(device)
        with torch.no_grad():
            feats = feat_ext(x).squeeze(-1).squeeze(-1)
        all_feats.append(feats.cpu().numpy())
        all_imgs.append(torch.stack(batch_i).cpu().numpy())
        batch_f, batch_i = [], []

        if (i + 1) % 3200 == 0 or i == len(files) - 1:
            mem = torch.cuda.memory_allocated(device) / 1024**3
            vram = torch.cuda.memory_reserved(device) / 1024**3
            print(f'  {i+1}/{len(files)} | VRAM={mem:.2f}GB (reserved={vram:.2f}GB) | img/s={3200/(time.time()-_t):.0f}' if i>=3200 else f'  {i+1}/{len(files)} | VRAM={mem:.2f}GB')
            if i >= 3200:
                _t = time.time()
            else:
                _t = time.time()

feats_arr = np.concatenate(all_feats, axis=0)[:50000]
imgs_arr = np.concatenate(all_imgs, axis=0)[:50000]
np.save(r'D:\AGI2\treino\coco_train_features.npy', feats_arr)
np.save(r'D:\AGI2\treino\coco_train_images.npy', imgs_arr)
print(f'OK! Features: {feats_arr.shape}, Images: {imgs_arr.shape}')
