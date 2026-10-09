"""
HDVValidation  Bateria de 6 Testes Semanticos para o Hiperespaco

Antes de treinar 50 epocas, precisamos saber:
  O HDV realmente representa SEMANTICA ou apenas compressao visual?

Testes:
  1. RECONSTRUCAO: PSNR/SSIM baseline (ja existe)
  2. ROBUSTEZ: corromper HDV em 10/20/40/60%  reconstrucao ainda reconhecivel?
  3. ALGEBRA: HDV(A) + HDV(B)  decoder  conceito composto?
  4. ANALOGIA: HDV(rei) - HDV(homem) + HDV(mulher)  HDV(rainha)?
  5. MEMORIA: catastrophic forgetting  novas imagens destroem as antigas?
  6. COMPOSICAO: bind de HDVs de objetos  decoder  cena?

Uso:
    python -c "from core.hdv_validation import HDVValidation; v = HDVValidation(); v.run_all()"
"""
import sys, os, time, torch, numpy as np, json, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_BASE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TestResult:
    name: str = ""
    passed: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    details: str = ""


class HDVValidation:
    """Bateria de 6 testes para validar o hiperespaco HDV."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results: List[TestResult] = []

        # Carrega pipeline
        print("[VALID] Carregando pipeline...")
        from core.grounding_engine import GroundingEngine
        from core.visual_pipeline import VisualPipeline
        import torchvision.transforms as T
        self.eng = GroundingEngine(device=self.device)
        self.pipe = VisualPipeline(self.eng.proj_visual, device=self.device)
        decoder_path = os.path.join(_BASE, "save", "visual_decoder.pt")
        if os.path.exists(decoder_path):
            sd = torch.load(decoder_path, map_location=self.device)
            self.pipe.decoder.load_state_dict(sd)
            self.pipe.decoder.eval()
            print(f"[VALID] Decoder carregado: {os.path.getsize(decoder_path)/1024:.0f}KB")

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    #  TESTE 1: RECONSTRUCAO 

    def test_reconstruction(self) -> TestResult:
        """Teste 1: Reconstrucao  PSNR/SSIM baseline."""
        print("\n[TESTE 1] Reconstrucao")
        import torchvision
        from PIL import Image

        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
        psnrs = []
        for idx in [0, 10, 25, 50, 100]:
            img_pil, label = testset[idx]
            tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.eng.feature_extractor(tensor).flatten(start_dim=1)
                hdv, _, _ = self.pipe.encode_frame(feats[0])
                recon = self.pipe.decode_bipolar(hdv.tensor)
            mse = ((recon - tensor[0]) ** 2).mean().item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
            psnrs.append(psnr)

        avg_psnr = sum(psnrs) / len(psnrs)
        result = TestResult("Reconstrucao", avg_psnr > 20, {"avg_psnr": avg_psnr, "per_sample": psnrs})
        print(f"  PSNR medio: {avg_psnr:.1f}dB {'' if avg_psnr > 20 else ''}")
        self.results.append(result)
        return result

    #  TESTE 2: ROBUSTEZ 

    def test_robustness(self) -> TestResult:
        """Teste 2: Corromper HDV e ver se reconstrucao mantem-se reconhecivel."""
        print("\n[TESTE 2] Robustez a corrupcao do HDV")
        import torchvision

        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
        img_pil, _ = testset[0]
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.eng.feature_extractor(tensor).flatten(start_dim=1)
            hdv, _, _ = self.pipe.encode_frame(feats[0])
            original_hdv = hdv.tensor.clone()

        results = {}
        for corruption in [0.1, 0.2, 0.4, 0.6]:
            noisy = original_hdv.clone()
            mask = torch.rand(10000, device=self.device) < corruption
            noisy[mask] = -noisy[mask]
            with torch.no_grad():
                recon = self.pipe.decode_bipolar(noisy)
            mse = ((recon - tensor[0]) ** 2).mean().item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
            results[f"corrupt_{int(corruption*100)}%"] = {"psnr": psnr, "mse": mse}
            print(f"  Corrupcao {int(corruption*100)}%: PSNR={psnr:.1f}dB")

        # Passa se PSNR > 10dB mesmo com 40% corrompido
        passed = results.get("corrupt_40%", {}).get("psnr", 0) > 10
        result = TestResult("Robustez", passed, results)
        self.results.append(result)
        return result

    #  TESTE 3: ALGEBRA HDV 

    def test_algebra(self) -> TestResult:
        """Teste 3: HDV(gato) + HDV(oculos)  decoder  'gato com oculos'?"""
        print("\n[TESTE 3] Algebra HDV  composicao de conceitos")
        import torchvision

        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

        # Pega imagem de gato (classe 3) e "adiciona" algo
        gato_img, _ = testset[3]  # gato
        sapo_img, _ = testset[6]  # sapo (cor verde, textura diferente)

        # Codifica ambos
        gato_tensor = self.transform(gato_img).unsqueeze(0).to(self.device)
        sapo_tensor = self.transform(sapo_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f_gato = self.eng.feature_extractor(gato_tensor).flatten(start_dim=1)
            f_sapo = self.eng.feature_extractor(sapo_tensor).flatten(start_dim=1)
            hdv_gato, _, _ = self.pipe.encode_frame(f_gato[0])
            hdv_sapo, _, _ = self.pipe.encode_frame(f_sapo[0])

            # Bundle: soma dos HDVs
            hdv_sum = torch.sign(hdv_gato.tensor + hdv_sapo.tensor)
            recon_sum = self.pipe.decode_bipolar(hdv_sum)

            # Bind: XOR dos HDVs
            hdv_bind = hdv_gato.tensor * hdv_sapo.tensor
            recon_bind = self.pipe.decode_bipolar(hdv_bind)

        mse_sum = ((recon_sum - gato_tensor[0]) ** 2).mean().item()
        mse_bind = ((recon_bind - gato_tensor[0]) ** 2).mean().item()
        print(f"  Bundle(gato+sapo): MSE={mse_sum:.4f}")
        print(f"  Bind(gato*sapo):   MSE={mse_bind:.4f}")

        result = TestResult("Algebra HDV", False, {"bundle_mse": mse_sum, "bind_mse": mse_bind},
                           "Operacoes HDV funcionam mas decoder nao foi treinado para composicao")
        self.results.append(result)
        return result

    #  TESTE 4: ANALOGIA 

    def test_analogy(self) -> TestResult:
        """Teste 4: HDV(rei) - HDV(homem) + HDV(mulher)  HDV(rainha).

        Em HDC classico: analogia = bind(inverse(A), B)  bind(C, D).
        Aqui testamos se a algebra HDV se aproxima disso.
        """
        print("\n[TESTE 4] Analogia HDV  A:B :: C:D")
        # Usa hash-based HDVs para conceitos que nao estao no CIFAR (teste conceitual)
        def concept_hdv(text: str) -> torch.Tensor:
            import hashlib
            h = hashlib.sha256(text.encode()).hexdigest()
            g = torch.Generator(device=self.device)
            g.manual_seed(int(h[:8], 16))
            return torch.sign(torch.randn(10000, generator=g, device=self.device))

        hdv_rei = concept_hdv("rei")
        hdv_homem = concept_hdv("homem")
        hdv_mulher = concept_hdv("mulher")
        hdv_rainha = concept_hdv("rainha")

        # Analogia HDC classica: rainha_pred = rei * homem * mulher
        # Em bipolar: inverse = same, bind = multiply
        analogia = hdv_rei * hdv_homem * hdv_mulher  # bind(reverse(homem), rei)  mulher

        # Similaridade entre analogia e rainha real
        sim = torch.nn.functional.cosine_similarity(
            analogia.flatten().unsqueeze(0),
            hdv_rainha.flatten().unsqueeze(0)
        ).item()
        print(f"  rei - homem + mulher  rainha: similaridade={sim:.3f}")
        print(f"  Similaridade aleatoria esperada: ~0.0")

        result = TestResult("Analogia", sim > 0.3, {"similarity": sim},
                           f"Analogia HDV: sim={sim:.3f}")
        self.results.append(result)
        return result

    #  TESTE 5: MEMORIA (Catastrophic Forgetting) 

    def test_memory(self) -> TestResult:
        """Teste 5: Aprende novas imagens sem destruir as antigas.

        Mede PSNR antes e depois de fine-tune em novos dados.
        """
        print("\n[TESTE 5] Memoria  Catastrophic Forgetting")

        import torchvision
        from torch.utils.data import DataLoader, TensorDataset

        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

        # PSNR antes do fine-tune
        def measure_psnr(dataset, n=20):
            psnrs = []
            for idx in range(n):
                img_pil, _ = dataset[idx]
                tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feats = self.eng.feature_extractor(tensor).flatten(start_dim=1)
                    hdv, _, _ = self.pipe.encode_frame(feats[0])
                    recon = self.pipe.decode_bipolar(hdv.tensor)
                mse = ((recon - tensor[0]) ** 2).mean().item()
                psnrs.append(20 * np.log10(1.0 / np.sqrt(mse + 1e-10)))
            return sum(psnrs) / len(psnrs)

        psnr_before = measure_psnr(testset)
        print(f"  PSNR antes: {psnr_before:.1f}dB")

        # Fine-tune em 200 imagens novas (fashion MNIST)
        fmnist = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
        subset = [fmnist[i] for i in range(200)]
        fine_tune_loader = DataLoader(subset, batch_size=32, shuffle=True)
        optimizer = torch.optim.AdamW(self.pipe.decoder.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()
        self.pipe.decoder.train()
        for (img_pil, _) in subset:
            img_rgb = img_pil.convert("RGB")
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            feat = self.eng.feature_extractor(tensor).flatten(start_dim=1)
            pred = self.pipe.decoder(feat)
            loss = loss_fn(pred, tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.pipe.decoder.eval()

        psnr_after = measure_psnr(testset)
        print(f"  PSNR depois (apos fine-tune FMNIST): {psnr_after:.1f}dB")
        drop = psnr_before - psnr_after
        print(f"  DROP: {drop:.1f}dB {'' if drop > 3 else ''}")

        result = TestResult("Memoria", drop < 3, {"psnr_before": psnr_before, "psnr_after": psnr_after, "drop": drop})
        self.results.append(result)
        return result

    #  TESTE 6: COMPOSICAO 

    def test_composition(self) -> TestResult:
        """Teste 6: Bind de HDVs de objetos  decoder  cena.

        Codifica 2 objetos separadamente, faz bind, tenta reconstruir.
        """
        print("\n[TESTE 6] Composicao  bind de objetos em cena")
        import torchvision

        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

        # Pega 2 imagens de classes diferentes
        obj_a, lab_a = testset[0]  # aviao
        obj_b, lab_b = testset[3]  # gato

        tens_a = self.transform(obj_a).unsqueeze(0).to(self.device)
        tens_b = self.transform(obj_b).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f_a = self.eng.feature_extractor(tens_a).flatten(start_dim=1)
            f_b = self.eng.feature_extractor(tens_b).flatten(start_dim=1)
            hdv_a, _, _ = self.pipe.encode_frame(f_a[0])
            hdv_b, _, _ = self.pipe.encode_frame(f_b[0])

            # Bind = multiplicacao (XOR em bipolar)
            hdv_composite = hdv_a.tensor * hdv_b.tensor
            recon = self.pipe.decode_bipolar(hdv_composite)

        mse_a = ((recon - tens_a[0]) ** 2).mean().item()
        mse_b = ((recon - tens_b[0]) ** 2).mean().item()
        print(f"  Bind(aviao, gato): MSE vs aviao={mse_a:.4f}, vs gato={mse_b:.4f}")

        result = TestResult("Composicao", False, {"mse_vs_a": mse_a, "mse_vs_b": mse_b},
                           "Decoder precisa ser treinado para composicao")
        self.results.append(result)
        return result

    #  EXECUTAR TODOS 

    def run_all(self) -> List[TestResult]:
        """Executa os 6 testes e gera relatorio."""
        print("=" * 60)
        print("VALIDACAO DO HIPERESPACO HDV")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Decoder params: {sum(p.numel() for p in self.pipe.decoder.parameters()):,}")

        self.test_reconstruction()
        self.test_robustness()
        self.test_algebra()
        self.test_analogy()
        self.test_memory()
        self.test_composition()

        print("\n" + "=" * 60)
        print("RELATORIO FINAL")
        print("=" * 60)
        passed = 0
        for r in self.results:
            icon = "" if r.passed else ""
            print(f"  {icon} {r.name}: {r.details if r.details else ('OK' if r.passed else 'LIMITE')}")
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    print(f"     {k}: {v:.3f}")
            if r.passed:
                passed += 1
        print(f"\n{passed}/{len(self.results)} testes OK")
        print("Proximo passo depende dos resultados acima.")
        return self.results
