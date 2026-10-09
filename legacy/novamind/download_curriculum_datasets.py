"""Baixa datasets reais do Kaggle para cada dominio do curriculo."""
import os, sys, json, kaggle
from pathlib import Path

BASE = Path("D:/novamind_data/curriculum")
PROGRESS = BASE / "download_progress.json"
os.makedirs(BASE, exist_ok=True)

# Datasets reais verificados que existem no Kaggle
DOMAIN_DATASETS = {
    "INFANTIL": [
        # Sem datasets especificos — dados gerados localmente
    ],
    "PRE_ESCOLAR": [
        ("rtatman/lego-database", "categorization"),
    ],
    "FUNDAMENTAL": [
        ("imoffic/periodic-table-dataset", "chemistry"),
        ("theworldbank/world-development-indicators", "geography"),
        ("nagadutt/physics-simulations", "physics"),
    ],
    "MEDIO": [
        ("iceatgt/ap-computer-science-a-exam-dataset", "programming"),
        ("mujtabamatin/computer-science-theory-qa-dataset", "cs_theory"),
        ("khushikyad001/finance-and-economics-dataset-2000-present", "economics"),
    ],
    "SUPERIOR": [
        ("ujjwalaggarwal402/medicine-dataset", "medicine"),
        ("younusmohamed/particle-physics-event-classification-dataset", "physics"),
        ("kamalesh1997/weather-astronomy-dataset", "astronomy"),
        ("basu369victor/prediction-of-asteroid-diameter", "astronomy"),
        ("steve1215rogg/student-lifestyle-dataset", "psychology"),
    ],
}


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"downloaded": []}


def save_progress(data):
    PROGRESS.write_text(json.dumps(data, indent=2))


def download_all():
    progress = load_progress()
    downloaded = set(progress.get("downloaded", []))
    
    for stage, datasets in DOMAIN_DATASETS.items():
        for ds_spec in datasets:
            ds_name, category = ds_spec
            
            if ds_name in downloaded:
                print(f"  [SKIP] {ds_name} ja baixado")
                continue
            
            dest = BASE / stage / category
            os.makedirs(dest, exist_ok=True)
            
            try:
                print(f"  [DOWNLOAD] {ds_name} -> {dest}")
                kaggle.api.dataset_download_files(ds_name, path=str(dest), unzip=True, quiet=True)
                
                # Conta arquivos baixados
                n_files = len(list(dest.rglob("*")))
                size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
                print(f"    OK: {n_files} arquivos, {size_mb:.1f} MB")
                
                downloaded.add(ds_name)
                progress["downloaded"] = list(downloaded)
                save_progress(progress)
                
            except Exception as e:
                print(f"    ERRO: {e}")
    
    print(f"\nTotal baixado: {len(downloaded)} datasets")


def show_status():
    progress = load_progress()
    downloaded = set(progress.get("downloaded", []))
    
    print("=" * 60)
    print("STATUS: Datasets do Curriculo")
    print("=" * 60)
    
    for stage, datasets in DOMAIN_DATASETS.items():
        print(f"\n[{stage}]")
        for ds_spec in datasets:
            ds_name, category = ds_spec
            status = "OK" if ds_name in downloaded else "PENDENTE"
            dest = BASE / stage / category
            
            if dest.exists():
                n_files = len(list(dest.rglob("*")))
                size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
                print(f"  [{status}] {ds_name:50s} {size_mb:.1f} MB ({n_files} files)")
            else:
                print(f"  [{status}] {ds_name:50s} -")

    print(f"\nTotal: {len(downloaded)}/{sum(len(v) for v in DOMAIN_DATASETS.values())} datasets")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_status()
    else:
        download_all()
        show_status()
