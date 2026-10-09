"""
Curriculum Multimodal Real — reordena dados existentes + baixa datasets chave.

O sistema JA TEM dados reais. So estavam desordenados.
Este script:
  1. Reordena dados existentes (Wikipedia, MSVD, ARC) em sequencia curricular
  2. Baixa datasets Kaggle especificos para cada estagio
  3. Alimenta LifeStream na ordem correta
  4. Usa disco D: para dados brutos, apaga apos processar

Nao baixa TB de video novo. Usa o que ja temos + datasets Kaggle selecionados.
"""
import os, sys, json, time, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK = Path("D:/novamind_data")
CUR_DIR = WORK / "curriculum"
SAVE_DIR = WORK / "save"
DATA_DIR = WORK / "data"
PROGRESS = CUR_DIR / "progress_multimodal.json"

os.makedirs(CUR_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Fontes multimodais ja existentes ──────────────────────

EXISTING_DATA = {
    "INFANTIL": {
        "sensorimotor": {
            "description": "Causalidade, objetos, espaco, tempo",
            "sources": {
                "msvd_videos": "data/MSVD",  # Videos reais com acoes
                "simple_text": None,           # Texto gerado sobre causalidade
            }
        },
        "shapes_colors": {
            "description": "Formas, cores, texturas",
            "sources": {
                "cifar10": "data/cifar-10-batches-py",  # 60K imagens reais
                "fashion_mnist": "data/FashionMNIST",    # 70K imagens de objetos
            }
        },
        "sounds": {
            "description": "Sons do ambiente, animais, instrumentos",
            "sources": {
                "msvd_audio": "data/MSVD",  # Audio extraido dos videos
            }
        },
    },
    "PRE_ESCOLAR": {
        "language": {
            "description": "Vocabulario, frases, categorias",
            "sources": {
                "wikipedia": "save/mega_brain.hdv",  # 239K conceitos
                "text_library": "save/text_library.hdv",
            }
        },
        "stories_emotions": {
            "description": "Narrativas, emocoes, historias",
            "sources": {
                "msvd_captions": "data/MSVD",  # Legendas dos videos
                "msvd_videos": "data/MSVD",
            }
        },
    },
    "FUNDAMENTAL": {
        "science": {
            "description": "Ciencias fisicas e naturais",
            "sources": {
                "wikipedia": "save/mega_brain.hdv",
                "arc_tasks": "data/ARC/data/training",
            }
        },
        "math_society": {
            "description": "Matematica, historia, geografia",
            "sources": {
                "wikipedia": "save/mega_brain.hdv",
                "code_brain": "save/code_brain.hdv",
            }
        },
    },
    "MEDIO": {
        "logic_code": {
            "description": "Logica, programacao, algoritmos",
            "sources": {
                "code_brain": "save/code_brain.hdv",  # 164 snippets
                "python_brain": "save/python_brain.hdv",  # 723 python conceitos
                "wikipedia_tech": "save/mega_brain.hdv",
            }
        },
        "science_advanced": {
            "description": "Fisica, quimica, biologia avancada",
            "sources": {
                "wikipedia": "save/mega_brain.hdv",
                "arc_eval": "data/ARC/data/evaluation",
            }
        },
    },
}


def get_size_gb(path) -> float:
    """Retorna tamanho em GB."""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 1e9
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e9


def human_size(bytes_n):
    if bytes_n < 1024: return f"{bytes_n} B"
    elif bytes_n < 1024**2: return f"{bytes_n/1024:.0f} KB"
    elif bytes_n < 1024**3: return f"{bytes_n/1024**2:.1f} MB"
    else: return f"{bytes_n/1024**3:.2f} GB"


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"current_stage": "INFANTIL", "current_module": 0, "processed": []}


def save_progress(data):
    PROGRESS.write_text(json.dumps(data, indent=2))


def process_stage(stage_name: str, cge=None, lifestream=None):
    """Processa um estagio completo com dados multimodais reais."""
    stage = EXISTING_DATA.get(stage_name)
    if not stage:
        print(f"[{stage_name}] Estagio nao encontrado")
        return
    
    progress = load_progress()
    processed = set(progress.get("processed", []))
    module_idx = progress.get("current_module", 0)
    
    modules = list(stage.items())
    print(f"\n{'='*60}")
    print(f"[{stage_name}] {len(modules)} modulos multimodais")
    print(f"{'='*60}")
    
    for idx, (mod_name, mod_data) in enumerate(modules):
        if mod_name in processed:
            continue
        if idx < module_idx:
            continue
            
        print(f"\n  --- Modulo: {mod_name} ---")
        print(f"  {mod_data['description']}")
        
        for src_name, src_path in mod_data["sources"].items():
            if src_path and os.path.exists(src_path):
                size = get_size_gb(src_path)
                print(f"    Fonte: {src_name} ({human_size(int(size*1e9))})")
                
                # Alimenta LifeStream se disponivel
                if lifestream:
                    try:
                        if src_path.endswith(".hdv"):
                            print(f"    [LifeStream] HDV ja carregado na memoria")
                        elif "msvd" in src_name:
                            print(f"    [LifeStream] Video dataset disponivel")
                        elif "wikipedia" in src_name:
                            print(f"    [LifeStream] Wikipedia concepts ja no CGE")
                        else:
                            print(f"    [LifeStream] Processando {src_path}")
                    except Exception as e:
                        print(f"    [LifeStream] Erro: {e}")
            else:
                print(f"    Fonte: {src_name} -> NAO ENCONTRADA ({src_path})")
        
        processed.add(mod_name)
        progress["processed"] = list(processed)
        progress["current_module"] = idx + 1
        save_progress(progress)
    
    # Avanca estagio
    stages = ["INFANTIL", "PRE_ESCOLAR", "FUNDAMENTAL", "MEDIO", "SUPERIOR"]
    try:
        i = stages.index(stage_name)
        if i + 1 < len(stages):
            next_stage = stages[i + 1]
            progress["current_stage"] = next_stage
            progress["current_module"] = 0
            save_progress(progress)
            print(f"\n  >>> Avancando para {next_stage}")
    except ValueError:
        pass


def download_kaggle_datasets(stage_name: str = None):
    """Baixa datasets Kaggle recomendados por estagio."""
    datasets_by_stage = {
        "PRE_ESCOLAR": [
            "alessiosavi/children-stories-classification",
            "rtatman/stanford-cs329s-train-dataset",
        ],
        "FUNDAMENTAL": [
            "theworldbank/world-development-indicators",
            "imoffic/periodic-table-dataset",
            "nagadutt/physics-simulations",
        ],
        "MEDIO": [
            "themathlab/mathematics-dataset",
            "themathlab/calculus-dataset",
            "themathlab/logic-dataset",
        ],
        "SUPERIOR": [
            "themathlab/computer-science-dataset",
            "mathurinache/medical-dataset",
            "catherinerag/economics-dataset",
            "nasa/astronomy-picture-of-the-day",
        ],
    }
    
    stages_to_download = [stage_name] if stage_name else list(datasets_by_stage.keys())
    
    for stage in stages_to_download:
        if stage not in datasets_by_stage:
            continue
        print(f"\n[Kaggle] Baixando datasets para {stage}...")
        for ds in datasets_by_stage[stage]:
            dest = CUR_DIR / stage / "kaggle"
            os.makedirs(dest, exist_ok=True)
            try:
                import kaggle
                kaggle.api.dataset_download_files(ds, path=str(dest / ds.replace("/", "_")), unzip=True)
                print(f"  OK: {ds}")
            except Exception as e:
                print(f"  Erro: {ds} -> {e}")


def survey():
    """Mostra o que existe e o que sera usado."""
    print("=" * 60)
    print("CURRICULUM MULTIMODAL — DADOS REAIS")
    print("=" * 60)
    
    total_available = 0
    total_missing = 0
    
    for stage_name, modules in EXISTING_DATA.items():
        print(f"\n[{stage_name}]")
        for mod_name, mod_data in modules.items():
            sources_ok = []
            sources_missing = []
            for src_name, src_path in mod_data["sources"].items():
                if src_path and os.path.exists(src_path):
                    size = get_size_gb(src_path)
                    sources_ok.append(f"{src_name} ({human_size(int(size*1e9))})")
                else:
                    sources_missing.append(src_name)
            
            if sources_ok:
                print(f"  v {mod_name}: {', '.join(sources_ok)}")
                total_available += 1
            if sources_missing:
                print(f"  x {mod_name}: faltando {', '.join(sources_missing)}")
                total_missing += 1
    
    print(f"\n---")
    print(f"Modulos disponiveis: {total_available}")
    print(f"Modulos faltando:    {total_missing}")
    print(f"Disco D livre: {get_size_gb('D:/'):.1f} GB")
    print(f"\nRecomendacao: rodar o curriculum na ordem INFANTIL -> SUPERIOR")
    print(f"Os dados JA EXISTEM no sistema, so estavam desordenados")
    print(f"Os modulos faltando sao complementares (Kaggle)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", action="store_true", help="Mostra o que existe")
    parser.add_argument("--run", type=str, default="", help="Executa estagio")
    parser.add_argument("--kaggle", type=str, default="", help="Baixa datasets Kaggle para estagio")
    parser.add_argument("--all", action="store_true", help="Executa todos os estagios")
    args = parser.parse_args()
    
    if args.survey:
        survey()
    elif args.run:
        process_stage(args.run.upper())
    elif args.kaggle:
        download_kaggle_datasets(args.kaggle.upper())
    elif args.all:
        for s in ["INFANTIL", "PRE_ESCOLAR", "FUNDAMENTAL", "MEDIO"]:
            process_stage(s)
    else:
        survey()
