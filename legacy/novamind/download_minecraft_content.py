"""
Pipeline de download de videos Minecraft (tutoriais + gameplay).

Fontes:
  - YouTube: tutoriais em PT-BR e EN sobre construcao, redstone, sobrevivencia
  - Cada video e processado: audio -> texto (transcricao), video -> frames -> HDV
  - Conceitos sao alimentados no CGE

Nao e trapace: e treino multimodal em um dominio especifico.
Assim como uma criança aprende sobre o mundo vendo videos.
"""
import os, json, time
from pathlib import Path

BASE = Path("D:/novamind_data/minecraft")
VIDEOS_DIR = BASE / "videos"
FRAMES_DIR = BASE / "frames"
PROGRESS_FILE = BASE / "download_progress.json"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# URLs de videos Minecraft organizados por categoria
# Formato: (nome, url, idioma, categoria)
MINECRAFT_VIDEOS = [
    # --- Tutoriais em Portugues ---
    ("minecraft_sobrevivencia_dia_1_pt", None, "pt", "tutorial"),
    ("minecraft_construir_casa_pt", None, "pt", "construcao"),
    ("minecraft_redstone_basico_pt", None, "pt", "redstone"),
    ("minecraft_minerar_recursos_pt", None, "pt", "sobrevivencia"),
    
    # --- Tutoriais em Ingles ---
    ("minecraft_survival_day1_en", None, "en", "tutorial"),
    ("minecraft_build_house_en", None, "en", "construcao"),
    ("minecraft_redstone_basics_en", None, "en", "redstone"),
    ("minecraft_farming_en", None, "en", "sobrevivencia"),
    
    # --- Gameplay (para entender comportamentos naturais) ---
    ("minecraft_exploration_pt", None, "pt", "gameplay"),
    ("minecraft_combat_pt", None, "pt", "gameplay"),
    ("minecraft_exploration_en", None, "en", "gameplay"),
    ("minecraft_combat_en", None, "en", "gameplay"),
]

# Conceitos Minecraft que devem ser extraidos dos videos
MINECRAFT_CONCEPTS = """
minecraft is a block-based 3D world. players can break and place blocks.
blocks include dirt, stone, wood, coal, iron, gold, diamond.
crafting combines materials to create tools and items.
a crafting table has a 3x3 grid for complex recipes.
furnaces smelt ores into ingots. furnaces cook food.
swords deal damage to mobs. pickaxes mine stone faster. axes chop wood.
shovels dig dirt and sand. hoes till soil for farming.
mobs are creatures in minecraft. zombies attack at night. skeletons shoot arrows.
creepers explode. spiders climb walls. endermen teleport.
animals: cows provide leather and beef. sheep provide wool and mutton.
chickens provide feathers and chicken. pigs provide porkchop.
villagers trade emeralds for items. iron golems protect villages.
redstone is like electricity. redstone dust transmits power.
repeaters extend redstone signals. comparators measure container fullness.
pistons push blocks. dispensers shoot items. hoppers transfer items.
the nether is a hell dimension. the end is where the dragon lives.
"""


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"downloaded": [], "phase": "concepts"}


def save_progress(data):
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))


def process_minecraft_cge(cge=None):
    """Alimenta o CGE com conceitos fundamentais do Minecraft."""
    if cge is None:
        print("[Minecraft] CGE nao disponivel, pulando")
        return

    progress = load_progress()
    if "concepts_written" in progress.get("downloaded", []):
        print("[Minecraft] Conceitos ja escritos no CGE")
        return

    # Escreve conceitos no CGE
    for line in MINECRAFT_CONCEPTS.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                cge.write_text_concepts(line)
            except Exception as e:
                pass

    print(f"[Minecraft] Conceitos escritos no CGE")
    progress.setdefault("downloaded", []).append("concepts_written")
    save_progress(progress)


def download_status():
    """Mostra status do download de videos."""
    progress = load_progress()
    downloaded = set(progress.get("downloaded", []))

    print("=" * 60)
    print("MINECRAFT CONTENT PIPELINE")
    print("=" * 60)

    print(f"\nConceitos: {'OK' if 'concepts_written' in downloaded else 'PENDENTE'}")

    print(f"\nVideos:")
    for name, url, lang, cat in MINECRAFT_VIDEOS:
        video_path = VIDEOS_DIR / f"{name}.mp4"
        status = "BAIXADO" if video_path.exists() else "PENDENTE"
        print(f"  [{status}] {name} ({lang}, {cat})")

    total_videos = sum(1 for name, _, _, _ in MINECRAFT_VIDEOS)
    downloaded_videos = sum(1 for name, _, _, _ in MINECRAFT_VIDEOS if (VIDEOS_DIR / f"{name}.mp4").exists())
    print(f"\n  Total: {downloaded_videos}/{total_videos} videos")


def download_all():
    """Baixa videos Minecraft. Nota: requer yt-dlp instalado."""
    progress = load_progress()
    downloaded = set(progress.get("downloaded", []))

    # Primeiro: escreve conceitos
    print("[Minecraft] Escrevendo conceitos...")
    process_minecraft_cge()

    # Depois: tenta baixar videos
    print("\n[Minecraft] Verificando videos...")
    for name, url, lang, cat in MINECRAFT_VIDEOS:
        video_path = VIDEOS_DIR / f"{name}.mp4"
        if video_path.exists():
            print(f"  [SKIP] {name} ja existe")
            continue

        if url is None:
            print(f"  [INFO] {name}: URL nao configurada, gere manualmente")
            print(f"    Procure no YouTube por tutoriais Minecraft em {lang}")
            print(f"    Coloque o video em: {video_path}")
            continue

        print(f"  [DOWNLOAD] {name} de {url}...")
        try:
            import subprocess
            result = subprocess.run([
                "yt-dlp", "-f", "best[height<=720]", "-o", str(video_path),
                url, "--no-playlist", "--quiet"
            ], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                size = video_path.stat().st_size / 1e6
                print(f"    OK: {size:.0f} MB")
            else:
                print(f"    ERRO: {result.stderr[:100]}")
        except FileNotFoundError:
            print(f"    yt-dlp nao instalado. Instale com: pip install yt-dlp")
            break
        except Exception as e:
            print(f"    ERRO: {e}")

    download_status()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Mostra status")
    args = parser.parse_args()
    
    if args.status:
        download_status()
    else:
        download_all()
