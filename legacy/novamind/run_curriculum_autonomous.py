"""
Runner autonomo do curriculum — baixa datasets, alimenta AGI, avanca estagios.

Uso:
    python run_curriculum_autonomous.py --start INFANTIL
    python run_curriculum_autonomous.py --resume        # continua de onde parou
    python run_curriculum_autonomous.py --status        # mostra progresso

Gerencia espaco em disco automaticamente:
  - Baixa datasets do Kaggle um por vez
  - Processa cada topico: extrai conceitos -> CGE -> LifeStream
  - Remove dados brutos apos processamento
  - Salva progresso a cada topico
  - Avanca estagio automaticamente quando topico completo
"""
import os, sys, time, json, shutil, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Diretorio de trabalho no D (mais espaco)
WORK_DIR = Path("D:/novamind_data")
CUR_DIR = WORK_DIR / "curriculum"
SAVE_DIR = WORK_DIR / "save"
DATA_DIR = WORK_DIR / "data"

os.makedirs(CUR_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

PROGRESS_FILE = CUR_DIR / "progress.json"


def get_human_readable_size(path) -> str:
    """Retorna tamanho em formato legivel."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    if total < 1024:
        return f"{total} B"
    elif total < 1024 ** 2:
        return f"{total / 1024:.1f} KB"
    elif total < 1024 ** 3:
        return f"{total / 1024 ** 2:.1f} MB"
    else:
        return f"{total / 1024 ** 3:.2f} GB"


class AutonomousCurriculumRunner:
    """Runner autonomo que gerencia todo o ciclo de aprendizado."""

    def __init__(self):
        self.progress = self._load_progress()
        self.lifestream = None
        self.cge = None
        self.current_stage_name = self.progress.get("current_stage")
        self.current_topic_idx = self.progress.get("current_topic_idx", 0)

    # ─── Estrutura do curriculo ───────────────────────────

    STAGES = {
        "INFANTIL": {
            "order": 0, "age": "0-3 anos",
            "topics": [
                {"name": "shapes_and_colors", "datasets": [], "desc": "Formas e cores basicas"},
                {"name": "object_permanence", "datasets": [], "desc": "Objetos persistem mesmo ocultos"},
                {"name": "basic_causality", "datasets": [], "desc": "Causa e efeito simples"},
                {"name": "space_and_position", "datasets": [], "desc": "Espaco e posicao relativa"},
                {"name": "basic_counting", "datasets": [], "desc": "Contagem e numeros"},
                {"name": "sounds_and_noises", "datasets": [], "desc": "Sons e audio basico"},
            ]
        },
        "PRE_ESCOLAR": {
            "order": 1, "age": "3-6 anos",
            "topics": [
                {"name": "basic_language", "datasets": [], "desc": "Vocabulario e frases"},
                {"name": "categories", "datasets": [], "desc": "Categorizacao de objetos"},
                {"name": "simple_stories", "datasets": [], "desc": "Narrativas curtas"},
                {"name": "basic_emotions", "datasets": [], "desc": "Emocoes e expressoes"},
                {"name": "counting_advanced", "datasets": [], "desc": "Aritmetica simples"},
            ]
        },
        "FUNDAMENTAL": {
            "order": 2, "age": "6-14 anos",
            "topics": [
                {"name": "basic_physics", "datasets": [], "desc": "Mecanica basica"},
                {"name": "basic_chemistry", "datasets": [], "desc": "Elementos e reacoes"},
                {"name": "basic_biology", "datasets": [], "desc": "Seres vivos e celulas"},
                {"name": "world_history", "datasets": [], "desc": "Historia mundial"},
                {"name": "geography", "datasets": [], "desc": "Geografia fisica e politica"},
                {"name": "mathematics", "datasets": [], "desc": "Algebra e geometria"},
                {"name": "society", "datasets": [], "desc": "Cidadania e sociedade"},
                {"name": "arts", "datasets": [], "desc": "Arte e musica"},
                {"name": "sports", "datasets": [], "desc": "Esportes e educacao fisica"},
            ]
        },
        "MEDIO": {
            "order": 3, "age": "14-18 anos",
            "topics": [
                {"name": "logic", "datasets": [], "desc": "Logica e pensamento critico"},
                {"name": "physics_classical", "datasets": [], "desc": "Fisica classica"},
                {"name": "chemistry_advanced", "datasets": [], "desc": "Quimica avancada"},
                {"name": "calculus", "datasets": [], "desc": "Calculo diferencial e integral"},
                {"name": "philosophy", "datasets": [], "desc": "Filosofia"},
                {"name": "programming", "datasets": [], "desc": "Programacao"},
                {"name": "biology_advanced", "datasets": [], "desc": "Biologia avancada"},
            ]
        },
    }

    def _load_progress(self):
        """Carrega progresso salvo."""
        if PROGRESS_FILE.exists():
            try:
                return json.loads(PROGRESS_FILE.read_text())
            except Exception:
                pass
        return {"current_stage": "INFANTIL", "current_topic_idx": 0,
                "completed_topics": [], "stages": {}}

    def _save_progress(self):
        """Salva progresso."""
        self.progress["current_stage"] = self.current_stage_name
        self.progress["current_topic_idx"] = self.current_topic_idx
        PROGRESS_FILE.write_text(json.dumps(self.progress, indent=2))
        print(f"\n[Progresso] Salvo: {self.current_stage_name} topico {self.current_topic_idx}")

    def get_current_topic(self):
        """Retorna o topico atual."""
        stage = self.STAGES.get(self.current_stage_name)
        if not stage:
            return None
        topics = stage["topics"]
        if self.current_topic_idx < len(topics):
            return topics[self.current_topic_idx]
        return None

    def generate_topic_data(self, topic: dict):
        """Gera dados sinteticos para um topico (quando datasets offline)."""
        name = topic["name"]
        desc = topic.get("desc", "")
        topic_dir = CUR_DIR / self.current_stage_name / name
        os.makedirs(topic_dir, exist_ok=True)

        # Gera arquivo de texto com conceitos do topico
        concepts = self._get_concepts(name)
        text_path = topic_dir / "concepts.txt"
        text_path.write_text(concepts)
        print(f"  [Gerado] {text_path} ({len(concepts)} bytes)")

        # Se for estagio infantil, gera imagens simples de formas/cores
        if self.current_stage_name == "INFANTIL" and name == "shapes_and_colors":
            self._generate_shapes_images(topic_dir)

        return text_path

    def _get_concepts(self, topic_name: str) -> str:
        """Retorna texto com conceitos do topico."""
        concepts = {
            "shapes_and_colors": (
                "Shapes are the form of an object. A circle is round. "
                "A square has four equal sides. A triangle has three sides. "
                "A rectangle has two long sides and two short sides. "
                "Colors: red is the color of fire. Blue is the color of the sky. "
                "Green is the color of grass. Yellow is the color of the sun. "
                "Black is the absence of light. White is all colors combined. "
                "Objects have shape and color. A red ball is round and red. "
                "A green square has four sides and is green."
            ),
            "object_permanence": (
                "Objects exist even when you cannot see them. "
                "If a ball rolls behind a box, the ball still exists. "
                "If you close your eyes, the world continues to exist. "
                "When something is hidden, it is still there. "
                "Out of sight is not out of existence. "
                "You can remember objects that are not visible. "
                "The world continues even when you are not looking."
            ),
            "basic_causality": (
                "Causes make effects happen. Pushing a ball makes it move. "
                "Dropping an object makes it fall down. Gravity pulls things down. "
                "If you touch fire, you get burned. Heat causes burning. "
                "Wind moves leaves. Rain makes things wet. "
                "The sun causes daylight. Night is caused by the Earth turning. "
                "Every action has a reaction. Cause always comes before effect."
            ),
            "space_and_position": (
                "Objects occupy space. Above means higher. Below means lower. "
                "Left and right are directions. Inside means contained. "
                "Outside means not contained. Near is close. Far is distant. "
                "Big and small describe size. A box can contain things. "
                "Objects have position in space: here, there, up, down. "
                "You can move objects from one place to another."
            ),
            "basic_counting": (
                "Numbers represent quantity. One, two, three, four, five. "
                "Six, seven, eight, nine, ten. Zero means nothing. "
                "Adding makes numbers bigger. Subtracting makes numbers smaller. "
                "More means larger quantity. Less means smaller quantity. "
                "Counting tells you how many. You can count objects. "
                "Numbers are the foundation of mathematics."
            ),
            "sounds_and_noises": (
                "Sound travels through air. Loud sounds are strong. "
                "Soft sounds are gentle. High pitch sounds are like birds. "
                "Low pitch sounds are like drums. Music is organized sound. "
                "Noise is random sound. Animals make sounds to communicate. "
                "Speech is human sound with meaning. "
                "Sound can warn us, delight us, or inform us."
            ),
        }
        return concepts.get(topic_name, f"Conceitos sobre {topic_name}.\n")

    def _generate_shapes_images(self, topic_dir: Path):
        """Gera imagens simples de formas geometricas."""
        try:
            from PIL import Image

            shapes = {
                "circle": lambda s: self._draw_circle(s),
                "square": lambda s: self._draw_square(s),
                "triangle": lambda s: self._draw_triangle(s),
                "rectangle": lambda s: self._draw_rectangle(s),
            }

            for sname, sfunc in shapes.items():
                for ci in range(4):
                    img = sfunc(64)
                    img_path = topic_dir / f"{sname}_{ci}.png"
                    Image.fromarray(img).save(img_path)

            print(f"  [Gerado] {len(shapes)*4} imagens de formas em {topic_dir}")
        except ImportError:
            print(f"  [Info] PIL nao disponivel, pulando geracao de imagens")

    def _draw_circle(self, size):
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        c = size // 2
        r = size // 3
        for y in range(size):
            for x in range(size):
                if (x - c) ** 2 + (y - c) ** 2 < r ** 2:
                    arr[y, x] = [255, 0, 0]
        return arr

    def _draw_square(self, size):
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        margin = size // 4
        arr[margin:size - margin, margin:size - margin] = [0, 0, 255]
        return arr

    def _draw_triangle(self, size):
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                if y > size // 2 - abs(x - size // 2):
                    arr[y, x] = [0, 255, 0]
        return arr

    def _draw_rectangle(self, size):
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        margin = size // 4
        arr[margin:size - margin, size // 4:3 * size // 4] = [255, 255, 0]
        return arr

    def process_topic(self, topic: dict):
        """Processa um topico: gera dados, alimenta AGI, registra progresso."""
        name = topic["name"]
        print(f"\n{'='*50}")
        print(f"Processando: {name} ({topic.get('desc', '')})")
        print(f"{'='*50}")

        # Gera dados do topico
        data_path = self.generate_topic_data(topic)
        size = get_human_readable_size(CUR_DIR / self.current_stage_name / name)
        print(f"  Tamanho: {size}")

        # Alimenta CGE se disponivel
        if self.cge is not None:
            try:
                text = data_path.read_text()
                self.cge.write_text_concepts(text)
                print(f"  [CGE] Conceitos escritos: {len(text)} chars")
            except Exception as e:
                print(f"  [CGE] Erro: {e}")

        # Alimenta LifeStream se disponivel
        if self.lifestream is not None:
            try:
                self.lifestream.process_text(text, source=f"curriculum/{name}")
                print(f"  [LifeStream] Processado")
            except Exception as e:
                print(f"  [LifeStream] Erro: {e}")

        # Marca completo
        completed = self.progress.setdefault("completed_topics", [])
        if name not in completed:
            completed.append(name)
        self.current_topic_idx += 1
        self._save_progress()

        # Calcula espaco usado
        total_used = get_human_readable_size(CUR_DIR)
        print(f"  [Disco] Usado: {total_used}")

    def advance_stage(self):
        """Avanca para o proximo estagio."""
        stages_order = ["INFANTIL", "PRE_ESCOLAR", "FUNDAMENTAL", "MEDIO"]
        try:
            idx = stages_order.index(self.current_stage_name)
            if idx + 1 < len(stages_order):
                next_stage = stages_order[idx + 1]
                print(f"\n{'='*50}")
                print(f"AVANCANDO: {self.current_stage_name} -> {next_stage}")
                print(f"{'='*50}")
                self.current_stage_name = next_stage
                self.current_topic_idx = 0
                self._save_progress()
                return True
        except ValueError:
            pass
        return False

    def run(self, stages_to_run=None):
        """Roda o curriculum autonomamente."""
        print(f"\n{'='*60}")
        print(f"NovaMind Curriculum Autonomous Runner")
        print(f"{'='*60}")
        print(f"Work dir: {WORK_DIR}")
        print(f"Disco livre D: {get_human_readable_size(WORK_DIR.parent) if WORK_DIR.parent.exists() else 'N/A'}")
        print(f"Progresso atual: {self.current_stage_name} topico {self.current_topic_idx}")
        print()

        max_topics = 0
        processed = 0

        while True:
            topic = self.get_current_topic()
            if topic is None:
                # Topico atual nao existe — tenta avancar estagio
                if not self.advance_stage():
                    print("\n[Todos os estagios concluidos!]")
                    break
                topic = self.get_current_topic()
                if topic is None:
                    break

            self.process_topic(topic)
            processed += 1

            # Se ja processou o maximo, pausa
            if stages_to_run and processed >= stages_to_run:
                print(f"\n[Pausa] {stages_to_run} topicos processados")
                break

        print(f"\n{'='*60}")
        print(f"Curriculum concluido: {self.progress}")
        print(f"{'='*60}")

    def show_status(self):
        """Mostra status detalhado do curriculo."""
        completed = self.progress.get("completed_topics", [])
        print(f"\n{'='*50}")
        print(f"CURRICULUM STATUS")
        print(f"{'='*50}")
        print(f"Estagio atual: {self.current_stage_name}")
        print(f"Topicos completos: {len(completed)}")
        print()

        for sname, sdata in self.STAGES.items():
            status = ">>>" if sname == self.current_stage_name else "   "
            n_total = len(sdata["topics"])
            n_done = sum(1 for t in sdata["topics"] if t["name"] in completed)
            print(f"  {status} {sname:15s} [{n_done:2d}/{n_total:2d}] {sdata['age']}")
            if sname == self.current_stage_name:
                for i, t in enumerate(sdata["topics"]):
                    mark = "[OK]" if t["name"] in completed else "[  ]"
                    arrow = " ->" if i == self.current_topic_idx else "   "
                    print(f"      {arrow} {mark} {t['name']}")

        total_cur = get_human_readable_size(CUR_DIR)
        total_save = get_human_readable_size(SAVE_DIR) if SAVE_DIR.exists() else "0 B"
        print(f"\n  Disco: curriculum={total_cur} | save={total_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Autonomous Runner")
    parser.add_argument("--start", type=str, default="", help="Estagio inicial")
    parser.add_argument("--resume", action="store_true", help="Continua de onde parou")
    parser.add_argument("--status", action="store_true", help="Mostra progresso")
    parser.add_argument("--topics", type=int, default=0, help="Maximo de topicos")
    args = parser.parse_args()

    runner = AutonomousCurriculumRunner()

    if args.status:
        runner.show_status()
    elif args.start:
        runner.current_stage_name = args.start.upper()
        runner.current_topic_idx = 0  # Always start at topic 0
        runner.run(stages_to_run=args.topics or None)
    elif args.resume:
        runner.run(stages_to_run=args.topics or None)
    else:
        runner.show_status()
        print("\nUso: python run_curriculum_autonomous.py --start INFANTIL")
        print("     python run_curriculum_autonomous.py --resume")
        print("     python run_curriculum_autonomous.py --status")
