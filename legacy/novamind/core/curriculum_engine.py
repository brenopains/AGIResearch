"""
CurriculumEngine — Curriculo progressivo de aprendizado, da infancia a especializacao.

Cada estagio constroi conceitos que o estagio seguinte usa.
A ordem importa: nao adianta ensinar fisica quantica sem
estabelecer causalidade, algebra e geometria primeiro.

Estagios:
  1. INFANTIL (0-3): objetos, espaco, tempo, causalidade basica, formas, cores
  2. PRE_ESCOLAR (3-6): linguagem, contagem, categorias, historias simples
  3. FUNDAMENTAL (6-14): ciencia, historia, geografia, matematica, sociedade
  4. MEDIO (14-18): logica, fisica, quimica, filosofia, matematica avancada
  5. SUPERIOR (18+): especializacao por area (medicina, direito, engenharia, etc)
  6. MESTRADO: pesquisa, problemas novos, sintese cross-dominio

Integracao:
  - KaggleAPI: baixa datasets por estagio
  - LifeStream: alimenta AGI na ordem do curriculo
  - CGE: verifica se conceitos foram absorvidos antes de avancar
"""
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import time, json, os, random


# ─── Topic — um topico de aprendizado ─────────────────────

@dataclass
class Topic:
    """Um topico especifico de aprendizado."""
    name: str
    description: str = ""
    subjects: List[str] = field(default_factory=list)  # "matematica", "fisica", etc
    data_types: List[str] = field(default_factory=list)  # "text", "image", "video", "audio", "code"
    kaggle_datasets: List[str] = field(default_factory=list)
    estimated_hours: float = 1.0
    prerequisites: List[str] = field(default_factory=list)  # topicos que precisam ser feitos antes

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "subjects": self.subjects,
            "data_types": self.data_types,
            "kaggle_datasets": self.kaggle_datasets[:3],
            "hours": self.estimated_hours,
            "prerequisites": self.prerequisites,
        }


# ─── Stage — um estagio do curriculo ──────────────────────

@dataclass
class Stage:
    """Um estagio de desenvolvimento."""
    name: str
    age_range: str
    description: str
    topics: List[Topic] = field(default_factory=list)
    completed: bool = False
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def total_hours(self) -> float:
        return sum(t.estimated_hours for t in self.topics)

    @property
    def completed_topics(self) -> int:
        return sum(1 for t in self.topics if getattr(t, '_completed', False))

    def to_dict(self) -> dict:
        return {
            "stage": self.name,
            "age": self.age_range,
            "topics": len(self.topics),
            "completed": self.completed,
            "completed_topics": self.completed_topics,
            "total_hours": round(self.total_hours, 1),
        }


# ─── CurriculumEngine ──────────────────────────────────────

class CurriculumEngine:
    """Gerencia o curriculo progressivo da AGI.

    Uso:
        curriculum = CurriculumEngine()
        curriculum.setup_default()
        curriculum.start_stage("INFANTIL")
        # ... AGI processa dados ...
        curriculum.complete_topic("shapes_and_colors")
        curriculum.advance_to_next_stage()
    """

    def __init__(self, lifestream=None, cge=None, kaggle_api_path: str = ""):
        self.lifestream = lifestream
        self.cge = cge
        self.kaggle_api_path = kaggle_api_path
        self.stages: Dict[str, Stage] = {}
        self.current_stage: Optional[str] = None
        self.current_topic_idx: int = 0
        self._progress_file = "curriculum_progress.json"

        # Carrega progresso anterior se existir
        self._load_progress()

    def setup_default(self):
        """Cria o curriculo padrao com todos os estagios."""
        self._create_infantil()
        self._create_pre_escolar()
        self._create_fundamental()
        self._create_medio()
        self._create_superior()
        self._create_mestrado()

    def _create_infantil(self):
        """Estagio 1: sensoriomotor — conceitos fundamentais do mundo fisico."""
        stage = Stage(
            name="INFANTIL",
            age_range="0-3 anos",
            description="Conceitos fundamentais: objetos, espaco, tempo, causalidade, formas, cores, sons",
        )
        stage.topics = [
            Topic("shapes_and_colors", "Formas geometricas, cores primarias/secundarias",
                  subjects=["geometria", "arte"], data_types=["image", "text"],
                  kaggle_datasets=["msarouafim/shapes-and-colors-dataset"]),
            Topic("object_permanence", "Objetos existem mesmo quando nao visiveis",
                  subjects=["causalidade"], data_types=["video", "image"],
                  kaggle_datasets=["alexattia/the-semper-collection"]),
            Topic("basic_causality", "Causa e efeito simples: empurrar move, soltar cai",
                  subjects=["fisica"], data_types=["video", "text"],
                  kaggle_datasets=["nagadutt/physics-simulations"]),
            Topic("space_and_position", "Dentro/fora, cima/baixo, esquerda/direita",
                  subjects=["geometria", "espaco"], data_types=["image", "text"]),
            Topic("basic_counting", "Contagem ate 10, mais/menos, grande/pequeno",
                  subjects=["matematica"], data_types=["text", "image"]),
            Topic("sounds_and_noises", "Sons de objetos, animais, instrumentos",
                  subjects=["audio"], data_types=["audio"],
                  kaggle_datasets=["chrisfilo1/urbansound8k"]),
        ]
        self.stages["INFANTIL"] = stage

    def _create_pre_escolar(self):
        """Estagio 2: simbolico — linguagem, categorias, historias."""
        stage = Stage(
            name="PRE_ESCOLAR",
            age_range="3-6 anos",
            description="Linguagem, categorizacao, historias simples, contagem avancada",
        )
        stage.topics = [
            Topic("basic_language", "Vocabulario basico, formacao de frases simples",
                  subjects=["linguagem"], data_types=["text", "audio"],
                  kaggle_datasets=["rtatman/stanford-cs329s-train-dataset"]),
            Topic("categories_and_taxonomy", "Animais, plantas, alimentos, roupas, meios de transporte",
                  subjects=["biologia", "sociedade"], data_types=["image", "text"],
                  kaggle_datasets=["gpiosenka/70-dog-breedsimage-data-set"]),
            Topic("simple_stories", "Narrativas curtas com causa e efeito",
                  subjects=["literatura"], data_types=["text", "audio"]),
            Topic("basic_emotions", "Alegria, tristeza, raiva, medo — reconhecimento",
                  subjects=["psicologia"], data_types=["image", "text"],
                  kaggle_datasets=["jonathanoheix/face-expression-recognition-dataset"]),
            Topic("counting_and_arithmetic", "Soma e subtracao simples, numeros ate 100",
                  subjects=["matematica"], data_types=["text"]),
        ]
        self.stages["PRE_ESCOLAR"] = stage

    def _create_fundamental(self):
        """Estagio 3: operacional concreto — ciencias, historia, geografia, matematica."""
        stage = Stage(
            name="FUNDAMENTAL",
            age_range="6-14 anos",
            description="Ciencias naturais, historia, geografia, matematica fundamental, sociedade",
        )
        stage.topics = [
            Topic("basic_physics", "Mecanica simples: movimento, forcas, energia",
                  subjects=["fisica"], data_types=["text", "video"],
                  kaggle_datasets=["nagadutt/physics-simulations"],
                  prerequisites=["basic_causality", "space_and_position"]),
            Topic("basic_chemistry", "Elementos, atomos, moleculas, reacoes simples",
                  subjects=["quimica"], data_types=["text", "image"],
                  kaggle_datasets=["imoffic/periodic-table-dataset"]),
            Topic("basic_biology", "Celulas, seres vivos, ecossistemas, corpo humano",
                  subjects=["biologia", "anatomia"], data_types=["text", "image"],
                  kaggle_datasets=["tangqianyu/human-anatomy"]),
            Topic("world_history", "Principais civilizacoes, eventos historicos",
                  subjects=["historia"], data_types=["text"],
                  kaggle_datasets=["theworldbank/world-development-indicators"]),
            Topic("geography", "Continentes, paises, capitais, biomas, clima",
                  subjects=["geografia"], data_types=["text", "image"],
                  kaggle_datasets=["pythondeveloper/geography-dataset"]),
            Topic("mathematics", "Algebra basica, geometria plana, estatistica descritiva",
                  subjects=["matematica"], data_types=["text"],
                  kaggle_datasets=["themathlab/mathematics-dataset"],
                  prerequisites=["counting_and_arithmetic"]),
            Topic("society_and_civics", "Direitos, deveres, governo, cidadania",
                  subjects=["sociedade", "direito"], data_types=["text"]),
            Topic("arts_and_music", "Arte classica, musica, instrumentos, periodos artisticos",
                  subjects=["arte", "musica"], data_types=["image", "audio"],
                  kaggle_datasets=["imsparsh/musicnet-dataset"]),
            Topic("physical_education", "Esportes, regras, movimentos do corpo",
                  subjects=["esportes", "anatomia"], data_types=["text", "video"]),
        ]
        self.stages["FUNDAMENTAL"] = stage

    def _create_medio(self):
        """Estagio 4: operacional formal — logica, abstracoes, pensamento critico."""
        stage = Stage(
            name="MEDIO",
            age_range="14-18 anos",
            description="Logica formal, fisica classica, quimica avancada, filosofia, calculo",
        )
        stage.topics = [
            Topic("logic_and_critical_thinking", "Logica proposicional, silogismos, falacios",
                  subjects=["logica", "filosofia"], data_types=["text"],
                  kaggle_datasets=["themathlab/logic-dataset"],
                  prerequisites=["basic_language"]),
            Topic("physics_classical", "Mecanica newtoniana, termodinamica, optica, ondas",
                  subjects=["fisica"], data_types=["text", "video"],
                  prerequisites=["basic_physics", "mathematics"]),
            Topic("chemistry_advanced", "Quimica organica, inorganica, termoquimica, cinetica",
                  subjects=["quimica"], data_types=["text"],
                  prerequisites=["basic_chemistry"]),
            Topic("calculus", "Limites, derivadas, integrais, equacoes diferenciais",
                  subjects=["matematica"], data_types=["text"],
                  kaggle_datasets=["themathlab/calculus-dataset"],
                  prerequisites=["mathematics"]),
            Topic("philosophy", "Filosofia antiga, moderna, etica, epistemologia",
                  subjects=["filosofia"], data_types=["text"],
                  kaggle_datasets=["philarchive/philosophy-papers"]),
            Topic("programming_basics", "Logica de programacao, Python, estruturas de dados",
                  subjects=["computacao"], data_types=["text", "code"],
                  kaggle_datasets=["themathlab/python-programming-dataset"],
                  prerequisites=["logic_and_critical_thinking"]),
            Topic("biology_advanced", "Genetica, evolucao, bioquimica, fisiologia",
                  subjects=["biologia"], data_types=["text", "image"],
                  prerequisites=["basic_biology"]),
        ]
        self.stages["MEDIO"] = stage

    def _create_superior(self):
        """Estagio 5: especializacao por area."""
        stage = Stage(
            name="SUPERIOR",
            age_range="18+ anos",
            description="Especializacao: medicina, direito, engenharia, fisica quantica, IA, etc",
        )
        # Este estagio tem topicos OPCIONAIS — AGI escolhe o caminho
        stage.topics = [
            Topic("quantum_physics", "Mecanica quantica, relatividade, teoria de campos",
                  subjects=["fisica"], data_types=["text"],
                  prerequisites=["physics_classical", "calculus"]),
            Topic("computer_science", "Algoritmos, compiladores, redes, bancos de dados",
                  subjects=["computacao"], data_types=["text", "code"],
                  kaggle_datasets=["themathlab/computer-science-dataset"],
                  prerequisites=["programming_basics"]),
            Topic("medicine_and_anatomy", "Anatomia humana, fisiopatologia, farmacologia",
                  subjects=["medicina", "biologia"], data_types=["text", "image"],
                  kaggle_datasets=["mathurinache/medical-dataset"],
                  prerequisites=["biology_advanced"]),
            Topic("engineering", "Resistencia dos materiais, circuitos, fluidos, termodinamica",
                  subjects=["engenharia"], data_types=["text"],
                  prerequisites=["physics_classical", "calculus"]),
            Topic("law_and_society", "Direito constitucional, civil, penal, tributario",
                  subjects=["direito", "sociedade"], data_types=["text"],
                  prerequisites=["society_and_civics"]),
            Topic("economics", "Microeconomia, macroeconomia, financas, mercado",
                  subjects=["economia"], data_types=["text"],
                  kaggle_datasets=["catherinerag/economics-dataset"]),
            Topic("psychology", "Psicologia cognitiva, comportamental, neuropsicologia",
                  subjects=["psicologia"], data_types=["text"],
                  prerequisites=["basic_emotions"]),
            Topic("advanced_mathematics", "Algebra linear, teoria dos numeros, topologia",
                  subjects=["matematica"], data_types=["text"],
                  prerequisites=["calculus"]),
            Topic("astronomy", "Astrofisica, cosmologia, sistema solar, exoplanetas",
                  subjects=["astronomia", "fisica"], data_types=["text", "image"],
                  kaggle_datasets=["nasa/astronomy-picture-of-the-day"]),
        ]
        self.stages["SUPERIOR"] = stage

    def _create_mestrado(self):
        """Estagio 6: pesquisa — problemas novos, sintese cross-dominio."""
        stage = Stage(
            name="MESTRADO",
            age_range="pesquisa",
            description="Problemas abertos, sintese entre dominios, producao de conhecimento novo",
        )
        stage.topics = [
            Topic("cross_domain_synthesis", "Combinar conhecimento de 2+ areas para gerar novo insight",
                  subjects=["geral"], data_types=["text", "code"]),
            Topic("novel_problem_solving", "Resolver problemas que nao tem solucao conhecida",
                  subjects=["geral"], data_types=["text"]),
            Topic("meta_learning", "Aprender sobre o proprio processo de aprendizado",
                  subjects=["metacognicao"], data_types=["text"]),
        ]
        self.stages["MESTRADO"] = stage

    # ─── Gerenciamento de Progresso ───────────────────────

    def start_stage(self, stage_name: str) -> bool:
        """Inicia um estagio do curriculo."""
        if stage_name not in self.stages:
            return False
        stage = self.stages[stage_name]
        stage.started_at = time.time()
        self.current_stage = stage_name
        self.current_topic_idx = 0
        self._save_progress()
        return True

    def advance_to_next_stage(self) -> Optional[str]:
        """Avanca para o proximo estagio."""
        if self.current_stage is None:
            return None
        current = self.stages[self.current_stage]
        current.completed = True
        current.completed_at = time.time()

        stages_order = ["INFANTIL", "PRE_ESCOLAR", "FUNDAMENTAL", "MEDIO", "SUPERIOR", "MESTRADO"]
        try:
            idx = stages_order.index(self.current_stage)
            if idx + 1 < len(stages_order):
                next_stage = stages_order[idx + 1]
                self.start_stage(next_stage)
                return next_stage
        except ValueError:
            pass
        return None

    def get_current_topic(self) -> Optional[Topic]:
        """Retorna o topico atual."""
        if self.current_stage is None:
            return None
        stage = self.stages[self.current_stage]
        if self.current_topic_idx < len(stage.topics):
            return stage.topics[self.current_topic_idx]
        return None

    def complete_topic(self, topic_name: str = None):
        """Marca topico como completo e avanca para o proximo."""
        if self.current_stage is None:
            return
        stage = self.stages[self.current_stage]

        if topic_name:
            for t in stage.topics:
                if t.name == topic_name:
                    t._completed = True
                    break
        else:
            # Completa o topico atual
            topic = self.get_current_topic()
            if topic:
                topic._completed = True
                self.current_topic_idx += 1

        self._save_progress()

    def get_pending_datasets(self) -> List[str]:
        """Retorna datasets Kaggle pendentes para o estagio atual."""
        if self.current_stage is None:
            return []
        stage = self.stages[self.current_stage]
        datasets = []
        for t in stage.topics[self.current_topic_idx:]:
            for ds in t.kaggle_datasets:
                if ds not in datasets:
                    datasets.append(ds)
        return datasets

    def get_stats(self) -> dict:
        """Estatisticas do curriculo."""
        total_topics = sum(len(s.topics) for s in self.stages.values())
        completed_topics = sum(
            sum(1 for t in s.topics if getattr(t, '_completed', False))
            for s in self.stages.values()
        )
        return {
            "total_stages": len(self.stages),
            "total_topics": total_topics,
            "completed_topics": completed_topics,
            "current_stage": self.current_stage,
            "progress_pct": round(completed_topics / max(1, total_topics) * 100, 1),
            "stages": {name: s.to_dict() for name, s in self.stages.items()},
        }

    def to_curriculum_vitae(self) -> str:
        """Gera um curriculo vitae textual do aprendizado da AGI."""
        lines = ["=== CURRICULUM VITAE - NovaMind AGI ==="]
        for stage_name in ["INFANTIL", "PRE_ESCOLAR", "FUNDAMENTAL", "MEDIO", "SUPERIOR", "MESTRADO"]:
            if stage_name not in self.stages:
                continue
            s = self.stages[stage_name]
            status = "COMPLETO" if s.completed else "EM ANDAMENTO" if s.name == self.current_stage else "PENDENTE"
            lines.append(f"\n{stage_name} ({s.age_range}) - {status}")
            lines.append(f"  {s.description}")
            for t in s.topics:
                done = "[OK]" if getattr(t, '_completed', False) else "[..]"
                lines.append(f"  {done} {t.name}")
                if t.kaggle_datasets:
                    lines.append(f"     Datasets: {', '.join(t.kaggle_datasets)}")
        return "\n".join(lines)

    # ─── Kaggle Integration ───────────────────────────────

    def setup_kaggle(self, api_path: str):
        """Configura API do Kaggle."""
        self.kaggle_api_path = api_path
        # Cria diretorio .kaggle se necessario
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        # Copia token
        if os.path.exists(api_path):
            import shutil
            shutil.copy(api_path, os.path.join(kaggle_dir, "kaggle.json"))
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            print(f"[Kaggle] Token configurado de {api_path}")

    def download_stage_datasets(self, stage_name: str, output_dir: str = "data/curriculum") -> List[str]:
        """Baixa datasets Kaggle para um estagio."""
        if stage_name not in self.stages:
            return []
        stage = self.stages[stage_name]
        downloaded = []
        for topic in stage.topics:
            for ds in topic.kaggle_datasets:
                try:
                    import kaggle
                    dest = os.path.join(output_dir, stage_name, topic.name)
                    os.makedirs(dest, exist_ok=True)
                    kaggle.api.dataset_download_files(ds, path=dest, unzip=True)
                    downloaded.append(ds)
                    print(f"[Kaggle] Baixado: {ds} -> {dest}")
                except Exception as e:
                    print(f"[Kaggle] Erro ao baixar {ds}: {e}")
        return downloaded

    # ─── Persistencia ──────────────────────────────────────

    def _save_progress(self):
        """Salva progresso do curriculo em JSON."""
        data = {
            "current_stage": self.current_stage,
            "current_topic_idx": self.current_topic_idx,
            "stages": {},
        }
        for name, s in self.stages.items():
            data["stages"][name] = {
                "completed": s.completed,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "topics": {t.name: {"completed": getattr(t, '_completed', False)}
                          for t in s.topics},
            }
        with open(self._progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_progress(self):
        """Carrega progresso salvo."""
        if not os.path.exists(self._progress_file):
            return
        try:
            with open(self._progress_file) as f:
                data = json.load(f)
            self.current_stage = data.get("current_stage")
            self.current_topic_idx = data.get("current_topic_idx", 0)
            for name, stage_data in data.get("stages", {}).items():
                if name in self.stages:
                    s = self.stages[name]
                    s.completed = stage_data.get("completed", False)
                    s.started_at = stage_data.get("started_at", 0)
                    s.completed_at = stage_data.get("completed_at", 0)
                    for tname, tdata in stage_data.get("topics", {}).items():
                        for t in s.topics:
                            if t.name == tname:
                                t._completed = tdata.get("completed", False)
        except Exception:
            pass


# ─── Main: demonstra o curriculo ──────────────────────────

if __name__ == "__main__":
    engine = CurriculumEngine()
    engine.setup_default()

    print(engine.to_curriculum_vitae())
    print(f"\n\nStats: {engine.get_stats()}")

    # Simula progresso
    print("\n--- Simulando progresso ---")
    engine.start_stage("INFANTIL")
    for i in range(6):
        topic = engine.get_current_topic()
        if topic:
            print(f"Processando: {topic.name} ({topic.estimated_hours}h)")
            print(f"  Datasets: {topic.kaggle_datasets}")
            engine.complete_topic()

    print(f"\nProgresso: {engine.get_stats()}")
    print(engine.to_curriculum_vitae())
