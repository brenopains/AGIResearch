import random
import re
import time
from .adapters.browser_adapter import BrowserAdapter
from .adapters.wikipedia_adapter import WikipediaAdapter
from .adapters.youtube_adapter import YoutubeAdapter


class KnowledgeManager:
    """
    Gerenciador de Conhecimento Autônomo.
    
    Decide O QUE aprender e DE ONDE buscar.
    Prioridade:
      1. Curiosidades da mente (unanswered_questions)
      2. Lacunas no modelo de mundo (conceitos com poucas conexões)
      3. Exploração aleatória pura — qualquer conceito do mundo real
    """

    WHITELISTS = {
        "pt_text": ["Wikipédia PT", "Wikilivros PT", "SciELO Brasil", "Domínio Público"],
        "pt_video": ["Manual do Mundo", "Ciência Todo Dia", "Curso em Vídeo", "Khan Academy Brasil", "AuthenticGames"],
        "en_text": ["Wikipedia EN", "Wikibooks EN", "arXiv", "Project Gutenberg"],
        "en_video": ["MIT OpenCourseWare", "Khan Academy", "3Blue1Brown", "Veritasium", "Computerphile", "Mogswamp", "ilmango", "Xisumavoid"],
    }

    TOPIC_BANK = [
        "Água", "Fogo", "Gravidade", "Luz", "Som", "Calor", "Eletricidade",
        "Ar", "Terra", "Vento", "Número", "Adição", "Subtração",
        "Animal", "Planta", "Célula", "Coração", "DNA", "Cérebro",
        "Sol", "Lua", "Estrela", "Planeta",
        "Computador", "Internet", "Inteligência artificial",
        "Minecraft", "Redstone", "Bioma", "Vila", "Ferreiro",
        "Sobrevivência", "Construção", "Mineração", "Potion",
    ]

    def __init__(self):
        self.adapters = {
            "browser": BrowserAdapter(),
            "wikipedia": WikipediaAdapter(),
            "youtube": YoutubeAdapter(),
        }
        self.topics_studied = set()

        # Estado observável pela UI
        self.last_query = "Idle"
        self.last_source = "—"
        self.last_content_preview = ""
        self.status = "initializing"

    def pick_next_topic(self, curiosities: list = None, mind=None) -> str:
        """
        Decide o próximo tópico de estudo — curiosidade pura + progresso.
        
        Prioridade:
          1. Perguntas abertas da mente (curiosidade epistêmica)
          2. Se curiosidade por progresso está alta: aprofunda tópico recente
          3. Conceitos no modelo de mundo com poucas conexões (knowledge gaps)
          4. Se curiosidade baixa (tédio): explora algo completamente diferente
          5. Tópico aleatório
        """
        # 1. Curiosidades da mente têm prioridade absoluta
        if curiosities:
            for q in curiosities:
                topic = " ".join(q.get("keywords", [])) if q.get("keywords") else q.get("text", "")
                topic = topic.strip()
                if topic and len(topic) > 2:
                    return topic

        # 2. Curiosidade por progresso (H22b adaptado): se tá aprendendo, aprofunda
        curiosity_score = 0.0
        if mind is not None and hasattr(mind, 'archetypes'):
            curiosity_score = mind.archetypes.get_curiosity_score()
            if curiosity_score > 0.3 and self.topics_studied:
                # Tá achando padrões — continua cavando
                recent = random.choice(list(self.topics_studied))
                # Adiciona um termo de profundidade
                variations = [
                    f"{recent} história",
                    f"{recent} ciência",
                    f"{recent} Minecraft",
                    f"como funciona {recent}",
                ]
                return random.choice(variations)

        # 3. Lacunas no modelo de mundo: conceitos mal conectados
        if mind is not None and hasattr(mind, 'world_model'):
            wm = mind.world_model
            nodes = list(wm.nodes.values())
            if len(nodes) > 5:
                gaps = sorted(nodes, key=lambda n: len(wm.adjacency.get(n.concept_id, set())))
                if gaps and len(wm.adjacency.get(gaps[0].concept_id, set())) < 3:
                    concept_label = gaps[0].label
                    if concept_label and len(concept_label) > 2:
                        return concept_label

        # 4. Revisitar algo já estudado com nova perspectiva
        if self.topics_studied and random.random() < 0.3:
            return random.choice(list(self.topics_studied))

        # 5. Exploração pura — tópico aleatório
        return random.choice(self.TOPIC_BANK)

    def pick_source(self, topic: str, mind=None) -> tuple:
        """
        Escolhe a melhor fonte para um tópico.
        Alterna entre texto e vídeo para grounding rico.
        Returns: (adapter_key, source_name, category)
        """
        # 55% text, 45% video (more video for richer sensory experience)
        if random.random() < 0.55:
            # Text sources
            category = random.choice(["pt_text", "en_text"])
            source = random.choice(self.WHITELISTS[category])

            if "wikipedia" in source.lower() or "wikipédia" in source.lower():
                return "wikipedia", source, category
            else:
                return "browser", source, category
        else:
            # Video sources
            category = random.choice(["pt_video", "en_video"])
            source = random.choice(self.WHITELISTS[category])
            return "youtube", source, category

    def fetch_knowledge(self, topic: str, mind=None) -> dict:
        """
        Busca conhecimento real sobre um tópico.
        Escolhe a fonte, busca o conteúdo, retorna grounding textual.
        """
        adapter_key, source_name, category = self.pick_source(topic, mind)
        adapter = self.adapters[adapter_key]

        # Update observable state for UI
        self.last_query = topic
        self.last_source = f"{source_name} ({adapter_key})"
        self.status = f"fetching:{adapter_key}"

        print(f"[KNOWLEDGE] Buscando '{topic}' em {source_name} via {adapter_key}")

        try:
            result = adapter.query(topic, source_name)

            # Suporta retorno dict (YouTube enriquecido) ou string (adapters antigos)
            if isinstance(result, dict):
                content = result.get("content", "")
                extra = {k: v for k, v in result.items() if k != "content"}
            else:
                content = str(result)
                extra = {}

            self.topics_studied.add(topic)
            self.last_content_preview = content[:200] if content else ""
            self.status = "processing"

            return {
                "type": adapter_key,
                "content": content,
                "source": source_name,
                "topic": topic,
                "category": category,
                **extra,
            }

        except Exception as e:
            self.status = f"error:{e}"
            print(f"[KNOWLEDGE] Erro: {e}")
            return {
                "type": "error",
                "content": f"Erro ao buscar {topic}: {e}",
                "source": source_name,
                "topic": topic,
                "category": category,
            }

    @staticmethod
    def split_into_sentences(text: str, max_words: int = 30) -> list:
        """
        Quebra texto longo em sentenças individuais para alimentar mind.think().
        Cada sentença é um evento cognitivo separado.
        """
        if not text:
            return []

        # Split by sentence boundaries
        raw = re.split(r'(?<=[.!?])\s+', text)
        sentences = []
        for s in raw:
            s = s.strip()
            if not s or len(s) < 10:
                continue
            # Truncate very long sentences
            words = s.split()
            if len(words) > max_words:
                s = " ".join(words[:max_words])
            sentences.append(s)

        return sentences
