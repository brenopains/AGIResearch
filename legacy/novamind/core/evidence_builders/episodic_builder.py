from typing import List, Optional
from ..cognition.contracts import EvidenceItem, EvidenceSource, EvidenceRelation, RecallDecision, RecallStatus
from ..cognition.recall_policy import RecallPolicy


class EpisodicBuilder:
    """EpisodeStore + RecallPolicy -> EvidenceItem(source=episodic).

    Nao sabe como o EpisodeStore funciona internamente.
    So chama query e passa resultado para o RecallPolicy.
    """

    def __init__(self, episode_store, recall_policy: Optional[RecallPolicy] = None):
        self.episode_store = episode_store
        self.recall_policy = recall_policy or RecallPolicy(threshold=0.25)

    def build(self, query: str, top_k: int = 5) -> List[EvidenceItem]:
        """Busca episodios similares via EpisodeStore.query_text e filtra com RecallPolicy.

        Returns lista de EvidenceItem (pode ser vazia se RecallPolicy rejeitar).
        """
        results = self.episode_store.query_text(query, top_k=min(top_k, 10))
        decision = self.recall_policy.evaluate(results, method="text")

        if decision.is_empty:
            return []

        return self._convert(decision)

    def build_hdv(self, query_hdv, top_k: int = 5) -> List[EvidenceItem]:
        """Busca por similaridade HDV."""
        results = self.episode_store.query_hdv(query_hdv, top_k=min(top_k, 10))
        decision = self.recall_policy.evaluate(results, method="hdv")

        if decision.is_empty:
            return []

        return self._convert(decision)

    def _convert(self, decision: RecallDecision) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        for ep in decision.episodes:
            content = ep.get("content", {})
            if isinstance(content, str):
                content = {"text": content[:200]}

            confidence = min(1.0, float(ep.get("score", 0)) * decision.best_similarity)
            relation = self._infer_relation(ep)

            items.append(EvidenceItem(
                source=EvidenceSource.EPISODIC,
                relation=relation,
                confidence=round(confidence, 3),
                raw_score=float(ep.get("score", 0)),
                trace=f"Episode[{ep.get('id', '?')}] sim={ep.get('score', 0):.3f}",
                metadata={
                    "episode_id": ep.get("id", ""),
                    "score": ep.get("score", 0),
                    "type": ep.get("type", ""),
                    "importance": ep.get("importance", 0),
                    "content_snippet": str(content)[:100],
                },
            ))

        return items

    def _infer_relation(self, episode: dict) -> EvidenceRelation:
        etype = str(episode.get("type", "")).lower()
        if "is_a" in etype or "category" in etype:
            return EvidenceRelation.IS_A
        if "property" in etype or "attribute" in etype:
            return EvidenceRelation.HAS_PROPERTY
        if "cause" in etype or "effect" in etype:
            return EvidenceRelation.CAUSES
        if "similar" in etype or "analogy" in etype:
            return EvidenceRelation.SIMILAR
        return EvidenceRelation.RELATED
