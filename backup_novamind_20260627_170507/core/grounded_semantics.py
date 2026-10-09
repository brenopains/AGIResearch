import re
from typing import Dict, List, Optional, Any

class CulturalRuleCompiler:
    """
    Grounded Semantic Compiler (Cultural Rule Compiler)
    
    Converts human language rules and experiences into executable cognitive primitives
    (SE, QUANDO, PORQUE, PARA, CAUSA, POSSIVEL).
    These rules are stored in semantic memory and can be executed by the world model
    or reasoning engine.
    """
    
    def __init__(self):
        # Patterns for extracting cognitive rules
        self.PATTERNS = [
            # "se [condição], [então] [consequência]"
            (r'(?i)^(?:se|quando)\s+(.+?)(?:,| então| talvez| pode)\s+(.+)$', self._parse_conditional),
            # "[sujeito] pode causar [objeto]"
            (r'(?i)^(.+?)\s+pode\s+causar\s+(.+)$', self._parse_causal_possibility),
            # "[sujeito] causa [objeto]"
            (r'(?i)^(.+?)\s+causa\s+(.+)$', self._parse_causal),
            # "[sujeito] serve para [objeto]"
            (r'(?i)^(.+?)\s+(?:serve|usado)\s+para\s+(.+)$', self._parse_purpose),
        ]

    def compile_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses a natural language string and attempts to compile it into a cognitive rule.
        Returns a list of structured rules.
        """
        text = text.strip().strip('.')
        rules = []
        
        for pattern, handler in self.PATTERNS:
            match = re.search(pattern, text)
            if match:
                rule = handler(match.groups(), text)
                if rule:
                    rules.append(rule)
        
        return rules

    def _parse_conditional(self, groups: tuple, source_text: str) -> Dict[str, Any]:
        cond_text, cons_text = groups
        
        # Determine modality of consequence
        modality = "CERTO"
        if "talvez" in cons_text.lower() or "pode" in cons_text.lower() or "possivel" in cons_text.lower():
            modality = "POSSIVEL"
            cons_text = re.sub(r'(?i)(talvez|pode|possivelmente|possivel)\s+', '', cons_text).strip()
            
        return {
            "type": "cognitive_rule",
            "primitive": "SE",
            "condition": self._extract_state(cond_text),
            "consequent": {
                "primitive": modality,
                "state": self._extract_state(cons_text)
            },
            "display": f"SE {cond_text} ENTÃO {modality}({cons_text})",
            "source_text": source_text
        }

    def _parse_causal_possibility(self, groups: tuple, source_text: str) -> Dict[str, Any]:
        cause, effect = groups
        return {
            "type": "cognitive_rule",
            "primitive": "SE",
            "condition": self._extract_state(cause),
            "consequent": {
                "primitive": "POSSIVEL",
                "state": {"primitive": "CAUSA", "effect": self._extract_state(effect)}
            },
            "display": f"SE {cause} ENTÃO POSSIVEL(CAUSA({effect}))",
            "source_text": source_text
        }
        
    def _parse_causal(self, groups: tuple, source_text: str) -> Dict[str, Any]:
        cause, effect = groups
        return {
            "type": "cognitive_rule",
            "primitive": "SE",
            "condition": self._extract_state(cause),
            "consequent": {
                "primitive": "CERTO",
                "state": {"primitive": "CAUSA", "effect": self._extract_state(effect)}
            },
            "display": f"SE {cause} ENTÃO CAUSA({effect})",
            "source_text": source_text
        }

    def _parse_purpose(self, groups: tuple, source_text: str) -> Dict[str, Any]:
        obj, purpose = groups
        return {
            "type": "cognitive_rule",
            "primitive": "PARA",
            "object": self._extract_state(obj),
            "purpose": self._extract_state(purpose),
            "display": f"OBJETO({obj}) SERVE_PARA({purpose})",
            "source_text": source_text
        }

    def _extract_state(self, text: str) -> Dict[str, Any]:
        """
        Converts a phrase into a minimal state representation.
        Ex: "água contaminada" -> {"subject": "água", "property": "contaminada"}
        """
        text = text.strip()
        words = text.split()
        
        # Simple extraction for adjective-noun pairs in Portuguese
        if len(words) == 2:
            return {"subject": words[0], "property": words[1], "raw": text}
            
        return {"subject": text, "raw": text}

