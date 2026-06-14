import random

class CuriosityLayer:
    """
    Layer 5: Curiosity.
    Maintains active questions and tracks the agent's attention.
    """
    def __init__(self):
        self.active_questions = []
        self.attention_distribution = {
            "environment_processing": 100,
            "internal_restructuring": 0
        }

    def update_attention(self, environment_processing_pct, internal_restructuring_pct):
        self.attention_distribution = {
            "environment_processing": environment_processing_pct,
            "internal_restructuring": internal_restructuring_pct
        }

    def ask_question(self, question):
        self.active_questions.append(question)
        if len(self.active_questions) > 5:
            self.active_questions.pop(0)

    def get_state(self):
        return {
            "active_questions": self.active_questions,
            "attention": self.attention_distribution
        }
