import random
import time

class TextualGroundingEnv:
    """
    Simulates a closed textual world to feed the Cognitive OS.
    Outputs independent observations in real-time.
    """
    def __init__(self):
        self.scenarios = {
            "minecraft_logic": [
                "jogador bate na árvore",
                "bloco de madeira dropado",
                "jogador pega madeira",
                "jogador abre inventário",
                "madeira convertida em tábuas",
                "tábuas convertidas em crafting table",
                "jogador coloca crafting table no chão",
                "jogador usa crafting table",
                "tábuas convertidas em gravetos",
                "tábuas e gravetos convertidos em picareta de madeira",
                "jogador quebra pedra com picareta",
                "bloco de pedra dropado"
            ],
            "logic_syllogisms": [
                "gato é animal",
                "cachorro é animal",
                "animal é ser vivo",
                "gato mia",
                "cachorro late"
            ],
            "family_tree": [
                "João é pai de Maria",
                "Maria é mãe de Ana",
                "Pedro é irmão de João"
            ],
            "visual_concepts": [
                "maçã = fruta vermelha redonda",
                "banana = fruta amarela comprida",
                "fruta é comida"
            ]
        }
        self.current_scenario = None
        self.step_idx = 0

    def start_scenario(self, scenario_name):
        if scenario_name in self.scenarios:
            self.current_scenario = scenario_name
            self.step_idx = 0
            return f"Iniciando cenário: {scenario_name}"
        return "Cenário não encontrado."

    def step(self):
        if not self.current_scenario:
            # Pick a random scenario if none is active
            self.start_scenario(random.choice(list(self.scenarios.keys())))
            
        scenario_data = self.scenarios[self.current_scenario]
        if self.step_idx < len(scenario_data):
            obs = scenario_data[self.step_idx]
            self.step_idx += 1
            return {"type": "text", "content": obs, "timestamp": time.time(), "source": "env"}
        else:
            self.current_scenario = None
            return {"type": "event", "content": "Cenário concluído.", "timestamp": time.time(), "source": "env"}

if __name__ == "__main__":
    env = TextualGroundingEnv()
    for _ in range(15):
        print(env.step())
        time.sleep(0.5)
