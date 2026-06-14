class WorldModel:
    """
    Layer 3: World Models.
    Abstracts knowledge from episodic memory into a persistent graph of concepts, entities, and causal links.
    Separates Observation ("I saw X") from Belief ("X exists").
    """
    def __init__(self):
        self.entities = {}   # e.g., "Entidade_42": {"type": "crafting_table", "properties": [...]}
        self.causal_links = [] # e.g., {"cause": "madeira", "effect": "tábuas"}
        self.beliefs = {}    # Strong convictions derived from multiple observations

    def compress_episodes(self, episodic_memory):
        """
        Reads uncompressed episodes and updates the world model.
        """
        uncompressed = episodic_memory.get_uncompressed_episodes()
        for ep in uncompressed:
            content = ep["event"]["content"]
            
            # Very simplistic natural language parsing to simulate concept extraction
            # In a real system, this would use embeddings / LLM / parsers
            
            if "convertida em" in content:
                parts = content.split(" convertida em ")
                if len(parts) == 2:
                    cause, effect = parts[0].strip(), parts[1].strip()
                    self.add_causal_link(cause, effect)
                    
            elif "convertidos em" in content:
                parts = content.split(" convertidos em ")
                if len(parts) == 2:
                    cause, effect = parts[0].strip(), parts[1].strip()
                    self.add_causal_link(cause, effect)
                    
            elif "=" in content:
                parts = content.split("=")
                if len(parts) == 2:
                    entity, definition = parts[0].strip(), parts[1].strip()
                    self.add_entity(entity, {"definition": definition})
                    
            elif " é " in content:
                parts = content.split(" é ")
                if len(parts) == 2:
                    subject, category = parts[0].strip(), parts[1].strip()
                    self.add_entity(subject, {"is_a": category})

            # Mark as compressed
            episodic_memory.mark_compressed(ep["episode_id"])

    def add_entity(self, name, properties):
        if name not in self.entities:
            self.entities[name] = {"name": name, "properties": {}, "appearances": 0}
        self.entities[name]["properties"].update(properties)
        self.entities[name]["appearances"] += 1
        
        # Form a belief if seen enough times
        if self.entities[name]["appearances"] >= 1:
            self.beliefs[name] = "exists"

    def add_causal_link(self, cause, effect):
        link = {"cause": cause, "effect": effect, "strength": 1}
        # Check if exists to increase strength
        existing = next((l for l in self.causal_links if l["cause"] == cause and l["effect"] == effect), None)
        if existing:
            existing["strength"] += 1
        else:
            self.causal_links.append(link)

    def get_graph(self):
        return {
            "entities": self.entities,
            "causal_links": self.causal_links,
            "beliefs": self.beliefs
        }
