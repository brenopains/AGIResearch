class RepresentationLibrary:
    """
    Layer 4: Representation Library.
    Maintains the Evolutionary Tree of Knowledge (DAG of representations R1, R2, ...).
    """
    def __init__(self):
        self.library = {}
        # Seed with initial primitive representations
        self.add_representation("R0", "Identity", [])

    def add_representation(self, rep_id, definition, parents):
        """
        Registers a new representation and its dependencies to build the DAG.
        """
        self.library[rep_id] = {
            "id": rep_id,
            "definition": definition,
            "parents": parents
        }

    def get_dag(self):
        """
        Returns the Directed Acyclic Graph of representations for the 3D visualizer.
        """
        return self.library
