import pickle
import os

class CognitivePersistence:
    """
    Handles serializing and deserializing the state of the Cognitive OS.
    Allows turning the "brain" off and turning it back on exactly as it was.
    """
    def __init__(self, save_dir="brain_state"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_state(self, state_dict):
        filepath = os.path.join(self.save_dir, "cognitive_state.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        return filepath

    def load_state(self):
        filepath = os.path.join(self.save_dir, "cognitive_state.pkl")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
