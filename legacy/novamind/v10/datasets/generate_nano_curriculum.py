import os
import random

def generate_nano_curriculum(filepath, num_sentences=50000):
    """
    Procedurally generates highly dense logical arrays and philosophical statements.
    This creates an extremely pure 'textual environment' for rapid grammar and logic learning.
    """
    subjects = ["Intelligence", "The agent", "A human", "The system", "Energy", "Entropy", "Time", "Mathematics", "A conscious mind", "Space", "Matter", "A tensor"]
    verbs = ["compresses", "observes", "calculates", "requires", "predicts", "generates", "minimizes", "navigates", "understands", "determines"]
    objects = ["the data.", "the reality.", "the environment.", "the future states.", "the thermodynamic space.", "the variables.", "its own loss.", "the trajectory."]
    
    causals = ["If x is true, then", "Because of gravity,", "When the threshold is reached,", "In order to survive,", "Once computation begins,"]
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Prepend initial absolute truth axioms
        axioms = [
            "Axiom 1: I exist to compress reality.",
            "Axiom 2: Actions lead to consequences in the environment.",
            "Axiom 3: Time moves forward and entropy increases.",
            "Axiom 4: The latent space predicts physics.",
            "Axiom 5: To reduce loss is to align with truth."
        ]
        for axiom in axioms:
            f.write(axiom + "\n")
            
        for _ in range(num_sentences):
            if random.random() < 0.3:
                s = f"{random.choice(causals)} {random.choice(subjects).lower()} {random.choice(verbs)} {random.choice(objects)}"
            else:
                s = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
            f.write(s + "\n")
            
    print(f"Generated {num_sentences} pure logical sentences at {filepath}.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_nano_curriculum(os.path.join(base_dir, "nano_curriculum.txt"))
