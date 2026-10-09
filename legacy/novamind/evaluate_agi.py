import os
import sys

# Redirect caches to D: drive
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

from core.mind import NovaMind
from brain_persistence import load_brain

def main():
    print("====================================================")
    # AGI QA Evaluation
    print("       NOVAMIND AGI ACCURACY EVALUATION             ")
    print("====================================================\n")

    # Load brain state in native, low-memory HDC mode
    print("[EVAL] Initializing NovaMind in native mode...")
    mind = NovaMind(use_semantic_recall=False)
    load_brain(mind)

    # Define test questions with expected target keywords
    test_cases = [
        {
            "question": "Who is Bentinho?",
            "expected_keywords": ["bentinho", "character", "dom", "casmurro", "machado"]
        },
        {
            "question": "Who is Capitu?",
            "expected_keywords": ["capitu", "eyes", "bentinho", "character", "dom", "casmurro"]
        },
        {
            "question": "Who is Ezequiel?",
            "expected_keywords": ["ezequiel", "son", "capitu", "bentinho", "character"]
        },
        {
            "question": "Who is Sherlock?",
            "expected_keywords": ["sherlock", "holmes", "detective", "watson"]
        },
        {
            "question": "Who is Watson?",
            "expected_keywords": ["watson", "doctor", "sherlock", "friend", "companion"]
        },
        {
            "question": "Who is Frankenstein?",
            "expected_keywords": ["frankenstein", "monster", "creator", "doctor"]
        },
        {
            "question": "Where does Sherlock live?",
            "expected_keywords": ["london", "baker", "street"]
        },
        {
            "question": "Where is Rio?",
            "expected_keywords": ["city", "rio", "brazil"]
        }
    ]

    print(f"\n[EVAL] Running {len(test_cases)} target QA checks...\n")
    
    passed = 0
    for idx, tc in enumerate(test_cases):
        q = tc["question"]
        expected = tc["expected_keywords"]
        
        print(f"[{idx+1}/{len(test_cases)}] Question: '{q}'")
        try:
            thought = mind.think(q)
            response_text = thought.get('response', {}).get('text', '')
            print(f"  Response: '{response_text}'")
            
            # Check case-insensitive keyword match
            resp_lower = response_text.lower()
            found = [kw for kw in expected if kw in resp_lower]
            
            if found:
                print(f"  Result:  SUCCESS (Found keywords: {found})")
                passed += 1
            else:
                print(f"  Result:  FAILED (Expected keywords: {expected})")
        except Exception as e:
            print(f"  Result:  ERROR: {e}")
        print("-" * 50)

    accuracy = (passed / len(test_cases)) * 100
    print("\n====================================================")
    print("               EVALUATION SUMMARY                   ")
    print("====================================================")
    print(f"Total Questions Evaluated : {len(test_cases)}")
    print(f"Passed Checks             : {passed}")
    print(f"Accuracy Rate             : {accuracy:.2f}%")
    
    # Print system size
    stats = mind.memory.get_stats()
    print(f"Brain State Size          : Concepts: {stats.get('total_concepts', 0)}, Episodic: {stats.get('episodic_memories', 0)}, Semantic: {stats.get('semantic_rules', 0)}")
    print("====================================================")

if __name__ == "__main__":
    main()
