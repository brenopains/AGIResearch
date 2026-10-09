"""Test OpenIE extractor on physics sentences"""
import sys; sys.path.insert(0, '.')
from core.openie_extractor import OpenIEKnowledgeExtractor

ext = OpenIEKnowledgeExtractor()
test_sentences = [
    'Gravity causes objects to fall toward Earth.',
    'Friction prevents motion between surfaces.',
    'The Sun enables photosynthesis in plants.',
    'Acceleration follows force according to Newton.',
    'Water is a liquid at room temperature.',
    'Mitochondria produce energy for cells.',
    'Heat increases the temperature of materials.',
    'Genetic mutations cause evolutionary changes.',
]
for sent in test_sentences:
    triples = ext.extract_from_text(sent)
    for t in triples:
        subj = t['subject'][:30]
        rel = t['relation'][:12]
        obj = t['object'][:30]
        print(f'{subj:30s} -> {rel:12s} -> {obj:30s} (conf={t["confidence"]:.2f})')
    print()
print('Stats:', ext.get_stats())
