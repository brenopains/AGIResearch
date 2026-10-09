"""Raw query: what the system knows about physics"""
import torch
data = torch.load(r'save/brain_live.pth', map_location='cpu', weights_only=False)
nodes = data['nodes']
edges = data['edges']
names = list(nodes.keys())
print('Total concepts:', len(names))
print('Total edges:', len(edges))

print()
print('GRAVIDADE:')
for e in edges:
    if not isinstance(e, dict): continue
    f = str(e.get('from',''))
    t = str(e.get('to',''))
    if f == 'gravity' or t == 'gravity':
        print(f'  {f} -> {t}  (type={e.get("edge_type")}, conf={e.get("confidence")})')

print()
print('FORCE:')
for e in edges:
    if not isinstance(e, dict): continue
    f = str(e.get('from',''))
    t = str(e.get('to',''))
    if f == 'force' or t == 'force':
        print(f'  {f} -> {t}  (type={e.get("edge_type")}, conf={e.get("confidence")})')

print()
print('MOTION:')
for e in edges:
    if not isinstance(e, dict): continue
    f = str(e.get('from',''))
    t = str(e.get('to',''))
    if f == 'motion' or t == 'motion':
        print(f'  {f} -> {t}  (type={e.get("edge_type")}, conf={e.get("confidence")})')

print()
print('EARTH:')
for e in edges:
    if not isinstance(e, dict): continue
    f = str(e.get('from',''))
    t = str(e.get('to',''))
    if f == 'earth' or t == 'earth':
        print(f'  {f} -> {t}  (type={e.get("edge_type")}, conf={e.get("confidence")})')

print()
print('ENERGY:')
for e in edges:
    if not isinstance(e, dict): continue
    f = str(e.get('from',''))
    t = str(e.get('to',''))
    if f == 'energy' or t == 'energy':
        print(f'  {f} -> {t}  (type={e.get("edge_type")}, conf={e.get("confidence")})')

print()
print('CADEIA COMPLETA gravitational -> ... -> gravity:')
current = 'gravitational'
for step in range(25):
    found = False
    for e in edges:
        if isinstance(e, dict) and str(e.get('from','')) == current:
            nxt = str(e.get('to',''))
            print(f'  [{step}] {current} -> {nxt}  type={e.get("edge_type")} conf={e.get("confidence")}')
            current = nxt
            found = True
            break
    if not found:
        break
