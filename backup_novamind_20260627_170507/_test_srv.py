"""Teste rapido do servidor AGI"""
import sys, os, time, subprocess, urllib.request, json
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

p = subprocess.Popen(
    [sys.executable, 'agi_server.py', '8767'],
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
)
time.sleep(4)

try:
    resp = urllib.request.urlopen('http://127.0.0.1:8767/data', timeout=5)
    data = json.loads(resp.read())
    print('LIVE! running:', data.get('running'))
    print('thoughts:', len(data.get('thoughts', [])))
    if data.get('thoughts'):
        print('last:', data['thoughts'][-1]['text'][:80])
except Exception as e:
    print('FAIL:', e)
    _, err = p.communicate(timeout=2)
    if err:
        print('STDERR:', err.decode('utf-8', errors='replace')[-300:])

p.kill()
