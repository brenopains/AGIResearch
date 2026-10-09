"""Quick test for AGI server"""
import urllib.request, json, time
import subprocess, sys, os

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

proc = subprocess.Popen([sys.executable, 'agi_server.py', '8767'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)

try:
    resp = urllib.request.urlopen('http://127.0.0.1:8767/mock_data', timeout=3)
    data = json.loads(resp.read())
    print('Server OK!')
    print('Cycle:', data['cycle'])
    print('Mode:', data['maestro']['mode'])
    print('Concept:', data['lifestream']['current_concept'])
    print('CGE nodes:', data['cge']['n_nodes'])
    print('Forged operators:', data['operators']['forged'])
    print('Sim branches:', data['simulation']['n_branches'])
    print('Current media:', data['lifestream']['current_media']['type'])
    print('Emotion:', data['maestro']['emotion'])
except Exception as e:
    print('Error:', e)
finally:
    proc.kill()
