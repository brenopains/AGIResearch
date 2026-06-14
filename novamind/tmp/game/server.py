"""
server.py — servidor do jogo: roda o mundo em thread (velocidade controlavel) e serve o front.
Rode:  python tmp/game/server.py   -> abra http://127.0.0.1:5005
A LIGACAO: o MAESTRO da AGI (cortex_alife.Cortex) governa os NPCs-elite ao vivo (top-down, sem trapaca).
"""
import os, sys, time, threading
from flask import Flask, jsonify, request, send_from_directory
sys.path.insert(0,os.path.dirname(__file__))
from world import World
from cortex_alife import Cortex, A_NAMES

app=Flask(__name__); HERE=os.path.dirname(__file__)
W=World(seed=0)
CX=Cortex(k=60, every=40)                     # o maestro como cortex top-down
state={'speed':1.0,'run':True,'cortex':True}  # cortex ON por padrao (a ligacao ativa)
lock=threading.Lock()

def loop():
    import time as _t
    while True:
        sp=state['speed']
        if sp<=0 or not state['run']: _t.sleep(0.04); continue
        nb=1 if sp<=2 else int(sp/2)            # turbo = lote de passos por iteracao (sem travar)
        with lock:
            for _ in range(nb):
                if state['cortex'] and W.ticks % CX.every == 0: CX.govern(W)
                else:
                    if not state['cortex']: W.goal_bias.zero_(); W.elite_mask.zero_()  # desliga o vies
                W.step()
        _t.sleep(0.0 if sp>=30 else 1.0/sp)     # render desacoplado; turbo roda o mais rapido que a GPU deixa
threading.Thread(target=loop,daemon=True).start()

@app.route('/')
def idx(): return send_from_directory(HERE,'index.html')
@app.route('/state')
def st():
    with lock:
        snap=W.snapshot()
        cs=CX.state(); cs['on']=state['cortex']
        cs['goal_name']=A_NAMES.get(cs.get('goal'), '-') if cs.get('goal') is not None else '-'
        # ids dos elite governados (p/ destacar no front)
        em=W.elite_mask.nonzero().flatten().cpu().numpy().tolist() if state['cortex'] else []
        snap['cortex']=cs; snap['elite']=em[:200]
        return jsonify(snap)
@app.route('/control',methods=['POST'])
def ctl():
    d=request.get_json(force=True); a=d.get('action')
    with lock:
        if a=='speed': state['speed']=float(d['v'])
        elif a=='freq': W.set_frequency(float(d['v']))
        elif a=='place': W.place(d['x'],d['y'],d['kind'])
        elif a=='spawn_player': W.spawn_player()
        elif a=='player_move': W.set_player_action(int(d['v']))
        elif a=='cortex': state['cortex']=bool(d['v'])      # liga/desliga o maestro
        elif a=='reset':
            globals()['W']=World(seed=int(time.time())%1000)
            globals()['CX']=Cortex(k=60, every=40)
    return jsonify({'ok':True})

if __name__=='__main__':
    print("Jogo em http://127.0.0.1:5005  (Ctrl+C p/ parar) | MAESTRO governando elite (cortex ON)")
    app.run(host='127.0.0.1',port=5005,threaded=True)
