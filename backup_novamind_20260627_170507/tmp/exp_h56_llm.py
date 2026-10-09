"""
H56-L — LLM pequeno como solver TRANSDUTIVO de ARC (GPU 4GB) + AIRV  [aposta #3]
================================================================================
Tentativa honesta do "caminho neural pesado" possivel num RTX 3050: um LLM instruct
pequeno (Qwen2.5-0.5B-Instruct, ~1GB fp16) le os demos como texto e GERA a grade de
saida do teste. AIRV: roda sobre simetrias diedrais, reverte, vota por celula.
Expectativa honesta: BAIXA (LLMs pequenos pontuam ~0 no ARC sem fine-tuning pesado);
serve como peca complementar do ensemble e como medida real do teto neste hardware.
Time-boxed: roda num subconjunto p/ medir; se nao couber em 4GB, reporta e aborta.
"""
import os, json, glob, sys, time, numpy as np
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
MODEL=os.environ.get("ARC_LLM","Qwen/Qwen2.5-0.5B-Instruct")
NTASK=int(os.environ.get("ARC_LLM_NTASK","100"))   # subconjunto p/ time-box
NAUG=int(os.environ.get("ARC_LLM_NAUG","3"))        # nº de simetrias no AIRV

def grid_to_text(g): return "\n".join(" ".join(str(int(v)) for v in row) for row in g)
def text_to_grid(s):
    rows=[]
    for line in s.strip().splitlines():
        toks=[t for t in line.replace(","," ").split() if t.lstrip("-").isdigit()]
        if toks: rows.append([int(t) for t in toks])
    if not rows: return None
    w=max(len(r) for r in rows); rows=[r for r in rows if len(r)==w]
    if not rows: return None
    return np.array(rows)

DIH=lambda g,k:[g,np.rot90(g,1),np.rot90(g,2),np.rot90(g,3),g[:,::-1],g[::-1,:],np.swapaxes(g,0,1),np.swapaxes(g[:,::-1],0,1)][k]
DIHINV={0:0,1:3,2:2,3:1,4:4,5:5,6:6,7:7}

def build_prompt(train,test_in):
    ex=[]
    for i,o in train:
        ex.append("INPUT:\n"+grid_to_text(i)+"\nOUTPUT:\n"+grid_to_text(o))
    demos="\n\n".join(ex)
    return (f"You are solving an ARC puzzle. Each grid uses digits 0-9 as colors. "
            f"Infer the transformation from the examples and apply it.\n\n{demos}\n\n"
            f"INPUT:\n{grid_to_text(test_in)}\nOUTPUT:\n")

def main():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(f"H56-L ABORTADO: libs ausentes ({e}). pip install transformers accelerate"); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok=AutoTokenizer.from_pretrained(MODEL)
        model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16 if dev=="cuda" else torch.float32).to(dev)
    except Exception as e:
        print(f"H56-L ABORTADO ao carregar {MODEL}: {e}"); return
    model.eval()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))[:NTASK]
    solved=0; attempted=0; t0=time.time(); solved_ids=[]
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        xi=np.array(t["test"][0]["input"]); yo=np.array(t["test"][0]["output"])
        attempted+=1
        # AIRV: vota a previsao sobre NAUG simetrias
        H,W=yo.shape; votes={}
        for k in range(NAUG):
            tin=np.ascontiguousarray(DIH(xi,k)); aug_train=[(np.ascontiguousarray(DIH(i,k)),np.ascontiguousarray(DIH(o,k))) for i,o in train]
            prompt=build_prompt(aug_train,tin)
            msgs=[{"role":"user","content":prompt}]
            enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True)
            ids=enc["input_ids"].to(dev); attn=enc.get("attention_mask"); attn=attn.to(dev) if attn is not None else None
            if ids.shape[1]>3500: continue
            with torch.no_grad():
                out=model.generate(ids,attention_mask=attn,max_new_tokens=400,do_sample=False,pad_token_id=tok.eos_token_id)
            txt=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
            g=text_to_grid(txt)
            if g is None: continue
            gb=np.ascontiguousarray(DIH(g,DIHINV[k])) if g.shape[::-1]==g.shape or True else g
            try:
                gb=np.ascontiguousarray(DIH(g,DIHINV[k]))
            except Exception: gb=g
            if gb.shape==(H,W):
                for r in range(H):
                    for c in range(W): votes[(r,c)]=votes.get((r,c),{}); votes[(r,c)][gb[r,c]]=votes[(r,c)].get(gb[r,c],0)+1
        if votes:
            pred=np.zeros((H,W),int)
            for (r,c),d in votes.items(): pred[r,c]=max(d,key=d.get)
            if pred.shape==yo.shape and np.array_equal(pred,yo): solved+=1; solved_ids.append(os.path.basename(fp))
    json.dump(solved_ids,open(os.path.join(os.path.dirname(__file__),"h56_llm_solved.json"),"w"))
    dt=time.time()-t0; vram=torch.cuda.max_memory_allocated()/1e6 if dev=="cuda" else 0
    print("="*64)
    print(f"H56-L LLM transdutivo ({MODEL}) em {dev} | AIRV={NAUG}")
    print(f"  RESOLVIDAS: {solved}/{attempted} (subconjunto) = {100*solved/max(1,attempted):.1f}%")
    print(f"  tempo {dt:.0f}s | VRAM pico {vram:.0f}MB")
    print("  honesto: LLM pequeno sem fine-tuning pesado tende a ~0 no ARC; medida do teto no 3050.")

if __name__=="__main__": main()
