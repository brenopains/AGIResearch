"""
DSL SEMENTE — operacoes de grid GERAIS por construcao (os 'instintos motores' da AGI).
O motor compoe estas (nao-comutativo) p/ resolver SEM LLM desde o inicio. O LLM so adiciona
NOVAS operacoes gerais no fracasso. Cada uma e generica (auto-configura a partir do grid),
nao decorada p/ uma task. Cada entrada: (categoria, codigo-fonte de transform(g)).
"""
SEED = [
("rotate_90",        "def transform(g):\n import numpy as np\n return np.rot90(np.array(g),1).tolist()"),
("rotate_180",       "def transform(g):\n import numpy as np\n return np.rot90(np.array(g),2).tolist()"),
("rotate_270",       "def transform(g):\n import numpy as np\n return np.rot90(np.array(g),3).tolist()"),
("flip_h",           "def transform(g):\n return [row[::-1] for row in g]"),
("flip_v",           "def transform(g):\n return g[::-1]"),
("transpose",        "def transform(g):\n import numpy as np\n return np.array(g).T.tolist()"),
("anti_transpose",   "def transform(g):\n import numpy as np\n return np.rot90(np.array(g).T,2).tolist()"),
("tile_2x2",         "def transform(g):\n import numpy as np\n return np.tile(np.array(g),(2,2)).tolist()"),
("tile_1x2",         "def transform(g):\n import numpy as np\n return np.tile(np.array(g),(1,2)).tolist()"),
("tile_2x1",         "def transform(g):\n import numpy as np\n return np.tile(np.array(g),(2,1)).tolist()"),
("mirror_right",     "def transform(g):\n return [row+row[::-1] for row in g]"),
("mirror_down",      "def transform(g):\n return g+g[::-1]"),
("scale_2x",         "def transform(g):\n import numpy as np\n return np.kron(np.array(g),np.ones((2,2),int)).tolist()"),
("scale_3x",         "def transform(g):\n import numpy as np\n return np.kron(np.array(g),np.ones((3,3),int)).tolist()"),
# recolor: mapeia a cor mais frequente (nao-zero) <-> menos frequente — auto-configura
("recolor_swap_freq","def transform(g):\n import numpy as np\n from collections import Counter\n a=np.array(g); c=Counter(a.flatten().tolist())\n c.pop(0,None)\n if len(c)<2: return g\n mc=max(c,key=c.get); lc=min(c,key=c.get)\n b=a.copy(); b[a==mc]=lc; b[a==lc]=mc; return b.tolist()"),
# gravidade: empurra nao-zeros p/ baixo em cada coluna
("gravity_down",     "def transform(g):\n import numpy as np\n a=np.array(g); h,w=a.shape; b=np.zeros_like(a)\n for j in range(w):\n  col=[v for v in a[:,j] if v!=0]; b[h-len(col):,j]=col\n return b.tolist()"),
# crop ao bounding-box do conteudo nao-zero
("crop_content",     "def transform(g):\n import numpy as np\n a=np.array(g); nz=np.argwhere(a!=0)\n if len(nz)==0: return g\n (y0,x0),(y1,x1)=nz.min(0),nz.max(0)+1; return a[y0:y1,x0:x1].tolist()"),
# borda/moldura: zera o interior, mantem so a borda
("keep_border",      "def transform(g):\n import numpy as np\n a=np.array(g).copy()\n if a.shape[0]>2 and a.shape[1]>2: a[1:-1,1:-1]=0\n return a.tolist()"),
# identidade (util em composicao / casos triviais)
("identity",         "def transform(g):\n return [list(r) for r in g]"),
# ---- VERBOS COMPONIVEIS EXTRA (object-level + direcionais) p/ enriquecer composicao ----
("gravity_up",       "def transform(g):\n import numpy as np\n a=np.array(g);h,w=a.shape;b=np.zeros_like(a)\n for j in range(w):\n  col=[v for v in a[:,j] if v!=0];b[:len(col),j]=col\n return b.tolist()"),
("gravity_left",     "def transform(g):\n import numpy as np\n a=np.array(g);h,w=a.shape;b=np.zeros_like(a)\n for i in range(h):\n  row=[v for v in a[i,:] if v!=0];b[i,:len(row)]=row\n return b.tolist()"),
("gravity_right",    "def transform(g):\n import numpy as np\n a=np.array(g);h,w=a.shape;b=np.zeros_like(a)\n for i in range(h):\n  row=[v for v in a[i,:] if v!=0];b[i,w-len(row):]=row\n return b.tolist()"),
# mantem so a cor mais frequente (nao-zero); zera o resto
("keep_most_color",  "def transform(g):\n import numpy as np\n from collections import Counter\n a=np.array(g);c=Counter(a.flatten().tolist());c.pop(0,None)\n if not c: return g\n mc=max(c,key=c.get);b=a.copy();b[a!=mc]=0;return b.tolist()"),
# remove a cor mais frequente (denoise)
("remove_most_color","def transform(g):\n import numpy as np\n from collections import Counter\n a=np.array(g);c=Counter(a.flatten().tolist())\n if not c: return g\n mc=max(c,key=c.get);b=a.copy();b[a==mc]=0;return b.tolist()"),
# binariza: tudo nao-zero vira a cor dominante
("binarize_dominant","def transform(g):\n import numpy as np\n from collections import Counter\n a=np.array(g);c=Counter(a.flatten().tolist());c.pop(0,None)\n if not c: return g\n mc=max(c,key=c.get);b=a.copy();b[a!=0]=mc;return b.tolist()"),
# contorno: borda dos objetos nao-zero (interior zerado se cercado)
("outline_objects",  "def transform(g):\n import numpy as np\n a=np.array(g);h,w=a.shape;b=a.copy()\n for i in range(h):\n  for j in range(w):\n   if a[i,j]!=0:\n    nb=[a[i+dy,j+dx] for dy,dx in((1,0),(-1,0),(0,1),(0,-1)) if 0<=i+dy<h and 0<=j+dx<w]\n    if nb and all(v!=0 for v in nb): b[i,j]=0\n return b.tolist()"),
# recorta a metade superior / esquerda (uteis em simetria)
("top_half",         "def transform(g):\n return g[:max(1,len(g)//2)]"),
("left_half",        "def transform(g):\n return [r[:max(1,len(r)//2)] for r in g]"),
# troca 0<->cor dominante (inverte fundo/figura)
("invert_bg",        "def transform(g):\n import numpy as np\n from collections import Counter\n a=np.array(g);c=Counter(a.flatten().tolist());c.pop(0,None)\n if not c: return g\n mc=max(c,key=c.get);b=a.copy();b[a==0]=mc;b[a==mc]=0;return b.tolist()"),
# diagonal principal espelhada (anti-transpose ja existe; este e flip diagonal sec.)
("rot90_flip",       "def transform(g):\n import numpy as np\n return np.fliplr(np.rot90(np.array(g),1)).tolist()"),
]
