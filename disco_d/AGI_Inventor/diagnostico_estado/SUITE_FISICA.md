# Confirmação curta em câmeras e trajetórias novas

Após o diagnóstico exploratório na semente 53, congelamos `geral.py`,
`oraculo.py` e `percepcao_pixels.py`. Rodaremos sementes 97, 101, 107 e 113,
oito episódios cada: 0–5 para ajuste e 6–7 para avaliação com paleta B.
Cada mundo troca câmera, trajetórias, cruzamentos e oclusões. A representação
de pixels, o extrator de entidades, o rastreador e o estimador permanecem
idênticos. O oráculo só entra na auditoria, não no extrator.

Perguntas antes de ver os resultados:

1. O estado verdadeiro reduz o arrependimento dos pixels na maioria dos
   mundos? Reportar ganhos pareados por mundo, sem somar raízes dependentes.
2. O rastreador de pixels recupera ao menos metade desse ganho na maioria
   dos mundos e vence o embaralhamento de entidades?
3. Se o oráculo também falhar, o limite não pode ser atribuído apenas à
   percepção. Se o oráculo ganhar e o extrator não, atacar identidade e
   composição de objetos, não ampliar indiscriminadamente datasets.

Esta suíte continua restrita a um mundo sintético. Passar aqui não seria
evidência de mente humana; falhar informa qual elo não justificou expansão.
