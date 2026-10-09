# Confirmação lacrada do rastreador com hipóteses

Os mundos 53, 97, 101, 107 e 113 foram usados para desenvolver o código. O
último ajuste antes da confirmação substituiu uma cor exata do oclusor por um
teste genérico de região acromática visível. Congelar agora
`hipoteses_estado.py`, `avaliar_hipoteses.py`, `percepcao_pixels.py` e o estimador.

Seis sementes novas: 127, 131, 137, 139, 149 e 151. Oito episódios por semente,
com 0–5 de ajuste e 6–7 de avaliação com aparência alterada. Mesmos 16 filtros
alternativos; nenhuma seleção por resultado reservado. Comparações emparelhadas
no **mundo** (não tratar 16 raízes correlatas como 16 réplicas): pixels, rastreador
guloso, hipótese de menor custo e média das hipóteses. Arrependimento de decisão
normalizado é o alvo primário. Contar também 32 permutações por mundo.

Gate exploratório pré-estabelecido: uma das duas formas de hipóteses precisa
vencer pixels e o guloso em pelo menos 5/6 mundos, com no máximo 1/32 nulos
iguais ou melhores em pelo menos 4/6. Reportar custos computacionais. Passar
esse gate mostraria uma melhora local de percepção/decisão em física visual,
não inteligência geral; falhar redireciona para o acoplamento entre estado,
dinâmica e valor, em vez de aumentar a largura do feixe.
