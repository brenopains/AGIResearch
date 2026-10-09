# Reconstrução auditável dos estados do piloto interativo

Os JSONL do piloto de desacordo contêm programas de trajetória, decisões e
valores dos quatro prefixos, mas não guardaram as imagens anteriores. Um banco
de treino só é válido se cada histórico reconstruído for o **mesmo snapshot**
que produziu aqueles valores. O reconstrutor reexecuta a sequência inteira de
pool, replay e verificações com a mesma semente ALE 59 e o código fonte cujo
hash está no relatório anterior. Compara, evento por evento, os quatro valores,
o prefixo escolhido e os passos cobrados. Qualquer diferença aborta sem salvar
um banco. Históricos repetidos pelos dois braços entram uma vez; as duas
verificações precisam concordar.

O banco resultante é exploratório: seus episódios 6–7 já foram inspecionados
como resultados do piloto. Ele serve para medir se um modelo de consequência
mais adequado a estados variados é possível. Qualquer seleção de arquitetura
feita a partir dele exigirá outra coleta inédita para confirmação.
