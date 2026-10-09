# Contrato para experimentos que realmente distinguem hipóteses

Uma consulta é uma intervenção identificada por `(estado, ação)` e um resultado
mensurável. Cada modelo deve fornecer **um intervalo de resultado**, sua origem
e se a previsão é exata ou estatística. Duas hipóteses formam testemunha apenas
se seus intervalos para a mesma consulta não se sobrepõem. O executor escolhe
a primeira testemunha por custo e ordem estável; não pontua "promessa". Após
a observação, mantém modelos compatíveis. Se nenhum sobreviver, sinaliza
`modelos_inadequados`; se não existir testemunha, `indistinguiveis_no_dominio`.

Para modelos estatísticos, não-sobreposição é uma separação **sob os intervalos
reportados**, não prova de que um dos modelos é verdadeiro. Calibrar raio apenas
em episódios de desenvolvimento. No Atari, os membros são os do comitê já
congelado, e o raio é o quantil 90% de erro fora do próprio episódio. O teste
é diagnóstico nos bancos já abertos; não aprender nem escolher usando os
resultados reservados.

Controle positivo exato: dois programas concordam em `x=1`, divergem em
`x=2`; observar o resultado elimina um. Controle de cobertura: um resultado
fora de todos os intervalos deve produzir diagnóstico de modelos inadequados,
nunca um vencedor inventado.
