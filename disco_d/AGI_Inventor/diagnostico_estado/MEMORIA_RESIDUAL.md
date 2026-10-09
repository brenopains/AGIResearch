# História visual e transições mistas: diagnóstico anterior à promoção

Registro antes da coleta e dos resultados. O objetivo humano não foi atingido.
Esta etapa testa informação temporal e aprendizado de transições; não substitui
o objetivo por desempenho em jogos. Treino97 e desenvolvimento103 são sementes
já utilizadas. Não chamar este ensaio de confirmação independente.

Coletar em cada jogo32raízes (primeiros quatro checkpoints de oito episódios),
duas sequências aleatórias de16blocos de8ações por raiz. As sequências são
geradas com RNG9791, iguais entre sementes/jogos, sem usar resultado. Guardar
os16quadros anteriores a cada estado, efeitos de quatro sondas de8ações,
recompensa e término de cada bloco. As sondas são intervenções reais, custeadas,
restauram estado e não entram na história passada da trajetória. Certificar
todas64raízes da caminhada contra o banco existente,128ramoscurtos nas
32raízes coletadas e replay da primeira sequência por episódio. Teto70.000
passos ALE por jogo/semente, até dois processos. Não executar se já existe.

Todos os braços principais treinam exatamente as mesmas transições mistas97.
O estado32 é a assinatura de consequências observada, usando a base antiga
congelada. É uma medida oracular por intervenções, não percepção autônoma.
A história visual só contém quadros anteriores à ação principal; nenhum
quadro de sonda contrafactual ou consequência posterior entra na entrada.

Gerar variáveis adicionais assim: ajustar estado→próximo estado/recompensa/
término em duas metades de episódios de97; predizer a metade que não treinou;
correlacionar a história com esses resíduos fora do ajuste; extrair até32
direções por SVD da correlação. A definição completa é projeção linear do
vetor de quadros, com média/matriz salvas, não um nome ou HDV. Trata-se de uma
construção estatística conhecida, sem alegação de novidade científica.

Braços de igual dimensão64, com o mesmo kernel e mesmas observações:

- estado32 e32zeros;
- estado32 e32variáveis de dois quadros, orientadas pelo resíduo;
- estado32 e32variáveis de16quadros, orientadas pelo resíduo (primário);
- estado32 e32variáveis de16quadros, com resíduos embaralhados dentro de ação;
- estado32 e32variáveis de16quadros com ordem passada embaralhada por exemplo
  (preservar os dois últimos quadros; cada exemplo tem permutação independente);
- estado32 e32variáveis de16quadros, extraídas sem alvo por PCA.

Cada bloco é normalizado por mediana de distâncias de até512estados de treino.
Modelos predizem atualização do estado64, recompensa e continuação. Na
composição, usam apenas seus próprios estados previstos e ações do programa;
recompensa é limitada ao intervalo observado e continuação a[0,1]. Término
absorve massa, nunca reinicia a trajetória como se fosse a mesma experiência.

Salvar todos os corpos/hashes antes de abrir103 no avaliador. O avaliador
mede um passo com estado real, composição a32/64/128passos, contraste de
recompensa e escolha offline entre as duas sequências já coletadas. A história
e assinatura inicial reais são oraculares: qualquer melhora é diagnóstico,
não um agente a ser promovido. Não consultar estados futuros durante rollout.

Portão diagnóstico fixo para o primário: menor MSE de diferenças de retorno
entre as duas sequências em128passos contra cada braço, limite inferior95%
do ganho pareado por episódio acima de zero (bootstrap10.000,RNG9797), pelo
menos dois episódios positivos; arrependimento da escolha estritamente menor
que cada braço, sem piorar nenhum episódio frente ao estado sozinho. Exigir
também menor erro de recompensa local que estado sozinho e resíduos embaralhados.
Se faltar diferença real entre sequências, inconclusivo. Não alterar o
primário, rank, normalização ou portão após resultados. Reportar separadamente
erro local, erro acumulado, detalhes da coleta e do ajuste.

Ganho da história completa além de dois quadros seria evidência de informação
temporal útil para o modelo testado. Nulo não prova suficiência do estado;
o gerador linear pode não extrair a informação. Ganho só local não promove
memória; ganho oracular de decisão ainda exigiria extrator sem sondas reais,
confirmação nova e transferência para outros mundos.
