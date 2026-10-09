# Estado como previsões de intervenções

Hipótese de desenvolvimento: imagens previstas são instáveis quando
realimentadas porque similaridade visual não garante que as mesmas ações
tenham as mesmas consequências. Definir provisoriamente o estado de um mundo
como a **assinatura de quatro intervenções curtas**: diferenças entre o
quadro atual e os quadros após oito passos de cada comando. Isto é uma
instância de estado preditivo, não uma invenção inédita por si só.

Usar os oito checkpoints de cada um dos oito episódios (64 estados por jogo
e semente). Em cada um, seguir cada comando
constante até 64 passos; a cada oito passos clonar o estado e aplicar os
quatro testes curtos. Guardar o par de quadros atual 21×16, as quatro
consequências visuais, recompensas e validade. Certificar as histórias de
raiz, os quatro testes iniciais e os estados do caminho contra os bancos
anteriores. Estados após terminal são inválidos e não treinam atualização.
Teto de 100 mil passos ALE por jogo e semente. A semente 103 já foi vista:
qualquer resultado nela é exploratório.

Aprender, somente em 97, um subespaço de 32 dimensões das assinaturas
observadas. Ajustar um operador por comando que atualize esse estado
preditivo em oito passos. O ponto inicial em 103 será produzido dos pixels
pela regressão local já confirmada; comparar também com ponto inicial
oracular. Avaliar a assinatura prevista após 8, 16, 32 e 64 passos contra
os testes reais nos estados alcançados, com ação cega, persistência e
previsão direta do horizonte como controles. A métrica mede contraste entre
intervenções, não reconstrução do fundo. Os resultados só justificariam
nova confirmação se o operador preservar contrastes em ambos os jogos e
superar os controles em episódios independentes.

Mesmo uma assinatura estável nesta bancada não seria uma mente artificial
geral: faltariam mudança de tarefa, descoberta de quais intervenções testar,
composição de novos operadores e transferência entre domínios.

**Errata de execução:** o coletor final guardou assinaturas visuais e
validade, mas não recompensas intermediárias. Portanto a confirmação em
`PRE_REGISTRO_PLANOS_109.md` mede conservação de consequências visuais;
nenhum ganho de recompensa ou decisão foi testado nessa rodada.
