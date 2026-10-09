# Ler a experiência em função da pergunta executável

Pré-registro anterior ao ajuste desta intervenção. Meta humana não atingida.
Treino97 e banco de programas novos103/RNG9811 já utilizados: desenvolvimento,
nunca confirmação independente. Não alterar corpos/vereditos anteriores.

Intervenção: a leitura de pixels será uma função construída para o programa
consultado, com parâmetros completos e execução direta. Entrada somente os
16quadros passados21×16 e palavras de ações; não usar assinatura de sondas,
estado oculto, nomes de entidades ou variáveis do jogo. As respostas de
treino são retornos reais das duas trajetórias pareadas e seus16prefixos.
512pares correlacionados por32raízes,1.024retornos; mesmas observações para
todos os braços novos, sem nova coleta física.

Corpo principal: centrar história pela média de treino; normalizar por raiz
da média de norma quadrática. Cada template é uma história observada. Ler
projeções sobre templates, acrescentar seus quadrados, e combiná-los com
coeficientes construídos da palavra consultada. Kernel sensorial1+u+u²,
u=produto interno das histórias normalizadas; kernel de palavra RBF da
rodada anterior. Ajustar diferenças de retornos com matriz espelhada1.024²
e regularização0,05. Não haverá escolha de grau/escala após resultados.

A leitura exportada para cada programa contém média, escala, termo linear,
viés, templates e coeficientes dos termos quadráticos. Comparação é diferença
entre duas leituras executadas na mesma história. Identidade, troca de sinal
e aditividade são invariantes de implementação, não provas de acerto futuro.
O alfabeto/horizonte são parâmetros; nesta bancada4ações/16blocos.

Importante: condicionar uma combinação linear ao programa não inventa um
espaço ilimitado de representações. O corpo quadrático também é equivalente
a selecionar/compor características fixas dos templates e seus quadrados.
Essa equivalência será testada, não escondida sob o nome "atenção". O gerador
cria funções parametrizadas dentro dessa linguagem definida por nós, não
conceitos humanos ou uma técnica científica nova.

Braços iguais em observações/tamanho de sistema:

- leitura condicionada quadrática16quadros (primário);
- leitura condicionada linear16quadros;
- leitura condicionada quadrática de apenas dois quadros;
- leitura quadrática com diferenças de retorno embaralhadas por comprimento
  e identidade das palavras;
- leitura quadrática com histórias de treino embaralhadas entre raízes,
  mantendo todos os prefixos da raiz juntos;
- somente palavras, nenhuma dependência de pixels;
- RBF sobre a história crua, regressão contrastiva congelada da rodada anterior,
  reajustada apenas97, sem intervenções como entrada.

Comparar também com todos os12braços já medidos no banco de palavras novas.
Os braços antigos que recebem assinatura medida são referências oraculares,
identificadas como tais; não fornecem essa assinatura ao primário. Salvar
corpos/leituras/hashes antes de avaliar. Em103, executar as leituras com a
história inicial; respostas reservadas só calculam erro/arrependimento.

Portão128passos: ganho médio de arrependimento, IC95% inferior acima de zero
e pelo menos dois episódios positivos contra cada braço novo/antigo; nenhuma
piora por episódio frente ao índice de programa constante. MSE de diferenças
menor que versão linear e alvos embaralhados. Bootstrap8episódios/10.000,
RNG9827. Se não existir contraste real, inconclusivo.32/64passos secundários.
Sem alterar primário/portão depois, sem promoção de braço de comparação.

Se falhar, extrair diagnósticos de limites dessa família: memória temporal,
dependência de pixels versus marginais das palavras, e equivalência com
espaço de características fixo. Não anunciar que uma peça única da AGI foi
isolada. Positivo exigiria semente/mundos novos e transferência de um corpo
aprendido antes de afirmar capacidade geral ou invenção de abstrações humanas.
