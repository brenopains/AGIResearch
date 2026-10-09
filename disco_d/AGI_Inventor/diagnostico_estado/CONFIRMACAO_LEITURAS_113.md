# Confirmação congelada de leituras variáveis em Pong

A semente 113 não possui banco na inspeção antes deste registro. Ela não
participou da geração de programas, fragmentos, treinamento ou seleção.
O piloto em 103 escolheu oito sondas variáveis, quatro sondas aleatórias e
uma sonda de associação embaralhada. A vantagem sobre a ação constante foi
apenas 1/64 de unidade de retorno; a confirmação não pode assumir que seja real.

Congelar fontes, bancos de treino, programas e seleções com
`py -3.11 -B confirmar_leituras_variaveis.py --freeze`. O manifesto registra
SHA-256 de toda fonte local usada e dos artefatos de treinamento/seleção.
Não alterar esses arquivos nem reajustar na semente 113.

O teste tem oito episódios, oito raízes por episódio, quatro programas de ação
constante de 128 passos. Reconstruir com replay e conferir os 256 ramos curtos
contra uma coleta independente. O modelo só vê dois quadros iniciais; as
respostas dos ramos não orientam seleção, ajuste ou escolha de ação.

Medida primária: arrependimento de retorno, média em 64 raízes. Aceitar somente
se o braço proposto tiver vantagem média e limite inferior bootstrap de 95%
acima de zero contra **cada** controle: base preditiva, melhor ação fixa
escolhida em 97, fragmentos aleatórios selecionados em 103 e associação
embaralhada selecionada em 103. Reamostrar episódios inteiros, 10.000 vezes,
RNG 9743. Exigir também melhora em pelo menos dois episódios e nenhuma piora
por episódio contra a base. Episódios já perfeitos podem empatar.

Se não houver contraste de retorno ou margem para melhorar a ação fixa,
registrar inconclusivo. Nos demais casos, falhar em qualquer condição é falha
da confirmação pré-registrada, sem reajustar critério. O teste não prova que
o gerador seja incapaz em geral. Mesmo um passe provaria somente uma melhora
local de escolha entre quatro programas neste jogo, sem estabelecer agência
geral, descoberta científica, transferência entre mundos ou inteligência humana.

```powershell
py -3.11 -B confirmar_leituras_variaveis.py --freeze
py -3.11 -B confirmar_leituras_variaveis.py
```

O script recusa sobrescrever uma confirmação terminada. As coletas ficam em
`resultados/pong_efeitos_locais_113.*` e `pong_horizontes_113.*`; veredito,
diferenças pareadas, escolhas, retornos reais, custos e hashes em
`resultados/pong_confirmacao_leituras_113.json`.
