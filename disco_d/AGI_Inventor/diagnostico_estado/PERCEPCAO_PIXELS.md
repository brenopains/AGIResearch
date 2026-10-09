# Diagnóstico exploratório: recuperar entidades dos pixels

O oráculo de posições e velocidades melhorou a decisão em órbitas; este ensaio
testa se um extrator simples, sem acesso ao estado do simulador, recupera parte
do ganho. O banco original já foi avaliado; o resultado nele é **exploratório**.

Entrada: quadros RGB do mesmo mundo, reproduzidos com semente 53. O código
de extração só recebe os quadros. Faz máscara de primeiro plano, componentes
conexos, centros de massa, ligação por menor deslocamento e diferenças entre
tempos. A representação admite até três componentes observáveis por quadro,
uma escolha específica desta bancada e não uma solução geral para ontologias.
Não usa cor para identificar objetos; a aparência muda em avaliação.

Compara arrependimento de escolha entre quatro sensores futuros: pixels 16×16,
entidades extraídas, oráculo de estado e 32 controles com entidades permutadas.
Os exemplos de avaliação não escolhem limiares ou atrasos. Um novo mundo com
outra semente seria necessário para confirmação após qualquer desenvolvimento.

Registrar também erro de localização contra estado oculto, **somente no
avaliador**, para distinguir falha de detecção da falha de uso das entidades.
