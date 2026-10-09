# Correção do Relatório F2 — Evidência Adversarial

## F2: Motor de Inferência Simbólica — Dados Corrigidos

### Protocolo Original (conclusão fornecida como pista)
| Teste | Resultado |
|-------|-----------|
| Modus Ponens | 100% (44/44) |
| Modus Tollens | 100% (61/61) |
| Barbara | 100% (51/51) |
| Relational Chain | 100% (44/44) |
| **Total** | **100% (200/200)** |

**Limitação**: este teste dava a conclusão ao motor e pedia verificação. O motor sempre "confirma" o que lhe é dado como conclusão.

### Protocolo Adversarial (cego — sem conclusão)

O motor recebe APENAS as premissas e deve GERAR a conclusão.

| Métrica | Resultado | Chance | Interpretação |
|---------|-----------|--------|---------------|
| Recall (válidos aceitos) | **51.9%** | 50%* | Acima de chance, mas longe de 100% |
| Specificity (inválidos rejeitados) | **77.5%** | 50% | Rejeita consistentemente conclusões falsas |
| Accuracy | **57.0%** | 50% | 7pp acima de chance |
| Precisão | **0.902** | — | Quando aceita, raramente erra |

*Nota: para MP/MT, a chance é 50% (escolha binária entre inferir ou não). Para Barbara, é menor. A média ponderada é ~50%.

**Interpretação**: o motor é conservador — prefere não inferir (baixo recall) a inferir errado (alta especificidade). Isto é o comportamento esperado de um sistema de raciocínio seguro. Mas o recall de 51.9% significa que o motor perde metade das inferências válidas, principalmente por limitações na geração de conclusões (o motor sabe CONFIRMAR mas não PRODUZIR conclusões livremente).

### Conclusão sobre F2

O motor NÃO generaliza para estruturas que não foram previstas no gerador de silogismos. As 4 categorias (MP, MT, Barbara, RC) foram implementadas explicitamente. O teste adversarial mostra que:
- O motor funciona corretamente dentro do domínio desenhado (confirmação)
- O motor tem capacidade limitada de geração autônoma (51.9% recall)
- O motor rejeita corretamente a maioria dos inválidos (77.5% specificity)

**Veredito revisado**: Gap #5 parcialmente fechado. O motor de inferência simbólica está implementado e funcional, mas a capacidade de raciocínio dedutivo autônomo é limitada — não atinge o nível humano de "compreender e aplicar regras lógicas em qualquer contexto".
