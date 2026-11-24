# Detecção de fraude em transações de cartão de crédito

Projeto do curso de Agentes de IA (DIO / Bradesco). Treinei e comparei modelos para detectar fraude em transações reais de cartão de crédito, um problema em que a fraude é rara e a acurácia engana.

- **Notebook:** [`deteccao-fraude-cartao.ipynb`](deteccao-fraude-cartao.ipynb), feito no Google Colab
- **Base de dados:** [creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv). O notebook baixa direto pelo link. O arquivo não está neste repositório porque passa do limite de 100 MB do GitHub.

## O problema e por que o desbalanceamento muda a avaliação

A base tem 284.807 transações. Cerca de 99,83% são normais e só 0,17% são fraude. Um modelo que responde "não é fraude" para tudo acerta quase 100% e nunca detecta uma fraude. Por isso a acurácia sozinha engana (nos meus modelos ela ficou em 1,00 mesmo quando o recall era baixo) e olhei a classe de fraude:

- **Recall:** das fraudes reais, quantas o modelo encontrou.
- **Precisão:** das transações que o modelo marcou como fraude, quantas eram fraude de verdade.
- **F1:** média harmônica entre precisão e recall.
- **Curva ROC e curva Precisão × Recall:** mostram o desempenho em vários limiares. A curva Precisão × Recall é a mais informativa com dados desbalanceados.

## O que fiz na preparação dos dados

- Exploração com `df.head()` e `value_counts(normalize=True)` para medir a proporção de fraudes.
- `np.log1p` em `Amount`, para comprimir valores muito grandes (o `1p` evita problema com zero).
- `StandardScaler` no valor transformado (média 0, desvio padrão 1).
- Separação em `X` (features) e `y` (`Class`) e divisão treino/teste de 70/30 com `random_state=42` e `stratify=y`, para manter a proporção de fraudes nos dois conjuntos. O teste ficou com 85.443 transações, sendo 148 fraudes.

## Comparação entre os modelos (classe de fraude, no teste)

Modelos treinados: Regressão Logística (baseline), Random Forest e XGBoost. Também testei undersampling, SMOTE, Pipeline com ajuste de limiar e `GridSearchCV` (métrica: recall).

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| Random Forest | 0,837 | 0,764 | **0,799** |
| XGBoost | 0,836 | 0,723 | 0,775 |
| XGBoost + GridSearch | 0,836 | 0,723 | 0,775 |
| Pipeline LR (limiar 0,1) | 0,752 | 0,716 | 0,734 |
| Regressão Logística (baseline) | 0,848 | 0,642 | 0,731 |
| Pipeline LR (limiar 0,5) | **0,860** | 0,622 | 0,722 |
| Pipeline LR (limiar 0,3) | 0,795 | 0,655 | 0,719 |
| Reg. Logística + SMOTE | 0,141 | 0,851 | 0,242 |
| Reg. Logística + undersampling | 0,056 | **0,878** | 0,106 |

AUC da curva ROC: Random Forest 0,968, Regressão Logística 0,943 e XGBoost 0,913.

O `GridSearchCV` escolheu `max_depth=5` e `n_estimators=100`, que são os mesmos valores do XGBoost que eu já tinha treinado. Por isso os dois têm resultado idêntico.

## Limiar de decisão e SHAP

**Limiar.** O padrão é 0,5. Comparei três limiares no Pipeline de regressão logística:

| Limiar | Precisão | Recall | F1 |
|---|---|---|---|
| 0,5 | 0,860 | 0,622 | 0,722 |
| 0,3 (usado na aula) | 0,795 | 0,655 | 0,719 |
| 0,1 | 0,752 | 0,716 | 0,734 |

Baixar o limiar fez o recall subir (de 0,62 para 0,72) e a precisão cair (de 0,86 para 0,75), como o esperado. Escolhi **0,1**, porque teve o maior recall e o maior F1 entre os três. O custo é mais falsos alarmes. Se o objetivo fosse incomodar o mínimo de clientes legítimos, o 0,5 seria melhor. Escolhi o limiar olhando o próprio teste, o que é uma limitação.

**SHAP.** As variáveis que mais influenciaram o XGBoost foram **V16, V11 e V10**, nessa ordem, seguidas de V4. O gráfico de importância das variáveis mostrou o mesmo padrão: V16, V11 e V14 concentram quase toda a importância, e a maioria das variáveis tem peso perto de zero. Como `V1` a `V28` vêm de PCA e são anônimas, sei quais pesam mais, mas não o que representam.

## Conclusões

- **Acurácia engana:** todos os modelos ficaram com acurácia perto de 1,00. A diferença entre eles aparece só no recall, na precisão e no F1 da classe de fraude.
- **Melhor modelo:** o Random Forest teve o melhor F1 (0,799), com recall de 0,764 e precisão de 0,837, e a maior AUC (0,968). Foi o modelo mais equilibrado.
- **Trade-off:** o undersampling (recall 0,878) e o SMOTE (recall 0,851) encontraram mais fraudes, mas a precisão caiu para 0,056 e 0,141. Na prática, quase todos os alertas seriam falsos alarmes. O mesmo trade-off apareceu ao baixar o limiar da regressão logística.
- **Balanceamento:** neste teste, usar pesos de classe (como no Random Forest) funcionou melhor do que equilibrar os dados. Isso vale para esta base e esta divisão, e a única forma de saber é testar.
- **Explicabilidade:** poucas variáveis (V16, V11, V10) concentram a decisão do modelo, o que ajuda a justificar por que uma transação foi marcada como fraude.

## O que mudei em relação ao que a Expert fez

Pelo que vi nas aulas, as diferenças do meu notebook foram:

- **Balanceamento só no treino:** apliquei undersampling e oversampling (SMOTE) apenas no conjunto de treino e avaliei no mesmo teste original, para os resultados serem comparáveis com os outros modelos.
- **Mais de um limiar:** além de 0,3, testei 0,5 e 0,1 e comparei os três lado a lado.
- **Curvas juntas:** plotei as curvas ROC e Precisão × Recall dos três modelos principais no mesmo gráfico.
- **Tabela comparativa e resumo automático:** guardei as métricas de todos os modelos em uma tabela e gerei, no fim, um resumo com o melhor modelo em recall, precisão e F1, os melhores parâmetros do GridSearch e as variáveis mais importantes do SHAP.
- **SHAP no modelo ajustado:** expliquei o XGBoost escolhido pelo `GridSearchCV`, com gráfico de barras e gráfico de dispersão (*beeswarm*).

## Comparação com os resultados da aula

Valores citados pela instrutora (não são os meus):

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| Regressão Logística | 0,87 | 0,68 | 0,77 |
| Random Forest | | 0,76 | 0,79 |
| Pipeline com limiar 0,3 | 0,86 | 0,71 | |
| XGBoost | 0,93 | 0,78 | 0,85 |

Meus resultados ficaram próximos dos dela na regressão logística e no Random Forest (recall 0,76 e F1 0,80, contra 0,76 e 0,79). O XGBoost ficou abaixo do dela (recall 0,72 e F1 0,78, contra 0,78 e 0,85), possivelmente porque os parâmetros que usei são diferentes dos dela, que eu não vi. Na aula, o `GridSearchCV` escolheu `max_depth=3` e as variáveis mais importantes no SHAP foram V4 e V14. No meu caso foram `max_depth=5` e V16, V11 e V10.

## Limitações

- A padronização de `Amount` foi feita antes da divisão treino/teste, como nas aulas. O ideal é ajustar o scaler só no treino.
- Os limiares foram comparados no próprio conjunto de teste.
- Os resultados vêm de uma única divisão dos dados, com apenas 148 fraudes no teste, então pequenas diferenças entre modelos podem não ser significativas.

## Como rodar

1. Abra o notebook no Google Colab.
2. Vá em `Ambiente de execução` → `Executar tudo`.
