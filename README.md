# Predição de Falhas em Máquinas Industriais

Trabalho final da disciplina de **Machine Learning** da Pós-Graduação em IA Generativa e Aplicações com LLMs da **PUC Minas**.

**Autoria:** Everton Recalde Franco e Silva

**Aplicação publicada:** _link a ser adicionado após o deploy_

A partir de seis características operacionais de uma máquina, um modelo Random Forest estima a probabilidade de falha. O modelo foi treinado com o dataset [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B), do UCI Machine Learning Repository. A aplicação em Streamlit usa o modelo já treinado, sem retreinar.

## A aplicação

Informe os dados abaixo e clique em **Prever falha**. A aplicação mostra a classe prevista ("Com falha" ou "Sem falha") e a probabilidade estimada de falha. A classe é "Com falha" quando a probabilidade passa de 50%.

| Campo | Coluna no dataset | Valores aceitos |
|:---|:---|:---|
| Tipo do produto | `Type` | L, M ou H |
| Temperatura do ar (K) | `Air temperature [K]` | 295,3 a 304,5 |
| Temperatura do processo (K) | `Process temperature [K]` | 305,7 a 313,8 |
| Velocidade de rotação (rpm) | `Rotational speed [rpm]` | 1168 a 2886 |
| Torque (Nm) | `Torque [Nm]` | 3,8 a 76,6 |
| Desgaste da ferramenta (min) | `Tool wear [min]` | 0 a 253 |

Os limites são o mínimo e o máximo de cada variável no dataset. Valores fora dessas faixas são recusados com uma mensagem explicativa, porque um Random Forest não extrapola além do que viu no treino.

## O modelo

**Dados.** São 10.000 registros, sem valores ausentes nem duplicados, e apenas 3,39% deles têm falha. Com esse desbalanceamento, a acurácia sozinha não serve de referência: prever sempre "sem falha" já acertaria 96,61%.

**Pré-processamento.** `UDI` e `Product ID` foram descartados por serem identificadores. As colunas de modo de falha (`TWF`, `HDF`, `PWF`, `OSF` e `RNF`) também ficaram de fora, porque já indicam a própria falha. As variáveis numéricas passam por `StandardScaler` e `Type` passa por `OneHotEncoder`, em um Pipeline salvo junto com o classificador. Os dados foram divididos em 80% para treino e 20% para teste, de forma estratificada.

**Comparação de modelos.** Validação cruzada estratificada com 5 divisões, no conjunto de treino:

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|:---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,9704 | 0,7607 | 0,1993 | 0,3119 | 0,8938 |
| **Random Forest** | **0,9804** | **0,8876** | **0,4836** | **0,6231** | **0,9677** |

O Random Forest, com 300 árvores, foi o escolhido. Recall e F1-Score foram as métricas principais, porque em manutenção preditiva o custo de não detectar uma falha é alto.

**Resultado no conjunto de teste** (2.000 registros):

| Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|---:|---:|---:|---:|---:|
| 98,15% | 89,74% | 51,47% | 65,42% | 97,09% |

### Limitações

O recall de 51,47% significa que o modelo deixa passar quase metade das falhas reais. Um resultado "Sem falha" não garante que a máquina esteja segura, e este trabalho acadêmico não deve orientar decisões reais de manutenção.

Possíveis melhorias, apontadas no notebook: sobreamostragem (SMOTE), ajuste de hiperparâmetros com GridSearch e algoritmos de boosting, como XGBoost e LightGBM.

## Como executar localmente

Requer Python 3.12 ou 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

A aplicação abre em <http://localhost:8501>.

## Publicação no Streamlit Community Cloud

1. Em [share.streamlit.io](https://share.streamlit.io), crie um novo app apontando para este repositório e para o arquivo `app.py`.
2. Em **Advanced settings**, selecione **Python 3.12**.
3. Clique em **Deploy**.

As versões em `requirements.txt` são fixas de propósito. O modelo foi salvo com o scikit-learn 1.6.1 e não carrega em versões muito mais novas, como a 1.9.1. Essa versão do scikit-learn também não tem pacote pronto para Python 3.14. Se o modelo for retreinado com outra versão, atualize `requirements.txt`.

## Estrutura do repositório

| Arquivo | Conteúdo |
|:---|:---|
| [`app.py`](app.py) | Aplicação Streamlit |
| [`modelo_manutencao_preditiva.joblib`](modelo_manutencao_preditiva.joblib) | Pipeline treinado (pré-processamento e Random Forest) |
| [`requirements.txt`](requirements.txt) | Dependências com versões fixas |
| [`predictive_maintenance_ai4i_trabalho_final_ml.ipynb`](predictive_maintenance_ai4i_trabalho_final_ml.ipynb) | Análise exploratória, treino e avaliação dos modelos |
| [`ai4i2020.csv`](ai4i2020.csv) | Dataset AI4I 2020 |

O notebook foi executado no Google Colab e lê o CSV do Google Drive. Para executá-lo em outro ambiente, ajuste o caminho na célula de carregamento dos dados.
