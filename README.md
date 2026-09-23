# Drug Discovery Virtual Screening

Projeto de ciência de dados para análise exploratória, treinamento e uso de modelos de *machine learning* na priorização virtual de pares composto-proteína. O fluxo inclui classificação de atividade, estimativa de afinidade de ligação e predição em lote.

O repositório contém um conjunto de dados de referência simulado com 2.000 pares composto-proteína. Os resultados servem para demonstrar o fluxo de análise e modelagem; não substituem ensaios experimentais nem devem ser interpretados como validação clínica.

![Distribuição de afinidade de ligação e atividade](img/02_target_distribution.png)

| Resumo do conjunto de referência | Valor |
|---|---:|
| Registros | 2.000 |
| Colunas | 17 |
| Registros ativos | 608 (30,4%) |
| Treino / teste | 1.600 / 400 |

## Resultados principais

Os valores abaixo foram medidos em um conjunto de teste separado com 400 registros. O classificador final é uma regressão logística ajustada; o regressor final é uma regressão linear.

| Tarefa | Modelo | Métrica | Resultado |
|---|---|---|---:|
| Classificação de atividade | Regressão logística | ROC-AUC | 0,9578 |
| Classificação de atividade | Regressão logística | Average precision | 0,9265 |
| Classificação de atividade | Regressão logística | F1, limiar 0,465 | 0,8122 |
| Estimativa de afinidade (pKi) | Regressão linear | RMSE | 0,7392 |
| Estimativa de afinidade (pKi) | Regressão linear | MAE | 0,3370 |
| Estimativa de afinidade (pKi) | Regressão linear | R² | 0,6155 |

O classificador escolhe o limiar de decisão usando previsões *out-of-fold* nos dados de treinamento. O conjunto de teste não é usado para essa escolha. Consulte `models/model_metadata.json` para métricas, limiar e versões das bibliotecas associadas aos artefatos salvos.

## Visão geral dos dados e modelos

O rótulo binário `active` é definido exatamente por `binding_affinity >= 7.0`. Por isso, `binding_affinity` não pode ser usado como variável de entrada do classificador. Da mesma forma, `active` é excluído das entradas do regressor. O identificador único `compound_id` também é removido das variáveis preditoras.

As figuras abaixo ilustram achados da análise exploratória, do benchmark e da etapa de predição.

<p align="center">
  <img src="img/03_target_leakage.png" alt="Relação determinística entre afinidade e atividade" width="49%">
  <img src="img/09_target_associations.png" alt="Associações entre descritores e afinidade" width="49%">
</p>

<p align="center">
  <img src="img/ml_03_roc_pr_curves.png" alt="Curvas ROC e precisão-revocação dos classificadores" width="49%">
  <img src="img/ml_10_regression_diagnostics.png" alt="Diagnósticos das previsões de afinidade" width="49%">
</p>

<p align="center">
  <img src="img/pred_02_shortlist.png" alt="Lista priorizada de compostos com maior pontuação" width="75%">
</p>

## Notebooks

Execute os notebooks a partir da raiz do repositório para que os caminhos relativos funcionem.

1. [`01_exploratory_data_analysis.ipynb`](notebook/01_exploratory_data_analysis.ipynb) — qualidade dos dados, análise univariada e multivariada, relação entre os alvos, valores ausentes, outliers e recomendações para modelagem.
2. [`02_machine_learning_models.ipynb`](notebook/02_machine_learning_models.ipynb) — engenharia de atributos, tratamento de categorias, benchmark de dez classificadores e dez regressores, ajuste do limiar, avaliação e gravação dos modelos.
3. [`03_batch_prediction.ipynb`](notebook/03_batch_prediction.ipynb) — carregamento dos artefatos, predição em lote, exportação dos resultados e verificações de consistência.

## Estrutura do repositório

```text
Drug-Discovery-Virtual/
├── img/                 # Figuras usadas neste README
├── input/               # CSV de entrada do projeto
├── models/              # Modelos finais, metadados e script de predição
│   └── all_models/      # Classificadores e regressores avaliados
├── notebook/            # EDA, benchmark e predição em lote
├── output/              # Tabelas, figuras e arquivos de predição gerados
│   └── figures/         # Figuras completas das análises
├── src/                 # Código-fonte auxiliar de predição
├── requirements.txt
└── README.md
```

## Instalação

Use Python 3.12 ou superior. Os artefatos serializados foram gerados com Python 3.12.4 e scikit-learn 1.3.2; mantenha a versão do scikit-learn indicada nos arquivos de dependências ao carregar os modelos salvos.

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para abrir e executar os notebooks, inicie o JupyterLab na raiz do projeto:

```bash
jupyter lab
```

As dependências para os modelos finais também estão listadas em [`models/requirements.txt`](models/requirements.txt). `xgboost` e `lightgbm` são necessários para reproduzir o benchmark completo; a inferência com os dois modelos finais não depende deles.

## Fazer predições

O script de linha de comando carrega os modelos salvos em `models/` e grava a tabela de resultados:

```bash
python models/predict.py input/drug_discovery_virtual_screening.csv output/predictions_scored.csv
```

Se o caminho de saída for omitido, o script cria um arquivo com sufixo `_scored.csv` ao lado do arquivo de entrada:

```bash
python models/predict.py caminho/para/compostos.csv
```

Também é possível chamar o modelo pelo Python:

```python
import pandas as pd

from models.predict import ScreeningModel

compostos = pd.read_csv("caminho/para/compostos.csv")
modelo = ScreeningModel()
resultado = modelo.predict(compostos)
resultado.to_csv("predictions_scored.csv", index=False)
```

### Colunas exigidas na entrada

O arquivo enviado para predição deve conter estas colunas:

```text
protein_id
molecular_weight
logp
h_bond_donors
h_bond_acceptors
rotatable_bonds
polar_surface_area
compound_clogp
protein_length
protein_pi
hydrophobicity
binding_site_size
mw_ratio
logp_pi_interaction
```

`compound_id` é opcional e, quando fornecido, é copiado para a saída para identificar cada registro. Valores ausentes são tratados pelo pipeline; proteínas não vistas no treinamento recebem uma codificação reservada. Colunas extras, inclusive `active` e `binding_affinity`, não são usadas para gerar as previsões.

As colunas de resultado são `activity_probability`, `predicted_active` e `predicted_binding_affinity`, além de `compound_id` e `protein_id` quando informados na entrada. `predicted_active` usa o limiar de decisão salvo nos metadados do modelo.

## Arquivos gerados

- `output/eda_*.csv` e `output/eda_summary.json`: estatísticas e resultados da análise exploratória.
- `output/ml_*.csv`: métricas do benchmark, importância de atributos e varredura de limiar.
- `output/predictions.csv` e `output/predictions_binary.csv`: predições do notebook de processamento em lote.
- `output/figures/`: figuras geradas pelos notebooks; as imagens usadas neste README estão em `img/`.
- `models/best_classifier.joblib` e `models/best_regressor.joblib`: pipelines finais serializados.
- `models/model_metadata.json`: contrato de entrada, ordem dos atributos, limiar, métricas e versões usadas na criação dos modelos.

## Licença

Distribuído sob a licença MIT. Consulte [`LICENSE`](LICENSE).
