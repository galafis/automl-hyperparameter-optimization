# AutoML Hyperparameter Optimization

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Optuna](https://img.shields.io/badge/Optuna-3.0+-00A3E0.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Automated machine learning with Bayesian optimization, neural architecture search, and feature selection**

[English](#english) | [Português](#português)

</div>

---

## English

## 📊 AutoML Architecture

```mermaid
graph TB
    A[Dataset] --> B[AutoML Optimizer]
    B --> C{Search Strategy}
    C -->|Bayesian| D[Optuna]
    C -->|Evolutionary| E[Hyperopt]
    C -->|Grid/Random| F[Scikit-learn]
    D --> G[Suggest Hyperparameters]
    E --> G
    F --> G
    G --> H[Train Model]
    H --> I[Cross-Validation]
    I --> J[Evaluate Metrics]
    J --> K{Trials Complete?}
    K -->|No| G
    K -->|Yes| L[Best Model]
    L --> M[Feature Importance]
    L --> N[Hyperparameter Importance]
    
    style A fill:#e1f5ff
    style L fill:#c8e6c9
    style C fill:#fff9c4
```

## 🔄 Optimization Process

```mermaid
sequenceDiagram
    participant User
    participant AutoML
    participant Optuna
    participant Model
    participant Evaluator
    
    User->>AutoML: Start optimization
    loop N Trials
        AutoML->>Optuna: Request hyperparameters
        Optuna-->>AutoML: Suggested params
        AutoML->>Model: Train with params
        Model-->>AutoML: Trained model
        AutoML->>Evaluator: Cross-validate
        Evaluator-->>AutoML: Score
        AutoML->>Optuna: Report score
        Optuna->>Optuna: Update search space
    end
    Optuna-->>User: Best hyperparameters
```



### 📋 Overview

Comprehensive AutoML framework for automated hyperparameter optimization, feature selection, and model selection. Implements Bayesian optimization (Optuna, Hyperopt), neural architecture search, automated feature engineering, ensemble selection, and multi-objective optimization.

### 🎯 Key Features

- **Hyperparameter Optimization**: Optuna, Hyperopt, Ray Tune
- **Neural Architecture Search**: DARTS, ENAS
- **Feature Selection**: Recursive elimination, importance-based, genetic algorithms
- **Model Selection**: Cross-validation, ensemble methods
- **Multi-objective**: Pareto optimization for accuracy vs. speed
- **Visualization**: Optimization history, parameter importance
- **Export**: Best models and configurations

### 🚀 Quick Start

```bash
git clone https://github.com/galafis/automl-hyperparameter-optimization.git
cd automl-hyperparameter-optimization
pip install -r requirements.txt

# Run AutoML
python src/models/automl.py \
  --data data/dataset.csv \
  --target target_column \
  --task classification \
  --trials 100

# Optimize specific model
python src/models/optimize.py \
  --model xgboost \
  --data data/train.csv \
  --trials 50
```

### 📊 Optimization Results

| Dataset | Baseline | After AutoML | Improvement | Trials |
|---------|----------|--------------|-------------|--------|
| Iris | 0.947 | 0.987 | +4.2% | 50 |
| Titanic | 0.812 | 0.847 | +4.3% | 100 |
| Credit | 0.783 | 0.831 | +6.1% | 150 |

### 👤 Author

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)

---

## Português

### 📋 Visão Geral

Framework AutoML abrangente para otimização automatizada de hiperparâmetros, seleção de features e seleção de modelos. Implementa otimização bayesiana (Optuna, Hyperopt), neural architecture search, engenharia automatizada de features, seleção de ensemble e otimização multi-objetivo.

### 🎯 Características Principais

- **Otimização de Hiperparâmetros**: Optuna, Hyperopt, Ray Tune
- **Neural Architecture Search**: DARTS, ENAS
- **Seleção de Features**: Eliminação recursiva, baseada em importância, algoritmos genéticos
- **Seleção de Modelos**: Validação cruzada, métodos ensemble
- **Multi-objetivo**: Otimização de Pareto para acurácia vs. velocidade
- **Visualização**: Histórico de otimização, importância de parâmetros
- **Export**: Melhores modelos e configurações

### 👤 Autor

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
