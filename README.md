# Aprendizagem de Máquina — PPGCC/UFPI

Repositório de trabalhos práticos da disciplina de **Aprendizagem de Máquina** do Programa de Pós-Graduação em Ciência da Computação (**PPGCC**) da Universidade Federal do Piauí (**UFPI**).

**Aluno:** Heverton Macedo  
**Laboratório:** VipLab UFPI

---

## Projetos

### 🔍 Glass Identification — Árvore de Decisão

**Arquivo:** `glass_decision_tree.ipynb`

Pipeline completo de classificação multiclasse aplicado ao [UCI Glass Identification Dataset](https://archive.ics.uci.edu/dataset/42/glass+identification), utilizando **Árvore de Decisão** como classificador.

**Dataset:** 214 instâncias, 9 features químicas contínuas (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe), 6 classes de tipos de vidro.

**Pipeline:**
1. Carregamento dos dados via UCI ML Repository
2. Análise Exploratória de Dados (EDA)
3. Remoção de duplicatas
4. Detecção e remoção de outliers (Z-score > 3,5)
5. Balanceamento de classes com SMOTE
6. Engenharia de features (flags binárias + razões químicas)
7. Divisão estratificada treino/teste
8. Otimização via GridSearchCV + StratifiedKFold (10 folds)
9. Avaliação com F1-macro, acurácia e matriz de confusão

**Resultados:**
- Acurácia: **89,53%**
- F1-macro: **89,07%**
- Melhores hiperparâmetros: `criterion=entropy`, `max_depth=7`

---

### 🌲 Glass Identification — Random Forest

**Arquivo:** `glass_random_forest.ipynb`

Extensão do pipeline anterior utilizando **Random Forest** (ensemble de árvores de decisão com bagging e amostragem aleatória de features), com comparativo direto contra a Árvore de Decisão única.

**Pipeline:** Mesmo pré-processamento do notebook anterior, com etapas adicionais:
- Análise do impacto do número de estimadores (`n_estimators`)
- GridSearchCV ampliado com hiperparâmetros exclusivos do Random Forest (`max_features`, `max_samples`, `bootstrap`)
- Cálculo do Out-of-Bag (OOB) Score
- Visualização da diversidade entre árvores individuais da floresta
- Importância média das features com desvio padrão entre árvores

**Resultados:**
- Acurácia: **94,19%**
- F1-macro: **94,00%**
- Melhores hiperparâmetros: `n_estimators=100`, `max_depth=None`, `max_features=sqrt`, `bootstrap=True`
- **Melhoria sobre Árvore de Decisão: +4,93 p.p. em F1-macro**

---

## Tecnologias

- Python 3
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas / numpy
- matplotlib / seaborn
- ucimlrepo

## Como Executar

```bash
pip install ucimlrepo imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
jupyter notebook
```

Abra o notebook desejado e execute as células em ordem.

---

## Referência do Dataset

> German, B. (1987). *Glass Identification*. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P
