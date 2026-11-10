# Aprendizagem de Máquina — PPGCC/UFPI

Repositório de trabalhos práticos da disciplina de **Aprendizagem de Máquina** do Programa de Pós-Graduação em Ciência da Computação (**PPGCC**) da Universidade Federal do Piauí (**UFPI**).

**Aluno:** Heverton Macedo  
**Laboratório:** VipLab UFPI

---

## Projetos

### 🌲 Glass Identification — Random Forest (com comparativo contra Árvore de Decisão)

**Arquivo:** `glass_random_forest.ipynb`

Pipeline completo de classificação multiclasse aplicado ao [UCI Glass Identification Dataset](https://archive.ics.uci.edu/dataset/42/glass+identification), utilizando **Random Forest** (ensemble de árvores com bagging e sorteio aleatório de features). O mesmo notebook treina uma **Árvore de Decisão** otimizada para comparação direta e avalia o impacto do **SMOTE**.

**Dataset:** 214 instâncias, 9 features químicas contínuas (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe), 6 classes de tipos de vidro (fortemente desbalanceadas).

**Pipeline:**
1. Carregamento dos dados via UCI ML Repository
2. Análise Exploratória de Dados (EDA) e correlações
3. Remoção de duplicatas (1 removida → 213 registros)
4. Detecção e remoção de outliers (Z-score > 3,5 → 202 registros)
5. Engenharia de features (flags `Ba_presente`/`Fe_presente` + razões `Na_Ca`, `Si_Al`, `Mg_Al`; 9 → 14 features)
6. Divisão estratificada treino/teste (161 / 41)
7. **SMOTE aplicado somente no treino, após o split** (evita data leakage; teste mantém a distribuição original)
8. Análise do impacto de `n_estimators`
9. GridSearchCV + StratifiedKFold (10 folds, 432 combinações) com hiperparâmetros do Random Forest (`max_features`, `max_samples`, `bootstrap`, etc.)
10. Avaliação com F1-macro, acurácia, relatório por classe e matriz de confusão
11. Importância média das features (com desvio padrão entre árvores) e visualização da diversidade entre árvores
12. Comparativo Random Forest × Árvore de Decisão (global e por classe)
13. **Etapa extra:** impacto do balanceamento — modelos com vs. sem SMOTE

**Resultados (conjunto de teste, com SMOTE):**

| Modelo | Acurácia | F1-macro |
|---|---|---|
| Árvore de Decisão | 65,85% | 63,44% |
| Random Forest | **87,80%** | **91,32%** |

- Melhores hiperparâmetros do RF: `n_estimators=200`, `max_depth=None`, `max_features=sqrt`, `min_samples_split=5`, `min_samples_leaf=1`, `bootstrap=False`
- F1-macro em validação cruzada: 89,85%
- Features mais importantes: `Na_Ca_ratio`, `Mg_Al_ratio`, `Mg`, `Ba`
- Como o melhor modelo usa `bootstrap=False`, o OOB Score não está disponível
- O SMOTE beneficiou principalmente as classes minoritárias (ex.: `Veh.Win.Float`, F1 0,67 → 0,86 no RF)

> **Nota:** o notebook de Árvore de Decisão isolado (`glass_decision_tree.ipynb`) foi removido do repositório; a Árvore de Decisão agora é treinada e comparada dentro do notebook de Random Forest.

---

## Tecnologias

- Python 3
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas / numpy / scipy
- matplotlib / seaborn
- ucimlrepo

## Como Executar

```bash
pip install ucimlrepo imbalanced-learn scikit-learn scipy pandas numpy matplotlib seaborn
jupyter notebook
```

Abra `glass_random_forest.ipynb` e execute as células em ordem.

---

## Referência do Dataset

> German, B. (1987). *Glass Identification*. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P
