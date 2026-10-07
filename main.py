"""Comparação reproduzível de classificadores com validação cruzada aninhada."""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def candidates():
    return {
        "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()), {
            "kneighborsclassifier__n_neighbors": [3, 5, 9],
            "kneighborsclassifier__weights": ["uniform", "distance"],
            "kneighborsclassifier__p": [1, 2],
        }),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=1), {
            "n_estimators": [80, 160], "max_depth": [None, 8],
            "min_samples_leaf": [1, 3],
        }),
        "SVM": (make_pipeline(StandardScaler(), SVC()), {
            "svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1],
            "svc__kernel": ["rbf"],
        }),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("resultados"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    data.frame.to_csv(args.output / "dataset.csv", index=False)
    outer = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
    rows, summaries, curves = [], [], []
    predictions = pd.DataFrame({"amostra": np.arange(len(y)), "classe_real": y})
    for name, (estimator, grid) in candidates().items():
        slug = name.lower().replace(" ", "_")
        folder = args.output / slug
        folder.mkdir(exist_ok=True)
        # A busca só recebe o treinamento do fold externo; o scaler faz parte dela.
        search = GridSearchCV(estimator, grid, scoring="accuracy", n_jobs=args.jobs,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              error_score="raise")
        oof = np.full(len(y), -1, dtype=int)
        scores = []
        for fold, (train, test) in enumerate(outer, 1):
            search.fit(X.iloc[train], y.iloc[train])
            oof[test] = search.predict(X.iloc[test])
            score = accuracy_score(y.iloc[test], oof[test])
            scores.append(score)
            rows.append({"modelo": name, "fold": fold, "n_teste": len(test),
                         "acuracia": score, "melhor_acuracia_interna": search.best_score_,
                         "parametros": json.dumps(search.best_params_)})
            print(f"{name}: fold {fold}/10, acurácia={score:.4f}", flush=True)
        assert np.all(oof >= 0)
        predictions[name] = oof
        matrix = confusion_matrix(y, oof, labels=[0, 1, 2])
        pd.DataFrame(matrix, index=data.target_names, columns=data.target_names).to_csv(
            folder / "matriz_confusao.csv")
        ConfusionMatrixDisplay(matrix, display_labels=data.target_names).plot(cmap="Blues")
        plt.title(f"{name} — predições fora do treinamento")
        plt.tight_layout()
        plt.savefig(folder / "matriz_confusao.png", dpi=160)
        plt.close()
        # A curva também refaz a busca exclusivamente em cada subconjunto de treino.
        print(f"{name}: calculando curva de aprendizagem...", flush=True)
        sizes, train_scores, valid_scores = learning_curve(
            search, X, y, cv=outer, train_sizes=[0.3, 0.5, 0.7, 1.0],
            scoring="accuracy", shuffle=True, random_state=SEED, n_jobs=1,
            error_score="raise")
        fig, ax = plt.subplots()
        for label, values in [("Treinamento", train_scores), ("Validação", valid_scores)]:
            mean, std = values.mean(axis=1), values.std(axis=1, ddof=1)
            ax.plot(sizes, mean, "o-", label=label)
            ax.fill_between(sizes, mean - std, mean + std, alpha=0.15)
        ax.set(xlabel="Amostras de treinamento", ylabel="Acurácia",
               title=f"Curva de aprendizagem — {name}", ylim=(0, 1.02))
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "curva_aprendizagem.png", dpi=160)
        plt.close(fig)
        for i, size in enumerate(sizes):
            for fold in range(10):
                curves.append({"modelo": name, "n_treino": int(size), "fold": fold + 1,
                               "acuracia_treino": train_scores[i, fold],
                               "acuracia_validacao": valid_scores[i, fold]})
        summaries.append({"modelo": name, "acuracia_media": np.mean(scores),
                          "desvio_padrao": np.std(scores, ddof=1),
                          "acuracia_oof": accuracy_score(y, oof),
                          "treino_curva_final": train_scores[-1].mean(),
                          "validacao_curva_final": valid_scores[-1].mean()})
    summary = pd.DataFrame(summaries).sort_values("acuracia_media", ascending=False)
    summary.to_csv(args.output / "comparacao.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output / "folds.csv", index=False)
    pd.DataFrame(curves).to_csv(args.output / "curvas.csv", index=False)
    predictions.to_csv(args.output / "predicoes.csv", index=False)
    fig, ax = plt.subplots()
    ax.bar(summary.modelo, summary.acuracia_media, yerr=summary.desvio_padrao, capsize=5)
    ax.set(ylabel="Acurácia média ± desvio padrão", ylim=(0, 1.08), title="Comparação em 10 folds")
    fig.tight_layout()
    fig.savefig(args.output / "comparacao.png", dpi=160)
    plt.close(fig)
    report = ["# Comparação de classificadores no Wine\n",
              "178 amostras, 13 atributos numéricos, três classes. Semente: 42.\n",
              "Validação estratificada: 10 folds externos e 5 internos para GridSearchCV. "
              "Os três modelos usam os mesmos folds externos. Desvio padrão amostral (ddof=1).\n",
              "| Modelo | Acurácia média | Desvio padrão | Acurácia OOF |",
              "|---|---:|---:|---:|"]
    for row in summary.itertuples():
        report.append(f"| {row.modelo} | {row.acuracia_media:.2%} | {row.desvio_padrao:.2%} | {row.acuracia_oof:.2%} |")
    report.extend(["", f"Maior média observada: **{summary.iloc[0]['modelo']}**. "
                   "Essa ordenação não demonstra superioridade estatística; os folds compartilham dados de treinamento. "
                   "A base é pequena e bem separável, limitando a generalização da conclusão.",
                   "", "## Curvas de aprendizagem", ""])
    for row in summary.itertuples():
        report.append(f"- {row.modelo}: no maior tamanho, treino={row.treino_curva_final:.2%}, "
                      f"validação={row.validacao_curva_final:.2%}, "
                      f"diferença={row.treino_curva_final-row.validacao_curva_final:.2%}.")
    report.extend(["", "Um desempenho alto no treino acompanhado de validação bem inferior sugere overfitting. "
                   "Desempenhos baixos e próximos podem sugerir underfitting. "
                   "As curvas são diagnósticas, não uma prova isolada; a faixa sombreada representa um desvio padrão, não intervalo de confiança.",
                   "", "As matrizes usam linhas reais e colunas previstas, com classes 0, 1 e 2. "
                   "Cada amostra é prevista uma vez fora do treinamento (OOF). Em multiclasse, "
                   "TP é a diagonal, FN a soma da linha menos TP, FP a soma da coluna menos TP "
                   "e TN o total menos TP, FN e FP para cada classe.",
                   "", "A acurácia OOF pondera as amostras; a média dos folds atribui peso igual a cada fold. "
                   "Pequenas diferenças decorrem dos tamanhos 17/18 dos folds.",
                   "", "Fonte: Aeberhard, S. & Forina, M. (1992). Wine. UCI Machine Learning Repository. "
                   "https://doi.org/10.24432/C5PC7J (CC BY 4.0)."])
    (args.output / "relatorio.md").write_text("\n".join(report), encoding="utf-8")
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                "seed": SEED}
    (args.output / "ambiente.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
