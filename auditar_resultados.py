"""Confere os artefatos e, opcionalmente, reproduz a avaliação externa completa.

Não altera o experimento original. Salva a análise complementar em resultados/revisao.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from main import SEED, candidates


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduzir', action='store_true',
                        help='Refaz as 30 buscas internas e compara com os arquivos originais.')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / 'resultados'
    output = source / 'revisao'

    output.mkdir(exist_ok=True)
    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    dataset = pd.read_csv(source / 'dataset.csv')
    predictions = pd.read_csv(source / 'predicoes.csv')
    folds = pd.read_csv(source / 'folds.csv')
    summary = pd.read_csv(source / 'comparacao.csv').set_index('modelo')
    curves = pd.read_csv(source / 'curvas.csv')
    pd.testing.assert_frame_equal(dataset, wine.frame, check_exact=False, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(predictions.amostra, np.arange(len(y)))
    np.testing.assert_array_equal(predictions.classe_real, y)

    assert not X.isna().any().any() and not X.duplicated().any()
    assert len(folds) == 30 and len(curves) == 120 and len(predictions) == 178
    outer = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
    coverage = np.zeros(len(y), dtype=int)
    fold_map = np.zeros(len(y), dtype=int)
    baseline = np.full(len(y), -1, dtype=int)
    baseline_scores = []

    for fold, (train, test) in enumerate(outer, 1):
        assert not np.intersect1d(train, test).size
        coverage[test] += 1
        fold_map[test] = fold
        dummy = DummyClassifier(strategy='most_frequent').fit(X.iloc[train], y.iloc[train])
        baseline[test] = dummy.predict(X.iloc[test])
        baseline_scores.append(accuracy_score(y.iloc[test], baseline[test]))
        
    np.testing.assert_array_equal(coverage, 1)
    pd.DataFrame({'amostra': predictions.amostra, 'fold_externo': fold_map}).to_csv(
        output / 'particoes.csv', index=False)
    models = candidates()
    metrics, per_class, reproductions = [], [], []

    for name, (estimator, grid) in models.items():
        saved = folds[folds.modelo == name].sort_values('fold')
        np.testing.assert_array_equal(saved.fold, np.arange(1, 11))
        oof = predictions[name].to_numpy()
        assert set(oof) <= {0, 1, 2}
        cm = confusion_matrix(y, oof, labels=[0, 1, 2])
        saved_cm = pd.read_csv(source / name.lower().replace(' ', '_') / 'matriz_confusao.csv', index_col=0)
        np.testing.assert_array_equal(cm, saved_cm.to_numpy())

        for fold, (train, test) in enumerate(outer, 1):
            row = saved.iloc[fold - 1]
            assert row.n_teste == len(test)
            np.testing.assert_allclose(row.acuracia, accuracy_score(y.iloc[test], oof[test]), atol=1e-14)
            assert json.loads(row.parametros) in list(ParameterGrid(grid))
            if args.reproduzir:
                search = GridSearchCV(estimator, grid, scoring='accuracy', n_jobs=args.jobs,
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), error_score='raise')
                search.fit(X.iloc[train], y.iloc[train])
                np.testing.assert_array_equal(search.predict(X.iloc[test]), oof[test])
                np.testing.assert_allclose(search.best_score_, row.melhor_acuracia_interna, atol=1e-14)
                assert search.best_params_ == json.loads(row.parametros)
                # Empates exatos seguem a ordem da grade, e não demonstram preferência única.
                ties = int(np.sum(search.cv_results_['rank_test_score'] == 1))
                reproductions.append({'modelo': name, 'fold': fold, 'reproduzido': True,
                                      'configuracoes_empatadas_no_topo': ties})
                print(f'{name}: fold {fold}/10 reproduzido, {ties} configuração(ões) no topo.', flush=True)
        np.testing.assert_allclose(summary.loc[name, 'acuracia_media'], saved.acuracia.mean(), atol=1e-14)
        np.testing.assert_allclose(summary.loc[name, 'desvio_padrao'], saved.acuracia.std(ddof=1), atol=1e-14)
        np.testing.assert_allclose(summary.loc[name, 'acuracia_oof'], accuracy_score(y, oof), atol=1e-14)
        part = curves[curves.modelo == name]
        assert set(part.n_treino) == {48, 80, 112, 160}

        for size, group in part.groupby('n_treino'):
            np.testing.assert_array_equal(np.sort(group.fold), np.arange(1, 11))
            assert group[['acuracia_treino', 'acuracia_validacao']].ge(0).all().all()
            assert group[['acuracia_treino', 'acuracia_validacao']].le(1).all().all()
        largest = part[part.n_treino == 160]
        np.testing.assert_allclose(summary.loc[name, 'treino_curva_final'], largest.acuracia_treino.mean())
        np.testing.assert_allclose(summary.loc[name, 'validacao_curva_final'], largest.acuracia_validacao.mean())
        metrics.append({'modelo': name, 'acuracia_media': saved.acuracia.mean(),
                        'desvio_padrao': saved.acuracia.std(ddof=1), 'acuracia_oof': accuracy_score(y, oof),
                        'acuracia_balanceada_oof': balanced_accuracy_score(y, oof),
                        'f1_macro_oof': f1_score(y, oof, average='macro'), 'erros': int((y != oof).sum())})
        precision, recall, f1, support = precision_recall_fscore_support(y, oof, labels=[0, 1, 2])

        for label in range(3):
            tp = int(cm[label, label])
            fn = int(cm[label].sum() - tp)
            fp = int(cm[:, label].sum() - tp)
            per_class.append({'modelo': name, 'classe': label, 'precisao': precision[label],
                              'revocacao': recall[label], 'f1': f1[label], 'suporte': int(support[label]),
                              'TP': tp, 'FN': fn, 'FP': fp, 'TN': len(y) - tp - fn - fp})
    metrics = pd.DataFrame(metrics).sort_values('acuracia_media', ascending=False)
    metrics.to_csv(output / 'metricas_complementares.csv', index=False)
    pd.DataFrame(per_class).to_csv(output / 'metricas_por_classe.csv', index=False)
    comparisons = []

    for a, b in combinations(models, 2):
        right_a, right_b = predictions[a] == y, predictions[b] == y
        comparisons.append({'modelo_a': a, 'modelo_b': b,
            'somente_a_acerta': int((right_a & ~right_b).sum()),
            'somente_b_acerta': int((~right_a & right_b).sum()),
            'ambos_acertam': int((right_a & right_b).sum()),
            'ambos_erram': int((~right_a & ~right_b).sum())})
    pd.DataFrame(comparisons).to_csv(output / 'erros_pareados.csv', index=False)

    if args.reproduzir:
        pd.DataFrame(reproductions).to_csv(output / 'reproducao.csv', index=False)
    manifest = {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in
                ['dataset.csv', 'predicoes.csv', 'folds.csv', 'comparacao.csv', 'curvas.csv']}
    manifest['codigo_main_sha256'] = hashlib.sha256((root / 'main.py').read_bytes()).hexdigest()
    (output / 'arquivos_verificados.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    baseline_result = {'media_folds': float(np.mean(baseline_scores)),
                       'desvio_folds': float(np.std(baseline_scores, ddof=1)),
                       'acuracia_oof': accuracy_score(y, baseline)}
    (output / 'baseline.json').write_text(json.dumps(baseline_result, indent=2), encoding='utf-8')

    # Tabela derivada diretamente das predições, pronta para inclusão no artigo.
    tex = [r'\begin{tabular}{lrr}', r'\toprule',
           r'Modelo & Acurácia balanceada (\%) & F1 macro (\%) \\', r'\midrule']
    for row in metrics.itertuples():
        balanced = f'{100 * row.acuracia_balanceada_oof:.2f}'.replace('.', ',')
        macro = f'{100 * row.f1_macro_oof:.2f}'.replace('.', ',')
        tex.append(f'{row.modelo} & {balanced} & {macro} ' + r'\\')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (root / 'artigo' / 'metricas_complementares.tex').write_text('\n'.join(tex), encoding='utf-8')

    print(metrics.to_string(index=False))
    print('Conferência concluída. A revisão não reexecuta as curvas de aprendizagem.', flush=True)


if __name__ == '__main__':
    main()
