"""
Predicting Student Dropout Using Machine Learning
=================================================

Full pipeline for the research proposal:
  RQ1  Compare Logistic Regression, Random Forest, XGBoost and LightGBM at two
       prediction points: T0 (enrolment data only) and T1 (+ 1st-semester data).
  RQ2  Explain the best model with SHAP (falls back to permutation importance
       if SHAP is not installed).
  RQ3  Fairness audit of the best T1 model by gender, age group and scholarship.

Dataset: "Predict Students' Dropout and Academic Success" (Realinho et al., 2022),
         data.csv (semicolon-separated), 4,424 rows x 37 columns.

Install (recommended):
    pip install pandas numpy scikit-learn scipy matplotlib xgboost lightgbm shap imbalanced-learn

Run:
    python dropout_prediction.py --data data.csv --out results

XGBoost, LightGBM, SHAP and imbalanced-learn are optional: if one is missing, the
script says so and skips it (XGBoost/LightGBM are replaced by scikit-learn's
HistGradientBoostingClassifier so there is still a boosting model).
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Optional libraries
# ----------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

SEED = 42
TARGET = "Target"

# Nominal (coded) variables -> one-hot encoded. Everything else is numeric/binary.
CATEGORICAL = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]


# ----------------------------------------------------------------------------
# 1. Data loading and preparation
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()  # one column name has a trailing tab
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing values: {df.isna().sum().sum()} | duplicate rows: {df.duplicated().sum()}")
    print("Target distribution:\n" + df[TARGET].value_counts().to_string())

    # Binary target: Dropout = 1, Enrolled/Graduate = 0
    df["dropout"] = (df[TARGET] == "Dropout").astype(int)
    print(f"Dropout rate: {df['dropout'].mean():.1%}\n")
    return df


def feature_sets(df):
    """T0 = enrolment-time features; T1 = T0 + 1st-semester performance.
    2nd-semester features are never used (too late for early intervention)."""
    sem1 = [c for c in df.columns if c.startswith("Curricular units 1st sem")]
    sem2 = [c for c in df.columns if c.startswith("Curricular units 2nd sem")]
    t0 = [c for c in df.columns if c not in sem1 + sem2 + [TARGET, "dropout"]]
    return {"T0": t0, "T1": t0 + sem1}


def make_preprocessor(features, scale):
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                  sparse_output=False), cat),
            ("num", num_step, num),
        ],
        verbose_feature_names_out=False,
    )


# ----------------------------------------------------------------------------
# 2. Models and hyperparameter grids
# ----------------------------------------------------------------------------
def model_specs(pos_weight):
    """Each model handles imbalance with class weights. Grids are kept small so the
    script runs in a few minutes; widen them for the final study."""
    specs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10]},
            True,
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [300], "clf__max_depth": [None, 10, 20],
             "clf__min_samples_leaf": [1, 5]},
            False,
        ),
    }
    if HAS_XGB:
        specs["XGBoost"] = (
            XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [300], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1]},
            False,
        )
    if HAS_LGBM:
        specs["LightGBM"] = (
            LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
            {"clf__n_estimators": [300], "clf__num_leaves": [15, 31],
             "clf__learning_rate": [0.05, 0.1]},
            False,
        )
    if not (HAS_XGB or HAS_LGBM):
        print("XGBoost/LightGBM not installed -> using HistGradientBoosting as the boosting model.")
        specs["HistGradientBoosting"] = (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            {"clf__max_depth": [3, 6], "clf__learning_rate": [0.05, 0.1]},
            False,
        )
    return specs


def build_pipeline(features, estimator, scale, use_smote=False):
    pre = make_preprocessor(features, scale)
    if use_smote:
        # SMOTE sits inside the pipeline, so it only ever touches training folds.
        est = clone(estimator)
        if "class_weight" in est.get_params():
            est.set_params(class_weight=None)
        if "scale_pos_weight" in est.get_params():
            est.set_params(scale_pos_weight=1)
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=SEED)), ("clf", est)])
    return Pipeline([("pre", pre), ("clf", estimator)])


# ----------------------------------------------------------------------------
# 3. Evaluation helpers
# ----------------------------------------------------------------------------
def test_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "Recall (dropout)": recall_score(y_true, pred),
        "Precision (dropout)": precision_score(y_true, pred, zero_division=0),
        "F1 (dropout)": f1_score(y_true, pred),
        "Macro-F1": f1_score(y_true, pred, average="macro"),
        "Balanced accuracy": balanced_accuracy_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
    }


def nemenyi_cd(k, n, alpha=0.05):
    """Critical difference for the Nemenyi post-hoc test (Demsar, 2006)."""
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}[k]
    return q * np.sqrt(k * (k + 1) / (6 * n))


def compare_models_statistically(fold_scores, label, out):
    """Friedman test over repeated-CV folds + Nemenyi critical difference."""
    names = list(fold_scores)
    mat = np.column_stack([fold_scores[m] for m in names])
    n, k = mat.shape
    lines = [f"Statistical comparison ({label}, F1 dropout, {n} CV folds)"]
    if k < 3:
        t, p = stats.wilcoxon(mat[:, 0], mat[:, 1])
        lines.append(f"Wilcoxon signed-rank: W={t:.2f}, p={p:.4f}")
    else:
        chi2, p = stats.friedmanchisquare(*mat.T)
        ranks = pd.DataFrame(mat, columns=names).rank(axis=1, ascending=False).mean()
        cd = nemenyi_cd(k, n)
        lines.append(f"Friedman chi2={chi2:.2f}, p={p:.4f}")
        lines.append("Mean rank (1 = best):\n" + ranks.sort_values().round(2).to_string())
        lines.append(f"Nemenyi critical difference (alpha=0.05): {cd:.2f} "
                     "-> models whose mean ranks differ by more than this differ significantly")
    text = "\n".join(lines)
    print(text + "\n")
    with open(os.path.join(out, f"stats_{label}.txt"), "w") as f:
        f.write(text + "\n")


# ----------------------------------------------------------------------------
# 4. Explainability (RQ2)
# ----------------------------------------------------------------------------
def original_feature(name, features):
    """Map a one-hot column (e.g. 'Course_9254') back to its source variable."""
    for f in sorted(features, key=len, reverse=True):
        if name == f or name.startswith(f + "_"):
            return f
    return name


def explain(pipe, X_train, X_test, y_test, features, label, out):
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    is_tree = not isinstance(clf, LogisticRegression)

    if HAS_SHAP:
        Xt = pd.DataFrame(pre.transform(X_test), columns=pre.get_feature_names_out())
        if is_tree:
            explainer = shap.TreeExplainer(clf)
        else:
            bg = pd.DataFrame(pre.transform(X_train.sample(200, random_state=SEED)),
                              columns=Xt.columns)
            explainer = shap.LinearExplainer(clf, bg)
        sv = explainer(Xt)
        if sv.values.ndim == 3:  # some tree models return one slice per class
            sv = sv[:, :, 1]

        plt.figure()
        shap.summary_plot(sv, Xt, max_display=15, show=False)
        plt.title(f"SHAP summary - {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"shap_summary_{label}.png"), dpi=150)
        plt.close()

        # Individual explanation: the student with the highest predicted risk
        idx = int(np.argmax(pipe.predict_proba(X_test)[:, 1]))
        plt.figure()
        shap.plots.waterfall(sv[idx], max_display=12, show=False)
        plt.title(f"Why student #{X_test.index[idx]} is flagged ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"shap_waterfall_{label}.png"), dpi=150)
        plt.close()

        imp = pd.Series(np.abs(sv.values).mean(axis=0), index=Xt.columns)
        imp = imp.groupby(lambda c: original_feature(c, features)).sum()
        method = "mean |SHAP| (one-hot columns summed)"
    else:
        print(f"SHAP not installed -> permutation importance for {label}.")
        r = permutation_importance(pipe, X_test, y_test, scoring="average_precision",
                                   n_repeats=10, random_state=SEED, n_jobs=-1)
        imp = pd.Series(r.importances_mean, index=X_test.columns)
        method = "permutation importance (drop in PR-AUC)"

    imp = imp.sort_values(ascending=False)
    imp.rename("importance").to_csv(os.path.join(out, f"feature_importance_{label}.csv"))

    top = imp.head(15)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values, color="#3b6ea5")
    plt.xlabel(method)
    plt.title(f"Top 15 predictors of dropout - {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"feature_importance_{label}.png"), dpi=150)
    plt.close()
    return imp


# ----------------------------------------------------------------------------
# 5. Fairness audit (RQ3)
# ----------------------------------------------------------------------------
def fairness_audit(X_test, y_test, prob, out, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    groups = {
        "Gender": X_test["Gender"].map({1: "Male", 0: "Female"}),
        "Age group": pd.cut(X_test["Age at enrollment"], [0, 20, 25, 200],
                            labels=["<=20", "21-25", ">25"]).astype(str),
        "Scholarship": X_test["Scholarship holder"].map({1: "Holder", 0: "Non-holder"}),
    }
    rows, gaps = [], []
    for attr, g in groups.items():
        part = []
        for level in sorted(g.unique()):
            m = (g == level).values
            yt, yp = y_test.values[m], pred[m]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            row = {
                "Attribute": attr, "Group": level, "n": int(m.sum()),
                "Actual dropout rate": yt.mean(),
                "Flagged rate": yp.mean(),
                "Recall / TPR": tp / (tp + fn) if tp + fn else np.nan,
                "FPR": fp / (fp + tn) if fp + tn else np.nan,
                "Precision": tp / (tp + fp) if tp + fp else np.nan,
            }
            rows.append(row)
            part.append(row)
        p = pd.DataFrame(part)
        gaps.append({
            "Attribute": attr,
            "Equal opportunity difference (max-min TPR)": p["Recall / TPR"].max() - p["Recall / TPR"].min(),
            "FPR difference (max-min)": p["FPR"].max() - p["FPR"].min(),
            "Demographic parity difference (max-min flagged rate)": p["Flagged rate"].max() - p["Flagged rate"].min(),
        })
    by_group = pd.DataFrame(rows)
    summary = pd.DataFrame(gaps)
    by_group.round(3).to_csv(os.path.join(out, "fairness_by_group.csv"), index=False)
    summary.round(3).to_csv(os.path.join(out, "fairness_summary.csv"), index=False)
    print("Fairness audit (best T1 model, test set):")
    print(by_group.round(3).to_string(index=False))
    print("\n" + summary.round(3).to_string(index=False) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.data)
    fsets = feature_sets(df)
    for k, v in fsets.items():
        print(f"{k}: {len(v)} features")

    # Hold-out test set (20%, stratified). The same split is used for T0 and T1.
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["dropout"],
                                           random_state=SEED)
    y_train, y_test = df.loc[train_idx, "dropout"], df.loc[test_idx, "dropout"]
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(train_idx):,} | Test: {len(test_idx):,}\n")

    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    compare_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    specs = model_specs(pos_weight)

    results, best = [], {}
    for label, features in fsets.items():
        X_train, X_test = df.loc[train_idx, features], df.loc[test_idx, features]
        fold_scores = {}
        fig, ax = plt.subplots(figsize=(7, 6))

        for name, (est, grid, scale) in specs.items():
            strategies = ["class weights"] + (["SMOTE"] if HAS_IMBLEARN else [])
            for strat in strategies:
                tag = name if strat == "class weights" else f"{name} + SMOTE"
                pipe = build_pipeline(features, est, scale, use_smote=(strat == "SMOTE"))

                # Hyperparameter tuning (5-fold CV on training data only)
                gs = GridSearchCV(pipe, grid, scoring="f1", cv=tune_cv, n_jobs=-1)
                gs.fit(X_train, y_train)

                # Repeated CV of the tuned model -> fold scores for the Friedman test
                scores = cross_val_score(gs.best_estimator_, X_train, y_train, scoring="f1",
                                         cv=compare_cv, n_jobs=-1)
                fold_scores[tag] = scores

                prob = gs.best_estimator_.predict_proba(X_test)[:, 1]
                row = {"Feature set": label, "Model": tag,
                       "CV F1 (mean)": scores.mean(), "CV F1 (sd)": scores.std(),
                       **test_metrics(y_test, prob),
                       "Best params": gs.best_params_}
                results.append(row)
                print(f"[{label}] {tag:32s} CV F1={scores.mean():.3f}  "
                      f"test F1={row['F1 (dropout)']:.3f}  recall={row['Recall (dropout)']:.3f}  "
                      f"PR-AUC={row['PR-AUC']:.3f}")
                PrecisionRecallDisplay.from_predictions(y_test, prob, name=tag, ax=ax)

                # Best model per feature set is chosen on CV score, never on the test set
                if label not in best or scores.mean() > best[label][1]:
                    best[label] = (gs.best_estimator_, scores.mean(), tag)

        ax.set_title(f"Precision-recall curves on test set - {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"pr_curves_{label}.png"), dpi=150)
        plt.close(fig)
        print()
        compare_models_statistically(fold_scores, label, args.out)

    res = pd.DataFrame(results)
    res.round(4).to_csv(os.path.join(args.out, "model_results.csv"), index=False)
    print("=== RQ1: model comparison (test set) ===")
    cols = ["Feature set", "Model", "CV F1 (mean)", "Recall (dropout)", "F1 (dropout)",
            "Macro-F1", "Balanced accuracy", "ROC-AUC", "PR-AUC"]
    print(res[cols].round(3).to_string(index=False))

    gain = (res[res["Feature set"] == "T1"].set_index("Model")["F1 (dropout)"]
            - res[res["Feature set"] == "T0"].set_index("Model")["F1 (dropout)"])
    print("\nF1 gain from adding 1st-semester data (T1 - T0):\n" + gain.round(3).to_string() + "\n")

    # RQ2: explain the best model at each prediction point
    importances = {}
    for label, (pipe, _, tag) in best.items():
        features = fsets[label]
        print(f"=== RQ2: explaining best {label} model ({tag}) ===")
        imp = explain(pipe, df.loc[train_idx, features], df.loc[test_idx, features],
                      y_test, features, label, args.out)
        importances[label] = imp
        print(imp.head(10).round(4).to_string() + "\n")
        y_prob = pipe.predict_proba(df.loc[test_idx, features])[:, 1]
        ConfusionMatrixDisplay.from_predictions(y_test, (y_prob >= 0.5).astype(int),
                                                display_labels=["Not dropout", "Dropout"],
                                                cmap="Blues")
        plt.title(f"Confusion matrix - best {label} model ({tag})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"confusion_matrix_{label}.png"), dpi=150)
        plt.close()

    comp = pd.DataFrame({"T0 rank": importances["T0"].rank(ascending=False),
                         "T1 rank": importances["T1"].rank(ascending=False)}).sort_values("T1 rank")
    comp.to_csv(os.path.join(args.out, "feature_rank_T0_vs_T1.csv"))

    # RQ3: fairness audit of the best T1 model
    print("=== RQ3: fairness audit ===")
    pipe, _, tag = best["T1"]
    X_test_full = df.loc[test_idx]
    prob = pipe.predict_proba(X_test_full[fsets["T1"]])[:, 1]
    fairness_audit(X_test_full, y_test, prob, args.out)

    print(f"All outputs saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
