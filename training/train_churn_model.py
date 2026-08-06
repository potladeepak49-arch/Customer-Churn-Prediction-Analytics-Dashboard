"""
Trains a churn model on the Telco dataset and spits out a risk score per
customer. Tries LR / RF / XGBoost with GridSearchCV, picks whichever does
best on recall+precision, then sweeps thresholds instead of just going with
the default 0.5 cutoff.

    python train_churn_model.py --data Telco-Customer-Churn.csv --out risk_scores.csv
    python train_churn_model.py --target_recall 0.8   # tune for capacity instead of F1
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from xgboost import XGBClassifier

# features + tenure/charges we use for the plain-english "why is this
# customer risky" blurb in the export. not exhaustive, just the ones that
# actually move the needle on this dataset.
RISK_DRIVER_FEATURES = [
    "Contract_Month-to-month", "tenure", "MonthlyCharges",
    "InternetService_Fiber optic", "TechSupport_No", "OnlineSecurity_No",
    "PaymentMethod_Electronic check", "PaperlessBilling_Yes"
]
DRIVER_LABELS = {
    "Contract_Month-to-month": "Month-to-month contract",
    "tenure": "Short tenure",
    "MonthlyCharges": "High monthly charges",
    "InternetService_Fiber optic": "Fiber optic service",
    "TechSupport_No": "No tech support add-on",
    "OnlineSecurity_No": "No online security add-on",
    "PaymentMethod_Electronic check": "Pays by electronic check",
    "PaperlessBilling_Yes": "Paperless billing",
}


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # a handful of TotalCharges rows are blank strings instead of numbers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def build_features(df: pd.DataFrame):
    ids = df["customerID"]
    y = df["Churn"]
    X_raw = df.drop(columns=["customerID", "Churn"])

    cat_cols = [c for c in X_raw.columns if X_raw[c].dtype == object or pd.api.types.is_string_dtype(X_raw[c])]
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)

    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[num_cols] = scaler.fit_transform(X_encoded[num_cols])

    return ids, X_encoded, X_scaled, y


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, verbose=True):
    results = {}
    models = {}
    best_params = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_grid = {
        "C": [0.01, 0.1, 1, 10],
    }
    lr_search = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        lr_grid, scoring="f1", cv=cv, n_jobs=-1
    )
    lr_search.fit(X_train_scaled, y_train)
    lr = lr_search.best_estimator_
    results["Logistic Regression"] = evaluate(y_test, lr.predict(X_test_scaled), lr.predict_proba(X_test_scaled)[:, 1])
    models["Logistic Regression"] = lr
    best_params["Logistic Regression"] = lr_search.best_params_

    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [1, 3],
    }
    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_grid, scoring="f1", cv=cv, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    results["Random Forest"] = evaluate(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])
    models["Random Forest"] = rf
    best_params["Random Forest"] = rf_search.best_params_

    # --- XGBoost ---
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    xgb_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
    }
    xgb_search = GridSearchCV(
        XGBClassifier(scale_pos_weight=n_neg / n_pos, eval_metric="logloss", random_state=42),
        xgb_grid, scoring="f1", cv=cv, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    xgb = xgb_search.best_estimator_
    results["XGBoost"] = evaluate(y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])
    models["XGBoost"] = xgb
    best_params["XGBoost"] = xgb_search.best_params_

    if verbose:
        print("\n=== Best hyperparameters found (5-fold CV, scored on F1) ===")
        for name, params in best_params.items():
            print(f"{name:22s} | {params}")

    return results, models, best_params


# by default .predict() just uses a 0.5 cutoff, which nobody actually chose
# on purpose. sweep a range of thresholds so precision/recall tradeoff is
# visible and you can pick one deliberately.
def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            precision = 1.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "flagged": int(y_pred.sum()),
            "flagged_pct": round(y_pred.mean(), 4),
        })
    return pd.DataFrame(rows)


def best_f1_threshold(sweep: pd.DataFrame) -> float:
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"])


def threshold_for_target_recall(sweep: pd.DataFrame, target_recall: float) -> float:
    """Among thresholds that still hit the target recall, pick the highest one
    (highest threshold -> best precision) that clears the bar."""
    qualifying = sweep[sweep["recall"] >= target_recall]
    if qualifying.empty:
        # Target recall unreachable at any threshold in the sweep -> fall back to the
        # threshold that gets closest, favoring recall.
        return float(sweep.sort_values("recall", ascending=False).iloc[0]["threshold"])
    return float(qualifying.sort_values("threshold", ascending=False).iloc[0]["threshold"])


def select_best(results: dict) -> str:
    # recall + precision combined, i.e. catch churners without too many false alarms
    return max(results, key=lambda m: results[m]["recall"] + results[m]["precision"])


def top_drivers_for_customer(row_encoded: pd.Series, importances: pd.Series, k: int = 3) -> str:
    scores = {}
    for feat in RISK_DRIVER_FEATURES:
        if feat not in row_encoded.index:
            continue
        val = row_encoded[feat]
        weight = importances.get(feat, 0)
        if feat == "tenure":
            risk_signal = -val  # low tenure = high risk
        elif feat == "MonthlyCharges":
            risk_signal = val   # high charges = high risk
        else:
            risk_signal = val   # 1 = flag present = risk
        scores[feat] = risk_signal * weight

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return "; ".join(DRIVER_LABELS[f] for f, s in top if s > 0) or "No dominant single driver"


def risk_tier(score: float, threshold: float) -> str:
    """Tiers are anchored on the optimized decision threshold rather than a
    fixed 0.7/0.4 split, so 'High' always means 'the model says act on this'."""
    if score >= threshold:
        return "High"
    elif score >= threshold * 0.6:
        return "Medium"
    return "Low"


def export_risk_scores(ids, X_encoded, best_model, threshold, out_path):
    probs = best_model.predict_proba(X_encoded)[:, 1]

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X_encoded.columns)
    else:
        importances = pd.Series(np.abs(best_model.coef_[0]), index=X_encoded.columns)

    drivers = [top_drivers_for_customer(X_encoded.iloc[i], importances) for i in range(len(X_encoded))]

    out = pd.DataFrame({
        "customer_id": ids.values,
        "risk_score": probs.round(4),
        "risk_tier": [risk_tier(p, threshold) for p in probs],
        "flag_for_outreach": ["Yes" if p >= threshold else "No" for p in probs],
        "top_drivers": drivers,
    }).sort_values("risk_score", ascending=False)

    out.to_csv(out_path, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train churn model and export risk scores.")
    parser.add_argument("--data", type=str, default="Telco-Customer-Churn.csv")
    parser.add_argument("--out", type=str, default="risk_scores.csv")
    parser.add_argument("--metrics_out", type=str, default="model_metrics.json")
    parser.add_argument("--target_recall", type=float, default=None,
                         help="Optional: pick the threshold that best achieves this recall "
                              "(e.g. 0.8 to catch 80%% of churners). Default optimizes F1.")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    ids, X_encoded, X_scaled, y = build_features(df)

    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, ids_train, ids_test) = train_test_split(
        X_encoded, X_scaled, y, ids, test_size=0.2, random_state=42, stratify=y
    )

    results, models, best_params = train_and_evaluate(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    print("\n=== Model Comparison (test set, tuned via GridSearchCV) ===")
    for name, m in results.items():
        print(f"{name:22s} | acc={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}")

    best_name = select_best(results)
    best_model = models[best_name]
    print(f"\nSelected model: {best_name} (best combined recall + precision)")
    print(f"Best params: {best_params[best_name]}")
    print(f"Confusion matrix at default 0.5 threshold [[TN, FP], [FN, TP]]: {results[best_name]['confusion_matrix']}")

    # --- Threshold optimization ---
    test_probs = best_model.predict_proba(X_test)[:, 1] if best_name != "Logistic Regression" \
        else best_model.predict_proba(X_test_scaled)[:, 1]
    sweep = threshold_sweep(y_test, test_probs)

    if args.target_recall is not None:
        chosen_threshold = threshold_for_target_recall(sweep, args.target_recall)
        strategy = f"best precision at >= {args.target_recall:.0%} recall"
    else:
        chosen_threshold = best_f1_threshold(sweep)
        strategy = "best F1"

    print(f"\n=== Threshold tradeoff (test set, {best_name}) ===")
    print(f"{'threshold':>9} {'precision':>10} {'recall':>8} {'f1':>6} {'flagged':>8} {'flagged %':>10}")
    show = sorted(set([0.3, 0.4, 0.5, round(chosen_threshold, 2)]))
    for t in show:
        row = sweep.iloc[(sweep["threshold"] - t).abs().idxmin()]
        marker = "  <- selected" if abs(row["threshold"] - chosen_threshold) < 0.005 else ""
        print(f"{row['threshold']:>9.2f} {row['precision']:>10.3f} {row['recall']:>8.3f} "
              f"{row['f1']:>6.3f} {row['flagged']:>8.0f} {row['flagged_pct']:>9.1%}{marker}")

    chosen_row = sweep.iloc[(sweep["threshold"] - chosen_threshold).abs().idxmin()]
    print(f"\nSelected threshold: {chosen_threshold:.2f} (optimizing {strategy})")
    print(f"At this threshold: precision={chosen_row['precision']:.3f}, recall={chosen_row['recall']:.3f}, "
          f"flags {chosen_row['flagged']:.0f} of {len(y_test)} test customers ({chosen_row['flagged_pct']:.1%})")

    with open(args.metrics_out, "w") as f:
        json.dump({
            "results": results,
            "selected_model": best_name,
            "best_params": best_params,
            "threshold_optimization": {
                "strategy": strategy,
                "chosen_threshold": round(chosen_threshold, 2),
                "metrics_at_threshold": chosen_row.to_dict(),
                "default_threshold_metrics": results[best_name],
            },
        }, f, indent=2)

    # Score the FULL customer base (not just the test set) for the dashboard export
    out_df = export_risk_scores(ids, X_encoded, best_model, chosen_threshold, args.out)

    print(f"\nExported {len(out_df)} customer risk scores -> {args.out}  (model: {best_name}, threshold: {chosen_threshold:.2f})")
    print(out_df["risk_tier"].value_counts())
    print(f"Flagged for outreach: {(out_df['flag_for_outreach']=='Yes').sum()} customers")
    print(f"\nMetrics saved -> {args.metrics_out}")


if __name__ == "__main__":
    main()
