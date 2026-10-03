import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4'}
if importlib.util.find_spec("packaging") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging>=23,<27"])
from packaging.requirements import Requirement

missing_packages = []
for module, requirement_text in DEPENDENCIES.items():
    requirement = Requirement(requirement_text)
    try:
        installed_version = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if importlib.util.find_spec(module) is None or installed_version is None or installed_version not in requirement.specifier:
        missing_packages.append(requirement_text)
if missing_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing_packages])
for thread_variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(thread_variable, "2")


SEED = 42
SAMPLE_SIZE = 1200
OUTPUT_ROOT = os.environ.get("FABRIC_STARTER_OUTPUT", "")
WORKFLOW = "10_rare_event_cost_sensitive_learning"


import json
import hashlib
import time
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from threadpoolctl import threadpool_limits

thread_limits = threadpool_limits(limits=2)
rng = np.random.default_rng(SEED)
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
lakehouse_files = Path("/lakehouse/default/Files")
output_base = Path(OUTPUT_ROOT).expanduser() if OUTPUT_ROOT else (lakehouse_files / "fabric_python_starters" if lakehouse_files.is_dir() else Path.cwd() / "fabric_python_outputs")
output_dir = output_base / WORKFLOW / run_id
output_dir.mkdir(parents=True, exist_ok=True)
working_context = tempfile.TemporaryDirectory(prefix="fabric_starter_")
working_dir = Path(working_context.name)
run_started = time.perf_counter()
artifact_records = []

def json_value(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    raise TypeError(type(value).__name__)

def record_artifact(path):
    path = Path(path)
    artifact_records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path

def save_table(frame, name):
    path = output_dir / (name + ".csv")
    frame.to_csv(path, index=False)
    record_artifact(path)
    return path

def save_json(payload, name):
    path = output_dir / (name + ".json")
    path.write_text(json.dumps(payload, indent=2, default=json_value, allow_nan=False), encoding="utf-8")
    record_artifact(path)
    return path

def save_model(model, name):
    path = output_dir / (name + ".joblib")
    joblib.dump(model, path)
    record_artifact(path)
    return path

def save_figure(figure, name):
    figure.tight_layout()
    path = output_dir / (name + ".png")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    record_artifact(path)
    plt.show()
    plt.close(figure)
    return path

def finish(metrics, tables=()):
    versions = {}
    for requirement in DEPENDENCIES.values():
        distribution = requirement.split(">=")[0]
        versions[distribution] = importlib.metadata.version(distribution)
    payload = {"workflow": WORKFLOW, "run_id": run_id, "seed": SEED, "python": sys.version.split()[0], "packages": versions, "storage": "lakehouse" if lakehouse_files in output_dir.parents else "local", "output_dir": str(output_dir), "elapsed_seconds": round(time.perf_counter() - run_started, 3), "metrics": metrics, "artifacts": list(artifact_records)}
    save_json(payload, "run_manifest")
    print(json.dumps(payload, indent=2, default=json_value, allow_nan=False))
    for table in tables:
        print(table.head(12).to_string(index=False))
    working_context.cleanup()
    return payload


from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

X, y = make_classification(n_samples=SAMPLE_SIZE * 3, n_features=14, n_informative=8, weights=[0.97, 0.03], flip_y=0.002, random_state=SEED)
X_develop, X_test, y_develop, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_develop, y_develop, stratify=y_develop, test_size=0.25, random_state=SEED)
models = {
    "balanced_logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "balanced_forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)
}
false_negative_cost = 250.0
false_positive_cost = 8.0
threshold_results = []
for name, candidate in models.items():
    candidate.fit(X_train, y_train)
    scores = candidate.predict_proba(X_valid)[:, 1]
    for threshold in np.linspace(0.01, 0.9, 90):
        decision = scores >= threshold
        false_negatives = int(((y_valid == 1) & ~decision).sum())
        false_positives = int(((y_valid == 0) & decision).sum())
        threshold_results.append({"model": name, "threshold": threshold, "cost": false_negatives * false_negative_cost + false_positives * false_positive_cost, "precision": precision_score(y_valid, decision, zero_division=0), "recall": recall_score(y_valid, decision, zero_division=0), "alerts": int(decision.sum())})
validation = pd.DataFrame(threshold_results)
best = validation.sort_values(["cost", "alerts"]).iloc[0]
model = models[best.model]
probability = model.predict_proba(X_test)[:, 1]
decision = probability >= best.threshold
test_predictions = pd.DataFrame({"actual": y_test, "probability": probability, "alert": decision})
test_predictions["cost"] = np.where((y_test == 1) & ~decision, false_negative_cost, np.where((y_test == 0) & decision, false_positive_cost, 0.0))
ranked = test_predictions.sort_values("probability", ascending=False).reset_index(drop=True)
ranked["rank"] = np.arange(1, len(ranked) + 1)


ranked["cumulative_positives"] = ranked.actual.cumsum()
ranked["recall_at_budget"] = ranked.cumulative_positives / max(y_test.sum(), 1)
ranked["precision_at_budget"] = ranked.cumulative_positives / ranked["rank"]
budgets = ranked.iloc[[min(n, len(ranked)) - 1 for n in [10, 25, 50, 100, 200]]]
save_table(validation, "validation_thresholds")
save_table(test_predictions, "test_predictions")
save_table(budgets, "budget_analysis")
save_model(model, "rare_event_model")
save_json({"threshold": float(best.threshold), "model": str(best.model), "false_negative_cost": false_negative_cost, "false_positive_cost": false_positive_cost}, "decision_policy")
assert np.isfinite(probability).all()
assert 0 < best.threshold < 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, group in validation.groupby("model"):
    axes[0].plot(group.threshold, group.cost, label=name)
axes[0].legend()
axes[0].set(xlabel="Threshold", ylabel="Validation cost")
axes[1].plot(ranked["rank"], ranked.recall_at_budget)
axes[1].set(xlabel="Review budget", ylabel="Recall")
save_figure(fig, "cost_and_budget")
result = finish({"average_precision": average_precision_score(y_test, probability), "recall": recall_score(y_test, decision), "precision": precision_score(y_test, decision, zero_division=0), "test_cost": float(test_predictions.cost.sum()), "threshold": float(best.threshold)}, [budgets])
