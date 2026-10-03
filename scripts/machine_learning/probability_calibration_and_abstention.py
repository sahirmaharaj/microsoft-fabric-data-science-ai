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
WORKFLOW = "11_probability_calibration_and_abstention"


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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, confusion_matrix

X_array, y_array = make_classification(n_samples=SAMPLE_SIZE, n_features=16, n_informative=9, n_redundant=3, weights=[0.72, 0.28], class_sep=0.9, random_state=SEED)
X = pd.DataFrame(X_array, columns=[f"feature_{i:02d}" for i in range(16)])
y = pd.Series(y_array, name="target")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

X_fit, X_policy, y_fit, y_policy = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=SEED)
base = RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=3, random_state=SEED, n_jobs=2)
calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=4)
calibrated.fit(X_fit, y_fit)
base.fit(X_fit, y_fit)
policy_probability = calibrated.predict_proba(X_policy)[:, 1]
policy_rows = []
for confidence in np.linspace(0.5, 0.95, 19):
    accepted = np.maximum(policy_probability, 1 - policy_probability) >= confidence
    accuracy = float(((policy_probability[accepted] >= 0.5) == y_policy.to_numpy()[accepted]).mean()) if accepted.any() else 0.0
    policy_rows.append({"confidence": confidence, "coverage": accepted.mean(), "accuracy": accuracy, "accepted": int(accepted.sum())})
policy = pd.DataFrame(policy_rows)
feasible = policy.loc[(policy.accuracy >= 0.9) & (policy.accepted >= 20)]
selected = feasible.sort_values("coverage", ascending=False).iloc[0] if len(feasible) else policy.iloc[0]
raw_probability = base.predict_proba(X_test)[:, 1]
probability = calibrated.predict_proba(X_test)[:, 1]
accepted = np.maximum(probability, 1 - probability) >= selected.confidence
predictions = pd.DataFrame({"actual": y_test.to_numpy(), "raw_probability": raw_probability, "calibrated_probability": probability, "accepted": accepted})
predictions["decision"] = np.where(accepted, np.where(probability >= 0.5, "positive", "negative"), "review")


reliability = []
for name, scores in [("raw", raw_probability), ("calibrated", probability)]:
    actual_rate, mean_probability = calibration_curve(y_test, scores, n_bins=8, strategy="quantile")
    reliability.append(pd.DataFrame({"model": name, "mean_probability": mean_probability, "actual_rate": actual_rate}))
reliability = pd.concat(reliability, ignore_index=True)
metrics = {"raw_brier": brier_score_loss(y_test, raw_probability), "calibrated_brier": brier_score_loss(y_test, probability), "log_loss": log_loss(y_test, probability), "test_coverage": float(accepted.mean()), "confidence": float(selected.confidence), "accepted_accuracy": float(((probability[accepted] >= 0.5) == y_test.to_numpy()[accepted]).mean()) if accepted.any() else 0.0}
save_table(policy, "validation_policy")
save_table(reliability, "reliability")
save_table(predictions, "decisions")
save_model(calibrated, "calibrated_classifier")
save_json({"confidence": float(selected.confidence), "min_validation_accuracy": 0.9, "constraint_met": bool(len(feasible))}, "abstention_policy")
assert predictions.decision.isin(["positive", "negative", "review"]).all()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, group in reliability.groupby("model"):
    axes[0].plot(group.mean_probability, group.actual_rate, marker="o", label=name)
axes[0].plot([0, 1], [0, 1], linestyle="--")
axes[0].legend()
axes[1].plot(policy.coverage, policy.accuracy, marker="o")
axes[1].set(xlabel="Coverage", ylabel="Selective accuracy")
save_figure(fig, "calibration_and_coverage")
result = finish(metrics, [policy, predictions])
