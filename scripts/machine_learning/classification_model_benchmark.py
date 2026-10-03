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
WORKFLOW = "09_classification_model_benchmark"


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

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=SEED)),
    "random_forest": RandomForestClassifier(n_estimators=140, min_samples_leaf=3, n_jobs=2, random_state=SEED),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=SEED)
}
folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
records = []
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=folds, scoring={"auc": "roc_auc", "ap": "average_precision", "f1": "f1"}, n_jobs=2)
    records.append({"model": name, "cv_auc": scores["test_auc"].mean(), "cv_auc_std": scores["test_auc"].std(), "cv_ap": scores["test_ap"].mean(), "cv_f1": scores["test_f1"].mean(), "fit_seconds": scores["fit_time"].mean()})
leaderboard = pd.DataFrame(records).sort_values("cv_auc", ascending=False)
selected = str(leaderboard.iloc[0].model)
model = models[selected].fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
prediction = probability >= 0.5
predictions = pd.DataFrame({"row_id": X_test.index, "actual": y_test.to_numpy(), "probability": probability, "prediction": prediction.astype(int)})
bootstrap_auc = []
for _ in range(300):
    positions = rng.integers(0, len(y_test), len(y_test))
    actual = y_test.to_numpy()[positions]
    if len(np.unique(actual)) == 2:
        bootstrap_auc.append(roc_auc_score(actual, probability[positions]))


interval = np.quantile(bootstrap_auc, [0.025, 0.975])
metrics = {"selected_model": selected, "test_auc": roc_auc_score(y_test, probability), "test_ap": average_precision_score(y_test, probability), "test_f1": f1_score(y_test, prediction), "auc_low": interval[0], "auc_high": interval[1]}
confusion = pd.DataFrame(confusion_matrix(y_test, prediction), columns=["predicted_0", "predicted_1"]).reset_index(names="actual")
model_path = save_model(model, "classifier")
assert np.allclose(joblib.load(model_path).predict_proba(X_test)[:, 1], probability)
assert set(X_train.index).isdisjoint(X_test.index)
save_table(leaderboard, "cross_validation")
save_table(predictions, "test_predictions")
save_table(confusion, "confusion_matrix")
save_json({"features": list(X.columns), "dtypes": X.dtypes.astype(str).to_dict(), "classes": [0, 1]}, "input_schema")
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
RocCurveDisplay.from_predictions(y_test, probability, ax=axes[0])
PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[1])
save_figure(fig, "classification_curves")
result = finish(metrics, [leaderboard, confusion])
