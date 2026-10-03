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
WORKFLOW = "12_multiclass_and_multilabel_classification"


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


from sklearn.datasets import load_digits, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, hamming_loss, top_k_accuracy_score

digits = load_digits()
indices = np.arange(len(digits.target))
train_indices, test_indices = train_test_split(indices, stratify=digits.target, test_size=0.25, random_state=SEED)
multiclass = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
multiclass.fit(digits.data[train_indices], digits.target[train_indices])
probabilities = multiclass.predict_proba(digits.data[test_indices])
predictions = multiclass.predict(digits.data[test_indices])
report = pd.DataFrame(classification_report(digits.target[test_indices], predictions, output_dict=True, zero_division=0)).T.reset_index(names="class")
confusion = confusion_matrix(digits.target[test_indices], predictions)
uncertainty = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1)
review_queue = pd.DataFrame({"sample_id": test_indices, "actual": digits.target[test_indices], "predicted": predictions, "entropy": uncertainty, "confidence": probabilities.max(axis=1)}).sort_values("entropy", ascending=False)
X_multi, y_multi = make_multilabel_classification(n_samples=SAMPLE_SIZE, n_features=25, n_classes=5, n_labels=2, allow_unlabeled=False, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.25, random_state=SEED)
multilabel = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(max_iter=1000)))
multilabel.fit(X_train, y_train)
label_probabilities = multilabel.predict_proba(X_test)
label_predictions = (label_probabilities >= 0.5).astype(int)
per_label = []
for label_index in range(y_multi.shape[1]):
    per_label.append({"label": label_index, "prevalence": float(y_test[:, label_index].mean()), "f1": f1_score(y_test[:, label_index], label_predictions[:, label_index], zero_division=0), "predicted_positive": int(label_predictions[:, label_index].sum())})
label_metrics = pd.DataFrame(per_label)
cooccurrence = pd.DataFrame(y_train.T @ y_train, columns=[f"label_{i}" for i in range(5)]).reset_index(names="label")
save_table(report, "digit_classification_report")
save_table(review_queue, "digit_review_queue")
save_table(label_metrics, "multilabel_metrics")
save_table(cooccurrence, "label_cooccurrence")
save_table(pd.DataFrame(label_probabilities, columns=[f"label_{i}" for i in range(5)]), "multilabel_probabilities")


save_model(multiclass, "digit_classifier")
save_model(multilabel, "multilabel_classifier")
assert probabilities.shape == (len(test_indices), 10)
assert label_predictions.shape == y_test.shape
assert np.allclose(probabilities.sum(axis=1), 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(confusion, cmap="Blues")
axes[0].set(xlabel="Predicted digit", ylabel="Actual digit")
label_metrics.plot.bar(x="label", y="f1", ax=axes[1], legend=False)
save_figure(fig, "multitask_metrics")
result = finish({"digit_macro_f1": f1_score(digits.target[test_indices], predictions, average="macro"), "digit_top3_accuracy": top_k_accuracy_score(digits.target[test_indices], probabilities, k=3), "multilabel_micro_f1": f1_score(y_test, label_predictions, average="micro"), "multilabel_hamming_loss": hamming_loss(y_test, label_predictions)}, [report, label_metrics])
