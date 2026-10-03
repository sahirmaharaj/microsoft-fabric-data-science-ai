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
WORKFLOW = "18_multivariate_anomaly_detection"


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


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score

normal = rng.multivariate_normal(np.zeros(6), 0.3 * np.ones((6, 6)) + 0.7 * np.eye(6), SAMPLE_SIZE)
abnormal = rng.normal(3.0, 1.8, (90, 6))
fit, calibration = normal[:700], normal[700:900]
test = np.vstack([normal[900:], abnormal])
y_test = np.r_[np.zeros(len(normal) - 900), np.ones(len(abnormal))].astype(int)
scaler = StandardScaler().fit(fit)
fit_scaled = scaler.transform(fit)
cal_scaled = scaler.transform(calibration)
test_scaled = scaler.transform(test)
models = {
    "isolation_forest": IsolationForest(n_estimators=160, contamination="auto", random_state=SEED, n_jobs=2),
    "local_outlier_factor": LocalOutlierFactor(n_neighbors=25, novelty=True),
    "robust_covariance": EllipticEnvelope(contamination=0.02, random_state=SEED)
}
output = pd.DataFrame({"actual_anomaly": y_test})
metrics = []
calibration_scores = {}
for name, model in models.items():
    model.fit(fit_scaled)
    cal_score = -model.score_samples(cal_scaled)
    scores = -model.score_samples(test_scaled)
    threshold = float(np.quantile(cal_score, 0.98, method="higher"))
    decisions = scores > threshold
    empirical_tail = (1 + (cal_score[:, None] >= scores[None, :]).sum(axis=0)) / (len(cal_score) + 1)
    output[name + "_score"] = scores
    output[name + "_pvalue"] = empirical_tail
    output[name + "_alert"] = decisions
    metrics.append({"model": name, "average_precision": average_precision_score(y_test, scores), "precision": precision_score(y_test, decisions, zero_division=0), "recall": recall_score(y_test, decisions), "threshold": threshold, "alerts": int(decisions.sum())})
    calibration_scores[name] = cal_score


pvalue_columns = [column for column in output if column.endswith("_pvalue")]
output["ensemble_score"] = -np.log(np.clip(output[pvalue_columns], 1e-8, 1)).mean(axis=1)
output["consensus_alert"] = output[[column for column in output if column.endswith("_alert")]].sum(axis=1) >= 2
feature_deviation = pd.DataFrame(np.abs(test_scaled), columns=[f"deviation_feature_{i}" for i in range(test.shape[1])])
output["largest_deviation_feature"] = feature_deviation.idxmax(axis=1)
output["largest_deviation"] = feature_deviation.max(axis=1)
metrics = pd.DataFrame(metrics)
save_table(output.sort_values("ensemble_score", ascending=False), "anomaly_queue")
save_table(metrics, "detector_metrics")
save_model({"scaler": scaler, "models": models, "calibration_scores": calibration_scores}, "anomaly_bundle")
assert output[pvalue_columns].gt(0).all().all()
assert output[pvalue_columns].le(1).all().all()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(test_scaled[:, 0], test_scaled[:, 1], c=output.ensemble_score, s=15, cmap="magma")
metrics.plot.bar(x="model", y=["precision", "recall"], ax=axes[1])
save_figure(fig, "anomaly_detection")
result = finish({"ensemble_ap": average_precision_score(y_test, output.ensemble_score), "consensus_alerts": int(output.consensus_alert.sum()), "test_anomalies": int(y_test.sum())}, [metrics, output.sort_values("ensemble_score", ascending=False)])
