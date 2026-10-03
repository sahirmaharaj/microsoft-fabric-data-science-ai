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
WORKFLOW = "17_customer_segmentation_and_cluster_selection"


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


from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA

latent, truth = make_blobs(n_samples=SAMPLE_SIZE, centers=4, n_features=5, cluster_std=1.8, random_state=SEED)
customers = pd.DataFrame(np.exp(latent / 5 + 3), columns=["spend", "visits", "basket", "tenure", "engagement"])
scaler = StandardScaler()
features = scaler.fit_transform(np.log1p(customers))
records = []
models = {}
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=15, random_state=SEED)
    labels = model.fit_predict(features)
    records.append({"k": k, "silhouette": silhouette_score(features, labels, sample_size=min(800, len(features)), random_state=SEED), "davies_bouldin": davies_bouldin_score(features, labels), "inertia": model.inertia_})
    models[k] = model
selection = pd.DataFrame(records).sort_values("silhouette", ascending=False)
best_k = int(selection.iloc[0].k)
model = models[best_k]
labels = model.labels_
customers["cluster"] = labels
profiles = customers.groupby("cluster").agg(["mean", "median", "count"])
profiles.columns = ["_".join(column) for column in profiles.columns]
profiles = profiles.reset_index()
distances = model.transform(features)
customers["distance_to_center"] = distances.min(axis=1)
sorted_distances = np.sort(distances, axis=1)
customers["assignment_margin"] = sorted_distances[:, 1] - sorted_distances[:, 0]
stability_rows = []
for iteration in range(15):
    indices = rng.choice(len(features), int(0.8 * len(features)), replace=False)
    candidate = KMeans(n_clusters=best_k, n_init=10, random_state=iteration).fit(features[indices])
    stability_rows.append({"iteration": iteration, "adjusted_rand": adjusted_rand_score(labels, candidate.predict(features))})


stability = pd.DataFrame(stability_rows)
mixture = GaussianMixture(n_components=best_k, covariance_type="full", random_state=SEED).fit(features)
customers["mixture_confidence"] = mixture.predict_proba(features).max(axis=1)
projection = PCA(2, random_state=SEED).fit_transform(features)
save_table(customers, "customer_segments")
save_table(profiles, "segment_profiles")
save_table(selection, "cluster_selection")
save_table(stability, "cluster_stability")
save_model({"scaler": scaler, "model": model, "mixture": mixture}, "segmentation_bundle")
assert len(np.unique(labels)) == best_k
assert np.isfinite(distances).all()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(projection[:, 0], projection[:, 1], c=labels, s=12, cmap="tab10")
selection.sort_values("k").plot(x="k", y="silhouette", ax=axes[1], legend=False)
save_figure(fig, "segments")
result = finish({"clusters": best_k, "silhouette": float(selection.iloc[0].silhouette), "mean_stability": float(stability.adjusted_rand.mean()), "synthetic_ari": adjusted_rand_score(truth, labels)}, [selection, profiles])
