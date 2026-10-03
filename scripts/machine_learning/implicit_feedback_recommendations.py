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
WORKFLOW = "19_implicit_feedback_recommendations"


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


from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

user_count, item_count, latent_dimension = 140, 100, 8
user_taste = rng.normal(size=(user_count, latent_dimension))
item_taste = rng.normal(size=(item_count, latent_dimension))
affinity = user_taste @ item_taste.T
interactions = []
held_out = {}
for user in range(user_count):
    preference = np.exp((affinity[user] - affinity[user].max()) / 2)
    preference /= preference.sum()
    items = rng.choice(item_count, size=20, replace=False, p=preference)
    held_out[user] = set(items[-4:])
    for item in items[:-4]:
        interactions.append({"user_id": user, "item_id": int(item), "strength": int(rng.integers(1, 6))})
interactions = pd.DataFrame(interactions)
matrix = csr_matrix((interactions.strength, (interactions.user_id, interactions.item_id)), shape=(user_count, item_count))
weighted = matrix.copy().astype(float)
weighted.data = np.log1p(weighted.data)
factorizer = TruncatedSVD(n_components=16, n_iter=10, random_state=SEED)
user_factors = factorizer.fit_transform(weighted)
item_factors = factorizer.components_.T
scores = user_factors @ item_factors.T
seen = matrix.toarray() > 0
scores[seen] = -np.inf
item_similarity = cosine_similarity(item_factors)
recommendations = []
evaluation = []
for user in range(user_count):
    candidates = list(np.argsort(scores[user])[::-1][:30])
    selected = []
    while candidates and len(selected) < 10:
        best = max(candidates, key=lambda item: scores[user, item] - 0.25 * max([item_similarity[item, other] for other in selected] or [0]))
        selected.append(best)
        candidates.remove(best)
    relevant = np.array([item in held_out[user] for item in selected])
    discounts = 1 / np.log2(np.arange(2, len(selected) + 2))
    ideal = discounts[:min(len(held_out[user]), len(selected))].sum()
    evaluation.append({"user_id": user, "recall_at_10": relevant.sum() / len(held_out[user]), "ndcg_at_10": float((relevant * discounts).sum() / ideal), "precision_at_10": relevant.mean()})
    for rank, item in enumerate(selected, 1):
        recommendations.append({"user_id": user, "item_id": item, "rank": rank, "score": float(scores[user, item]), "held_out_relevant": item in held_out[user]})


recommendations = pd.DataFrame(recommendations)
evaluation = pd.DataFrame(evaluation)
coverage = recommendations.item_id.nunique() / item_count
popularity = interactions.groupby("item_id").size().reindex(range(item_count), fill_value=0)
recommendations["novelty"] = -np.log2((recommendations.item_id.map(popularity) + 1) / (user_count + 1))
assert not any(seen[row.user_id, row.item_id] for row in recommendations.itertuples())
assert recommendations.groupby("user_id").size().eq(10).all()
save_table(recommendations, "recommendations")
save_table(evaluation, "ranking_metrics")
save_table(interactions, "training_interactions")
save_model({"factorizer": factorizer, "item_factors": item_factors, "user_factors": user_factors, "seen": seen}, "recommender")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(evaluation.recall_at_10, bins=10)
recommendations.item_id.value_counts().head(15).plot.bar(ax=axes[1])
save_figure(fig, "recommendation_metrics")
result = finish({"recall_at_10": float(evaluation.recall_at_10.mean()), "ndcg_at_10": float(evaluation.ndcg_at_10.mean()), "catalog_coverage": coverage, "mean_novelty": float(recommendations.novelty.mean())}, [evaluation, recommendations])
