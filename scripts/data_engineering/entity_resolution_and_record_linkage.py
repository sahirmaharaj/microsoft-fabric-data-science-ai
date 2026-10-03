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
WORKFLOW = "06_entity_resolution_and_record_linkage"


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


import re
import unicodedata
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

first_names = ["Amina", "Sahir", "Lebo", "Thandi", "Priya", "Jonas", "Fatima", "Daniel"]
last_names = ["Maharaj", "Nkosi", "Naidoo", "Jacobs", "Botha", "Patel", "Mokoena", "Adams"]
left = pd.DataFrame([{"record_id": i, "name": f"{first} {last}", "city": ["Cape Town", "Durban", "Pretoria"][i % 3], "phone": f"+27 82 {1000000+i}"} for i, (first, last) in enumerate((a, b) for a in first_names for b in last_names)])
right = left.copy()
right["record_id"] += 1000
right["name"] = right.name.str.upper()
right.loc[right.index % 4 == 0, "name"] = right.loc[right.index % 4 == 0, "name"].str.replace("a", "", case=False)
right["phone"] = right.phone.str.replace(" ", "", regex=False)
right.loc[right.index % 7 == 0, "phone"] = ""
right = right.sample(frac=1, random_state=SEED).reset_index(drop=True)

def normalize(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()

for frame in [left, right]:
    frame["normalized_name"] = frame.name.map(normalize)
    frame["normalized_phone"] = frame.phone.str.replace(r"\D", "", regex=True)
    frame["normalized_city"] = frame.city.map(normalize)
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
vectors = vectorizer.fit_transform(pd.concat([left.normalized_name, right.normalized_name]))
name_scores = cosine_similarity(vectors[:len(left)], vectors[len(left):])
score_matrix = np.zeros_like(name_scores)
candidates = []
for i, a in left.iterrows():
    for j, b in right.iterrows():
        same_city = a.normalized_city == b.normalized_city
        same_phone = bool(a.normalized_phone) and a.normalized_phone == b.normalized_phone
        if not same_city and not same_phone:
            continue
        edit_score = SequenceMatcher(None, a.normalized_name, b.normalized_name).ratio()
        score = 0.45 * name_scores[i, j] + 0.25 * edit_score + 0.2 * same_phone + 0.1 * same_city
        score_matrix[i, j] = score
        candidates.append({"left_id": a.record_id, "right_id": b.record_id, "score": score, "name_similarity": name_scores[i, j], "phone_match": same_phone})


left_indices, right_indices = linear_sum_assignment(-score_matrix)
links = []
for i, j in zip(left_indices, right_indices):
    alternatives = np.delete(score_matrix[i], j)
    margin = score_matrix[i, j] - alternatives.max()
    score = float(score_matrix[i, j])
    links.append({"left_id": int(left.iloc[i].record_id), "right_id": int(right.iloc[j].record_id), "left_name": left.iloc[i]["name"], "right_name": right.iloc[j]["name"], "score": score, "margin": margin, "decision": "match" if score >= 0.65 and margin > 0.04 else "review"})
links = pd.DataFrame(links)
links["correct"] = links.right_id.eq(links.left_id + 1000)
accepted = links.loc[links.decision.eq("match")]
assert links.left_id.is_unique and links.right_id.is_unique
save_table(pd.DataFrame(candidates), "candidates")
save_table(links, "resolved_links")
save_table(links.loc[links.decision.eq("review")], "manual_review")
save_model(vectorizer, "name_vectorizer")
fig, ax = plt.subplots(figsize=(8, 4))
for decision, group in links.groupby("decision"):
    ax.scatter(group.score, group.margin, label=decision)
ax.legend()
ax.set(xlabel="Match score", ylabel="Alternative margin")
save_figure(fig, "match_confidence")
result = finish({"records": len(left), "candidate_pairs": len(candidates), "accepted": len(accepted), "accepted_precision": float(accepted.correct.mean()) if len(accepted) else 0.0, "assignment_accuracy": float(links.correct.mean())}, [links])
