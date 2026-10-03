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
WORKFLOW = "35_near_duplicate_detection_and_canonicalization"


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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOPICS = {
    "billing": ["invoice payment refund subscription charge account", "billing receipt credit debit renewal balance", "refund invoice disputed charge payment overdue", "subscription price receipt renewal credit payment"],
    "technical": ["server network timeout connection error database", "application crashes login password authentication access", "network latency connection server performance error", "database query timeout storage memory troubleshooting"],
    "delivery": ["parcel courier shipment delivery tracking warehouse", "package address delivery shipment delayed tracking", "courier damaged parcel replacement warehouse shipping", "shipment dispatch address tracking arrival package"],
    "analytics": ["dashboard metric report dataset visualization insights", "lakehouse pipeline data transformation analytics warehouse", "report refresh semantic model dashboard measure", "dataset governance quality lineage analytics pipeline"],
    "security": ["security access identity encryption permission audit", "credential threat malware authentication incident policy", "privacy protection encryption compliance access security", "audit permission identity threat incident credential"],
    "machine_learning": ["model training prediction features validation accuracy", "classification regression experiment inference model evaluation", "neural learning optimization embedding prediction training", "feature selection validation experiment drift inference"]
}
modifiers = ["urgent", "please investigate", "customer reported", "team reviewing", "follow up", "new request", "weekly review", "needs attention"]
records = []
for topic, templates in TOPICS.items():
    pool = sorted(set(" ".join(templates).split()))
    for index in range(70):
        words = rng.choice(pool, size=int(rng.integers(5, 10)), replace=False)
        records.append({"document_id": f"{topic}_{index:03d}", "text": " ".join(words) + ". " + modifiers[index % len(modifiers)] + f" case {index + 1000}.", "topic": topic})
corpus = pd.DataFrame(records)

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

original=corpus.iloc[:120].copy().reset_index(drop=True)
variants=original.iloc[:35].copy()
variants["document_id"]=variants.document_id+"_copy"
variants["text"]=variants.text.str.replace("urgent","urgent update",regex=False).str.replace("case","ticket",regex=False)
data=pd.concat([original,variants],ignore_index=True)
data["normalized"]=data.text.str.lower().str.replace(r"[^a-z ]"," ",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
data["exact_hash"]=data.normalized.map(lambda value:hashlib.sha256(value.encode()).hexdigest())
vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2)
X=vectorizer.fit_transform(data.normalized)
similarity=cosine_similarity(X)
threshold=0.93


row,column=np.where(np.triu(similarity,k=1)>=threshold)
pairs=pd.DataFrame({"left_index":row,"right_index":column,"similarity":similarity[row,column]})
graph=coo_matrix((np.ones(len(row)*2),(np.r_[row,column],np.r_[column,row])),shape=(len(data),len(data)))
component_count,labels=connected_components(graph,directed=False)
data["component"]=labels
representatives=[]
canonical_map={}
for component,frame in data.groupby("component"):
    positions=frame.index.to_numpy()
    centrality=similarity[np.ix_(positions,positions)].mean(axis=1)
    representative=int(positions[np.argmax(centrality)])
    for position in positions:
        canonical_map[position]=representative
    representatives.append({"component":component,"canonical_document_id":data.loc[representative,"document_id"],"members":len(frame),"minimum_internal_similarity":float(similarity[np.ix_(positions,positions)].min()),"requires_chain_review":bool(similarity[np.ix_(positions,positions)].min()<threshold)})
data["canonical_document_id"]=[data.loc[canonical_map[index],"document_id"] for index in data.index]
data["is_canonical"]=data.document_id.eq(data.canonical_document_id)
components=pd.DataFrame(representatives)
pairs["left_id"]=data.iloc[row].document_id.to_numpy()
pairs["right_id"]=data.iloc[column].document_id.to_numpy()
true_duplicate_retrieval=[]
for index in range(35):
    true_duplicate_retrieval.append(labels[index]==labels[len(original)+index])
save_table(data,"canonicalized_documents")
save_table(pairs,"candidate_duplicate_pairs")
save_table(components,"duplicate_components")
save_table(data.loc[data.is_canonical],"deduplicated_corpus")
save_model(vectorizer,"duplicate_vectorizer")
assert data.loc[data.is_canonical].component.is_unique
assert data.canonical_document_id.isin(data.document_id).all()
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(pairs.similarity,bins=20)
components.members.value_counts().sort_index().plot.bar(ax=axes[1])
save_figure(fig,"duplicate_analysis")
result=finish({"input_documents":len(data),"canonical_documents":int(data.is_canonical.sum()),"duplicate_pair_candidates":len(pairs),"synthetic_duplicate_recall":float(np.mean(true_duplicate_retrieval)),"chain_review_components":int(components.requires_chain_review.sum())},[components,pairs])
