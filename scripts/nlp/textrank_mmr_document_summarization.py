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
WORKFLOW = "37_textrank_mmr_document_summarization"


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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

knowledge = [
    {"id":"policy_01","title":"Refund policy","text":"Refund requests must be submitted within 30 days of purchase. Approved refunds are returned to the original payment method. Refund processing takes five business days."},
    {"id":"policy_02","title":"Account security","text":"All administrator accounts require multifactor authentication. Passwords must contain at least twelve characters. Access permissions are reviewed every quarter."},
    {"id":"policy_03","title":"Data retention","text":"Operational logs are retained for 90 days. Financial transaction records are retained for seven years. Data deletion requests are reviewed by the privacy team."},
    {"id":"policy_04","title":"Shipping","text":"Standard delivery takes three to five business days. Express delivery takes one business day. Customers receive tracking information when a parcel leaves the warehouse."},
    {"id":"policy_05","title":"Incident response","text":"Critical security incidents must be escalated within fifteen minutes. The incident commander coordinates containment and recovery. A post incident review must be completed within five business days."},
    {"id":"policy_06","title":"Model governance","text":"Machine learning models require validation before deployment. Model performance and drift are checked weekly. A model owner must approve every production model change."},
    {"id":"policy_07","title":"Data quality","text":"Every ingestion pipeline validates required fields and data types. Invalid records are stored in a quarantine dataset. Duplicate events are identified by their event identifier."},
    {"id":"policy_08","title":"Support hours","text":"The support team operates from 08:00 to 18:00 on weekdays. Critical incidents receive support at all hours. General support requests receive an initial response within one business day."}
]
chunks = []
for document in knowledge:
    for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", document["text"])):
        chunks.append({"chunk_id":document["id"]+f"_{index}", "document_id":document["id"], "title":document["title"], "text":sentence})
chunks = pd.DataFrame(chunks)

from scipy.sparse import csr_matrix

sentences=chunks.copy()
vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
X=vectorizer.fit_transform(sentences.text)
similarity=cosine_similarity(X)
np.fill_diagonal(similarity,0)
row_sums=similarity.sum(axis=1,keepdims=True)
transition=np.divide(similarity,row_sums,out=np.zeros_like(similarity),where=row_sums>0)
dangling=row_sums.ravel()==0
scores=np.full(len(sentences),1/len(sentences))
damping=0.85
convergence=[]
for iteration in range(200):
    updated=(1-damping)/len(sentences)+damping*(transition.T@scores+scores[dangling].sum()/len(sentences))
    delta=float(np.abs(updated-scores).sum())
    convergence.append({"iteration":iteration,"l1_change":delta})
    scores=updated
    if delta<1e-10:
        break


scores/=scores.sum()
sentences["textrank"]=scores
query="security model governance data retention policies"
query_vector=vectorizer.transform([query])
query_relevance=cosine_similarity(query_vector,X).ravel()
relevance=0.6*(scores/max(scores))+0.4*query_relevance
selected=[]
remaining=list(range(len(sentences)))
word_budget=90
used_words=0
while remaining:
    eligible=[position for position in remaining if used_words+len(sentences.iloc[position].text.split())<=word_budget]
    if not eligible:
        break
    position=max(eligible,key=lambda i:0.7*relevance[i]-0.3*max([similarity[i,j] for j in selected] or [0]))
    selected.append(position)
    used_words+=len(sentences.iloc[position].text.split())
    remaining.remove(position)
summary=sentences.iloc[sorted(selected)].copy()
summary["selection_rank"]=[selected.index(index)+1 for index in sorted(selected)]
summary_text=" ".join(summary.text)
source_documents=summary.document_id.unique().tolist()
redundancy=[]
for i,left in enumerate(selected):
    for right in selected[i+1:]:
        redundancy.append(float(similarity[left,right]))
save_table(sentences.sort_values("textrank",ascending=False),"sentence_ranks")
save_table(summary,"selected_sentences")
save_table(pd.DataFrame(convergence),"pagerank_convergence")
save_json({"query":query,"summary":summary_text,"citations":summary.chunk_id.tolist(),"source_documents":source_documents,"word_count":used_words,"word_budget":word_budget},"extractive_summary")
save_model(vectorizer,"summarizer_vectorizer")
assert used_words<=word_budget
assert all(sentence in chunks.text.to_list() for sentence in summary.text)
assert np.isclose(scores.sum(),1)
fig,axes=plt.subplots(1,2,figsize=(11,4))


axes[0].plot([item["iteration"] for item in convergence],[item["l1_change"] for item in convergence])
axes[0].set_yscale("log")
sentences.groupby("document_id").textrank.sum().plot.bar(ax=axes[1])
save_figure(fig,"summarization_diagnostics")
result=finish({"source_sentences":len(sentences),"summary_sentences":len(summary),"summary_words":used_words,"document_coverage":len(source_documents)/len(knowledge),"mean_pairwise_redundancy":float(np.mean(redundancy)) if redundancy else 0.0},[summary])
