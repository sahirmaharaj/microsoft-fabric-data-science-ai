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
WORKFLOW = "32_topic_modeling_and_document_clusters"


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

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

vectorizer=TfidfVectorizer(min_df=3,max_df=0.8,ngram_range=(1,2),max_features=2500)
X=vectorizer.fit_transform(corpus.text)
terms=vectorizer.get_feature_names_out()
models={}
selection=[]
for topic_count in [4,6,8,10]:
    model=NMF(n_components=topic_count,init="nndsvda",max_iter=500,random_state=SEED)
    document_weights=model.fit_transform(X)
    top_words=[set(np.argsort(component)[-10:]) for component in model.components_]
    overlaps=[len(a&b)/len(a|b) for i,a in enumerate(top_words) for b in top_words[i+1:]]
    selection.append({"topics":topic_count,"reconstruction_error":model.reconstruction_err_,"mean_topic_overlap":float(np.mean(overlaps)),"iterations":model.n_iter_})
    models[topic_count]=(model,document_weights)


model,weights=models[6]
normalized=normalize(weights,norm="l1")
assignments=corpus.copy()
assignments["topic_id"]=normalized.argmax(axis=1)
assignments["topic_strength"]=normalized.max(axis=1)
assignments["topic_entropy"]=-(normalized*np.log(np.clip(normalized,1e-12,1))).sum(axis=1)
topic_terms=[]
for index,component in enumerate(model.components_):
    for rank,term_index in enumerate(np.argsort(component)[-15:][::-1],1):
        topic_terms.append({"topic_id":index,"rank":rank,"term":terms[term_index],"weight":component[term_index]})
topic_terms=pd.DataFrame(topic_terms)
representatives=assignments.sort_values("topic_strength",ascending=False).groupby("topic_id").head(4)
counts=CountVectorizer(min_df=3,max_df=0.8,max_features=2000)
count_matrix=counts.fit_transform(corpus.text)
lda=LatentDirichletAllocation(n_components=6,max_iter=15,learning_method="batch",random_state=SEED,n_jobs=2)
lda_weights=lda.fit_transform(count_matrix)
comparison=pd.DataFrame({"document_id":corpus.document_id,"nmf_topic":weights.argmax(axis=1),"lda_topic":lda_weights.argmax(axis=1),"known_topic":corpus.topic})
save_table(pd.DataFrame(selection),"topic_count_diagnostics")
save_table(topic_terms,"topic_keywords")
save_table(assignments,"document_topics")
save_table(representatives,"representative_documents")
save_table(comparison,"model_comparison")
save_model({"vectorizer":vectorizer,"nmf":model,"count_vectorizer":counts,"lda":lda},"topic_models")
assert np.allclose(normalized.sum(axis=1),1)
fig,axes=plt.subplots(1,2,figsize=(11,4))
assignments.topic_id.value_counts().sort_index().plot.bar(ax=axes[0])
axes[1].hist(assignments.topic_strength,bins=20)
save_figure(fig,"topic_distribution")
result=finish({"topics":6,"nmf_known_topic_ari":adjusted_rand_score(corpus.topic,assignments.topic_id),"nmf_lda_agreement":adjusted_rand_score(comparison.nmf_topic,comparison.lda_topic),"vocabulary":len(terms)},[topic_terms,representatives])
