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
WORKFLOW = "33_hybrid_bm25_lsa_search"


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

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

count_vectorizer=CountVectorizer(min_df=1)
counts=count_vectorizer.fit_transform(corpus.text).astype(float)
lengths=np.asarray(counts.sum(axis=1)).ravel()
average_length=lengths.mean()
document_frequency=np.asarray((counts>0).sum(axis=0)).ravel()
idf=np.log(1+(len(corpus)-document_frequency+0.5)/(document_frequency+0.5))
tfidf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=3000)
X=tfidf.fit_transform(corpus.text)
svd=TruncatedSVD(n_components=32,random_state=SEED)
embeddings=normalize(svd.fit_transform(X))

def bm25(query,k1=1.5,b=0.75):
    query_terms=count_vectorizer.transform([query]).indices
    scores=np.zeros(len(corpus))
    for term in query_terms:
        frequency=counts[:,term].toarray().ravel()
        denominator=frequency+k1*(1-b+b*lengths/average_length)
        scores+=idf[term]*frequency*(k1+1)/np.maximum(denominator,1e-12)
    return scores



def hybrid_search(query,limit=8):
    lexical=bm25(query)
    dense_query=normalize(svd.transform(tfidf.transform([query])))
    semantic=(embeddings@dense_query.T).ravel()
    lexical_order=np.argsort(-lexical)
    semantic_order=np.argsort(-semantic)
    fused=np.zeros(len(corpus))
    for ordering in [lexical_order,semantic_order]:
        fused[ordering]+=1/(60+np.arange(1,len(corpus)+1))
    positions=np.argsort(-fused)[:limit]
    result=corpus.iloc[positions].copy()
    result["bm25"]=lexical[positions]
    result["lsa_cosine"]=semantic[positions]
    result["rrf_score"]=fused[positions]
    result["rank"]=np.arange(1,len(result)+1)
    return result

queries={"billing":"invoice refund payment","technical":"server timeout connection","delivery":"parcel courier tracking","analytics":"dashboard dataset report","security":"encryption threat audit","machine_learning":"model training validation"}
rankings=[]
evaluation=[]
for expected,query in queries.items():
    result=hybrid_search(query,10)
    result["query"]=query
    rankings.append(result)
    relevant=result.topic.eq(expected).to_numpy()
    reciprocal=1/(np.flatnonzero(relevant)[0]+1) if relevant.any() else 0.0
    evaluation.append({"query":query,"precision_at_10":relevant.mean(),"mrr":reciprocal})
rankings=pd.concat(rankings,ignore_index=True)
evaluation=pd.DataFrame(evaluation)
save_table(rankings,"search_results")
save_table(evaluation,"retrieval_metrics")
save_model({"tfidf":tfidf,"svd":svd,"embeddings":embeddings,"count_vectorizer":count_vectorizer,"counts":counts,"corpus":corpus,"idf":idf,"lengths":lengths},"hybrid_index")
fig,ax=plt.subplots(figsize=(10,4))
evaluation.plot.barh(x="query",y="precision_at_10",ax=ax,legend=False)


save_figure(fig,"retrieval_quality")
result=finish({"documents":len(corpus),"mean_precision_at_10":float(evaluation.precision_at_10.mean()),"mean_reciprocal_rank":float(evaluation.mrr.mean())},[evaluation,rankings])
