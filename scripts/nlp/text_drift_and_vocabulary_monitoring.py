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
WORKFLOW = "38_text_drift_and_vocabulary_monitoring"


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

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

reference=corpus.sample(240,random_state=SEED).copy()
current=pd.concat([corpus.loc[corpus.topic.eq("security")].sample(50,random_state=SEED),corpus.sample(120,random_state=7)],ignore_index=True)
current["text"]=current.text+" quantum ransomware zero trust emerging exploit"
vectorizer=CountVectorizer(min_df=2)
reference_counts=vectorizer.fit_transform(reference.text)
current_counts=vectorizer.transform(current.text)
reference_frequency=np.asarray(reference_counts.sum(axis=0)).ravel()+1
current_frequency=np.asarray(current_counts.sum(axis=0)).ravel()+1
reference_distribution=reference_frequency/reference_frequency.sum()


current_distribution=current_frequency/current_frequency.sum()
js_distance=float(jensenshannon(reference_distribution,current_distribution))
analyzer=vectorizer.build_analyzer()
vocabulary=set(vectorizer.vocabulary_)

def document_statistics(frame):
    output=[]
    for row in frame.itertuples():
        tokens=analyzer(row.text)
        unknown=[token for token in tokens if token not in vocabulary]
        output.append({"document_id":row.document_id,"tokens":len(tokens),"oov_rate":len(unknown)/max(len(tokens),1),"characters":len(row.text),"topic":row.topic})
    return pd.DataFrame(output)

reference_stats=document_statistics(reference)
current_stats=document_statistics(current)
terms=vectorizer.get_feature_names_out()
term_shift=pd.DataFrame({"term":terms,"reference_probability":reference_distribution,"current_probability":current_distribution,"log_ratio":np.log(current_distribution/reference_distribution)}).sort_values("log_ratio",key=abs,ascending=False)
new_terms={}
for text in current.text:
    for token in analyzer(text):
        if token not in vocabulary:
            new_terms[token]=new_terms.get(token,0)+1
new_terms=pd.DataFrame([{"term":term,"count":count} for term,count in new_terms.items()]).sort_values("count",ascending=False)
bootstrap_distances=[]
reference_dense=reference_counts.toarray()
for iteration in range(300):
    a=reference_dense[rng.integers(0,len(reference_dense),len(reference_dense))].sum(axis=0)+1
    b=reference_dense[rng.integers(0,len(reference_dense),len(reference_dense))].sum(axis=0)+1
    bootstrap_distances.append(float(jensenshannon(a/a.sum(),b/b.sum())))
threshold=float(np.quantile(bootstrap_distances,0.99))
length_test=ks_2samp(reference_stats.tokens,current_stats.tokens)
save_table(term_shift,"term_distribution_shift")
save_table(new_terms,"new_vocabulary")
save_table(current_stats,"current_document_statistics")
save_json({"js_distance":js_distance,"reference_bootstrap_p99":threshold,"drift_alert":js_distance>threshold,"length_ks_pvalue":float(length_test.pvalue)},"drift_decision")


save_model(vectorizer,"reference_vocabulary")
fig,axes=plt.subplots(1,2,figsize=(11,4))
term_shift.head(15).plot.barh(x="term",y="log_ratio",ax=axes[0],legend=False)
new_terms.plot.bar(x="term",y="count",ax=axes[1],legend=False)
save_figure(fig,"text_drift")
result=finish({"js_distance":js_distance,"drift_threshold":threshold,"drift_detected":js_distance>threshold,"current_oov_rate":float(current_stats.oov_rate.mean()),"new_terms":len(new_terms)},[term_shift,new_terms])
