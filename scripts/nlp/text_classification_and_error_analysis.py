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
WORKFLOW = "31_text_classification_and_error_analysis"


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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train, test = train_test_split(corpus, test_size=0.25, stratify=corpus.topic, random_state=SEED)
pipeline = Pipeline([("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=6000)), ("classifier", LogisticRegression(max_iter=1000))])
search = GridSearchCV(pipeline, {"classifier__C": [0.2, 1, 5], "vectorizer__ngram_range": [(1,1),(1,2)]}, scoring="f1_macro", cv=4, n_jobs=2)
search.fit(train.text, train.topic)
model = search.best_estimator_
predictions = model.predict(test.text)
probabilities = model.predict_proba(test.text)
results = test.copy()
results["prediction"] = predictions


results["confidence"] = probabilities.max(axis=1)
results["correct"] = results.prediction.eq(results.topic)
results["entropy"] = -(probabilities*np.log(np.clip(probabilities,1e-12,1))).sum(axis=1)
report = pd.DataFrame(classification_report(test.topic,predictions,output_dict=True,zero_division=0)).T.reset_index(names="label")
feature_names = model.named_steps["vectorizer"].get_feature_names_out()
class_features=[]
for index,label in enumerate(model.classes_):
    coefficients=model.named_steps["classifier"].coef_[index]
    for position in np.argsort(coefficients)[-12:][::-1]:
        class_features.append({"class":label,"feature":feature_names[position],"weight":coefficients[position]})
class_features=pd.DataFrame(class_features)
new_tickets=pd.DataFrame({"text":["my invoice has a duplicate payment charge", "server connection timeout database error", "parcel tracking shows delayed delivery", "dashboard refresh dataset error"]})
new_tickets["prediction"]=model.predict(new_tickets.text)
new_tickets["confidence"]=model.predict_proba(new_tickets.text).max(axis=1)
new_tickets["needs_review"]=new_tickets.confidence.lt(0.6)
save_table(results.sort_values("entropy",ascending=False),"test_predictions")
save_table(report,"classification_report")
save_table(class_features,"class_terms")
save_table(new_tickets,"new_ticket_predictions")
save_model(model,"ticket_router")
assert set(train.document_id).isdisjoint(test.document_id)
assert len(model.classes_)==6
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].imshow(confusion_matrix(test.topic,predictions,labels=model.classes_),cmap="Blues")
axes[0].set_xticks(range(6),model.classes_,rotation=75)
axes[0].set_yticks(range(6),model.classes_)
axes[1].hist(results.confidence,bins=20)
save_figure(fig,"text_classifier")
result=finish({"macro_f1":f1_score(test.topic,predictions,average="macro"),"vocabulary_size":len(feature_names),"test_documents":len(test),"errors":int((~results.correct).sum())},[report,new_tickets])
