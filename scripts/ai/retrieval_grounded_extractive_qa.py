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
WORKFLOW = "34_retrieval_grounded_extractive_qa"


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

vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words="english")
index=vectorizer.fit_transform(chunks.title+" "+chunks.text)

def answer_question(question,limit=3,minimum_similarity=0.12):
    query=vectorizer.transform([question])
    scores=cosine_similarity(query,index).ravel()
    ranked=np.argsort(-scores)
    selected=[]
    for position in ranked:
        if scores[position]<minimum_similarity or len(selected)>=limit:
            break
        if selected:
            redundancy=cosine_similarity(index[position],index[selected]).max()
            if redundancy>0.85:
                continue
        selected.append(position)
    evidence=[]
    for position in selected:
        row=chunks.iloc[position]
        evidence.append({"chunk_id":row.chunk_id,"document_id":row.document_id,"quote":row.text,"score":float(scores[position])})
    return {"question":question,"answer":" ".join(item["quote"] for item in evidence) if evidence else "INSUFFICIENT_EVIDENCE", "citations":[item["chunk_id"] for item in evidence], "evidence":evidence,"mode":"extractive","abstained":not bool(evidence)}



questions=[
    ("How long do refund requests have to be submitted?","policy_01"),
    ("How long are operational logs retained?","policy_03"),
    ("When must a critical security incident be escalated?","policy_05"),
    ("How often is model drift checked?","policy_06"),
    ("How are duplicate events identified?","policy_07"),
    ("What is the weather on Mars?",None)
]
answers=[]
evaluations=[]
for question,expected_document in questions:
    response=answer_question(question)
    answers.append(response)
    retrieved_documents={entry["document_id"] for entry in response["evidence"]}
    grounded=all(entry["quote"] in chunks.set_index("chunk_id").loc[entry["chunk_id"],"text"] for entry in response["evidence"])
    evaluations.append({"question":question,"expected_document":expected_document or "none","retrieval_hit":expected_document in retrieved_documents if expected_document else response["abstained"],"grounded":grounded,"abstained":response["abstained"],"citations":len(response["citations"])})
evaluation=pd.DataFrame(evaluations)
answer_table=pd.DataFrame([{key:value for key,value in answer.items() if key!="evidence"} for answer in answers])
save_json(answers,"answers_with_evidence")
save_table(chunks,"knowledge_chunks")
save_table(evaluation,"qa_evaluation")
save_table(answer_table,"answers")
save_model({"vectorizer":vectorizer,"index":index,"chunks":chunks},"extractive_qa_index")
assert evaluation.grounded.all()
assert all(set(answer["citations"]).issubset(set(chunks.chunk_id)) for answer in answers)
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(np.arange(len(evaluation)),evaluation.citations)
ax.set(xlabel="Question",ylabel="Evidence citations")
save_figure(fig,"retrieval_evidence")
result=finish({"questions":len(questions),"retrieval_hit_rate":float(evaluation.retrieval_hit.mean()),"grounding_rate":float(evaluation.grounded.mean()),"abstentions":int(evaluation.abstained.sum()),"generator":"extractive"},[evaluation,answer_table])
