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
WORKFLOW = "47_resumable_batch_scoring_pipeline"


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


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import sqlite3

X,y=make_classification(n_samples=SAMPLE_SIZE,n_features=10,n_informative=6,random_state=SEED)
columns=[f"feature_{i}" for i in range(X.shape[1])]
train=pd.DataFrame(X,columns=columns)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(train,y)
source=train.copy()
source.insert(0,"record_id",[f"record-{i:05d}" for i in range(len(source))])
source_path=output_dir/"scoring_input.csv"
source.to_csv(source_path,index=False)
source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
model_path=save_model(model,"batch_model")
model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
connection=sqlite3.connect(working_dir/"scoring.sqlite")
connection.executescript("""
CREATE TABLE predictions(record_id TEXT PRIMARY KEY, probability REAL NOT NULL, decision INTEGER NOT NULL, model_hash TEXT NOT NULL);
CREATE TABLE checkpoints(batch_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL, model_hash TEXT NOT NULL, row_count INTEGER NOT NULL);
""")

def score_source(chunk_size=170):
    audit=[]
    for index,chunk in enumerate(pd.read_csv(source_path,chunksize=chunk_size)):
        batch_id=f"{source_hash}:{model_hash}:{chunk_size}:{index}"
        previous=connection.execute("SELECT row_count FROM checkpoints WHERE batch_id=?",(batch_id,)).fetchone()
        if previous:
            audit.append({"chunk":index,"rows":len(chunk),"status":"already_committed"})
            continue
        if not chunk.record_id.is_unique or chunk[columns].isna().any().any():
            raise ValueError("invalid_input_batch")
        probability=model.predict_proba(chunk[columns])[:,1]
        if not np.isfinite(probability).all():
            raise ValueError("nonfinite_prediction")
        records=[(row_id,float(p),int(p>=0.5),model_hash) for row_id,p in zip(chunk.record_id,probability)]
        with connection:
            connection.executemany("INSERT INTO predictions VALUES(?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET probability=excluded.probability,decision=excluded.decision,model_hash=excluded.model_hash",records)
            connection.execute("INSERT INTO checkpoints VALUES(?,?,?,?)",(batch_id,source_hash,model_hash,len(chunk)))
        audit.append({"chunk":index,"rows":len(chunk),"status":"committed"})
    return pd.DataFrame(audit)



first_pass=score_source()
second_pass=score_source()
predictions=pd.read_sql_query("SELECT * FROM predictions ORDER BY record_id",connection)
checkpoints=pd.read_sql_query("SELECT * FROM checkpoints",connection)
assert len(predictions)==len(source)
assert predictions.record_id.is_unique
assert second_pass.status.eq("already_committed").all()
expected=model.predict_proba(source[columns])[:,1]
assert np.allclose(predictions.probability,expected)
connection.close()
__import__("shutil").copy2(working_dir/"scoring.sqlite",output_dir/"scoring.sqlite")
record_artifact(output_dir/"scoring.sqlite")
record_artifact(source_path)
save_table(predictions,"predictions")
save_table(first_pass,"first_pass_audit")
save_table(second_pass,"resume_audit")
save_table(checkpoints,"checkpoints")
save_json({"features":columns,"source_sha256":source_hash,"model_sha256":model_hash,"chunk_size":170,"rows":len(source)},"scoring_contract")
fig,ax=plt.subplots(figsize=(9,4))
ax.hist(predictions.probability,bins=25)
save_figure(fig,"scoring_distribution")
result=finish({"scored_rows":len(predictions),"chunks":len(first_pass),"resume_skips":len(second_pass),"positive_predictions":int(predictions.decision.sum())},[first_pass,second_pass])
