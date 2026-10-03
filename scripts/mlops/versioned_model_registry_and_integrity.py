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
WORKFLOW = "48_versioned_model_registry_and_integrity"


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


from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = make_friedman1(n_samples=SAMPLE_SIZE, n_features=10, noise=1.2, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

import sqlite3
from sklearn.metrics import mean_absolute_error

connection=sqlite3.connect(working_dir/"registry.sqlite")
connection.executescript("""
CREATE TABLE model_versions(name TEXT NOT NULL, version INTEGER NOT NULL, artifact_path TEXT NOT NULL, sha256 TEXT NOT NULL, metrics_json TEXT NOT NULL, schema_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(name,version));
CREATE TABLE aliases(name TEXT NOT NULL, alias TEXT NOT NULL, version INTEGER NOT NULL, PRIMARY KEY(name,alias));
CREATE TABLE transitions(name TEXT NOT NULL, alias TEXT NOT NULL, previous_version INTEGER, next_version INTEGER, changed_at TEXT);
""")

def register_model(name,model,metrics,schema):
    with connection:
        version=connection.execute("SELECT COALESCE(MAX(version),0)+1 FROM model_versions WHERE name=?",(name,)).fetchone()[0]
        artifact_path=save_model(model,f"{name}_v{version}")
        digest=hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        created=datetime.now(timezone.utc).isoformat()
        connection.execute("INSERT INTO model_versions VALUES(?,?,?,?,?,?,?)",(name,version,str(artifact_path),digest,json.dumps(metrics),json.dumps(schema),created))
    return int(version)

def set_alias(name,alias,version):
    if connection.execute("SELECT 1 FROM model_versions WHERE name=? AND version=?",(name,version)).fetchone() is None:
        raise ValueError("unknown_model_version")
    with connection:
        previous=connection.execute("SELECT version FROM aliases WHERE name=? AND alias=?",(name,alias)).fetchone()
        connection.execute("INSERT INTO aliases VALUES(?,?,?) ON CONFLICT(name,alias) DO UPDATE SET version=excluded.version",(name,alias,version))
        connection.execute("INSERT INTO transitions VALUES(?,?,?,?,?)",(name,alias,previous[0] if previous else None,version,datetime.now(timezone.utc).isoformat()))



def load_alias(name,alias):
    row=connection.execute("SELECT m.artifact_path,m.sha256,m.schema_json,m.version FROM model_versions m JOIN aliases a ON m.name=a.name AND m.version=a.version WHERE a.name=? AND a.alias=?",(name,alias)).fetchone()
    if row is None:
        raise ValueError("unknown_alias")
    path=Path(row[0])
    if path.parent.resolve()!=output_dir.resolve():
        raise ValueError("untrusted_model_path")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row[1]:
        raise ValueError("artifact_integrity_failure")
    return joblib.load(path),json.loads(row[2]),row[3]

X_fit,X_valid,y_fit,y_valid=train_test_split(X_train,y_train,test_size=0.25,random_state=SEED)
candidates={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=2)),"forest":RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=SEED,n_jobs=2)}
versions=[]
for label,model in candidates.items():
    model.fit(X_fit,y_fit)
    metrics={"validation_mae":mean_absolute_error(y_valid,model.predict(X_valid))}
    version=register_model("demand",model,metrics,{"features":X.shape[1],"dtype":"float64"})
    versions.append({"version":version,"candidate":label,**metrics})
leaderboard=pd.DataFrame(versions).sort_values("validation_mae")
set_alias("demand","champion",int(leaderboard.iloc[0].version))
loaded,schema,version=load_alias("demand","champion")
prediction=loaded.predict(X_test)
assert schema["features"]==X_test.shape[1]
assert version==int(leaderboard.iloc[0].version)
registry=pd.read_sql_query("SELECT * FROM model_versions",connection)
transitions=pd.read_sql_query("SELECT * FROM transitions",connection)
save_table(leaderboard,"candidate_metrics")
save_table(registry,"model_versions")
save_table(transitions,"alias_history")
save_table(pd.DataFrame({"actual":y_test,"prediction":prediction}),"champion_predictions")
connection.close()
__import__("shutil").copy2(working_dir/"registry.sqlite",output_dir/"registry.sqlite")
record_artifact(output_dir/"registry.sqlite")


result=finish({"registered_versions":len(registry),"champion_version":version,"test_mae":mean_absolute_error(y_test,prediction)},[leaderboard,transitions])
