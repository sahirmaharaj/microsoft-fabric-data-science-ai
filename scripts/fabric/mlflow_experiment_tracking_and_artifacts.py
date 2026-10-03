import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4', 'mlflow': 'mlflow-skinny>=2.15,<4', 'sqlalchemy': 'sqlalchemy>=2,<3', 'alembic': 'alembic>=1.13,<2'}
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
WORKFLOW = "55_mlflow_experiment_tracking_and_artifacts"


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

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag

USE_FABRIC_TRACKING=os.environ.get("FABRIC_STARTER_USE_FABRIC_MLFLOW","0")=="1"
if USE_FABRIC_TRACKING:
    tracking_uri=mlflow.get_tracking_uri()
    client=MlflowClient(tracking_uri=tracking_uri)
    experiment_name="fabric_python_starters_"+run_id
    experiment_id=client.create_experiment(experiment_name)
else:
    tracking_path=working_dir/"mlflow_tracking.sqlite"
    tracking_uri="sqlite:///"+str(tracking_path)
    client=MlflowClient(tracking_uri=tracking_uri)
    experiment_name="fabric_python_starters_"+run_id
    experiment_id=client.create_experiment(experiment_name,artifact_location=(output_dir/"mlflow_artifacts").resolve().as_uri())
X_fit,X_valid,y_fit,y_valid=train_test_split(X_train,y_train,test_size=0.25,random_state=SEED)
run_records=[]
fitted_models={}
for depth,trees in [(4,60),(7,90),(None,120)]:
    model=RandomForestRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=3,random_state=SEED,n_jobs=2)
    run=client.create_run(experiment_id,tags={"workflow":WORKFLOW,"dataset":"synthetic_friedman1","seed":str(SEED),"runtime":"fabric" if USE_FABRIC_TRACKING else "local"})
    identifier=run.info.run_id
    try:
        started=time.perf_counter()
        model.fit(X_fit,y_fit)
        prediction=model.predict(X_valid)
        metrics={"validation_mae":mean_absolute_error(y_valid,prediction),"validation_r2":r2_score(y_valid,prediction),"fit_seconds":time.perf_counter()-started}
        timestamp=int(time.time()*1000)
        client.log_batch(identifier,metrics=[Metric(key,float(value),timestamp,0) for key,value in metrics.items()],params=[Param("n_estimators",str(trees)),Param("max_depth",str(depth)),Param("min_samples_leaf","3")],tags=[RunTag("status","evaluated")])
        model_path=save_model(model,"model_"+identifier)
        client.log_artifact(identifier,str(model_path),artifact_path="model")
        validation_path=save_table(pd.DataFrame({"actual":y_valid,"prediction":prediction}),"validation_"+identifier)
        client.log_artifact(identifier,str(validation_path),artifact_path="evaluation")
        client.set_terminated(identifier,status="FINISHED")
        run_records.append({"run_id":identifier,"trees":trees,"depth":str(depth),**metrics})
        fitted_models[identifier]=model
    except Exception:
        client.set_terminated(identifier,status="FAILED")
        raise


leaderboard=pd.DataFrame(run_records).sort_values("validation_mae")
selected_id=str(leaderboard.iloc[0].run_id)
client.set_tag(selected_id,"selection","best_validation_mae")
selected=fitted_models[selected_id]
test_prediction=selected.predict(X_test)
test_mae=mean_absolute_error(y_test,test_prediction)
client.log_metric(selected_id,"test_mae",float(test_mae))
retrieved=client.search_runs([experiment_id],order_by=["metrics.validation_mae ASC"])
assert len(retrieved)==3
assert retrieved[0].info.run_id==selected_id
artifact_inventory=[]
for item in client.list_artifacts(selected_id,"model"):
    artifact_inventory.append({"path":item.path,"is_dir":item.is_dir,"file_size":item.file_size})
save_table(leaderboard,"mlflow_leaderboard")
save_table(pd.DataFrame(artifact_inventory),"selected_artifacts")
save_json({"experiment_id":experiment_id,"experiment_name":experiment_name,"selected_run_id":selected_id,"tracking_mode":"fabric" if USE_FABRIC_TRACKING else "local_sqlite","test_mae":test_mae},"experiment_receipt")
if not USE_FABRIC_TRACKING:
    import sqlite3
    source_db=sqlite3.connect(tracking_path)
    target_db=sqlite3.connect(output_dir/"mlflow_tracking.sqlite")
    source_db.backup(target_db)
    target_db.close()
    source_db.close()
    record_artifact(output_dir/"mlflow_tracking.sqlite")
fig,ax=plt.subplots(figsize=(9,4))
leaderboard.plot.bar(x="trees",y="validation_mae",ax=ax,legend=False)
save_figure(fig,"experiment_comparison")
result=finish({"runs":len(leaderboard),"selected_run_id":selected_id,"test_mae":test_mae,"tracking_mode":"fabric" if USE_FABRIC_TRACKING else "local_sqlite"},[leaderboard])
