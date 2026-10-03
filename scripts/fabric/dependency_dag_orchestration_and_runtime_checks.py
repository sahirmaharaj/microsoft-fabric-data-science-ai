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
WORKFLOW = "56_dependency_dag_orchestration_and_runtime_checks"


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


from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

@dataclass(frozen=True)
class Task:
    name: str
    dependencies: tuple
    retries: int=1

specifications=[Task("extract",()),Task("validate",("extract",)),Task("profile",("validate",)),Task("train",("validate",)),Task("score",("train","validate")),Task("publish",("score","profile"))]

def extract(inputs):
    X,y=make_regression(n_samples=800,n_features=8,noise=5,random_state=SEED)
    return {"X":X,"y":y}

def validate(inputs):
    source=inputs["extract"]
    if not np.isfinite(source["X"]).all() or not np.isfinite(source["y"]).all():
        raise ValueError("nonfinite_data")
    if len(source["X"])!=len(source["y"]):
        raise ValueError("row_alignment")
    return source

def profile(inputs):
    source=inputs["validate"]
    return {"rows":len(source["y"]),"features":source["X"].shape[1],"target_mean":float(source["y"].mean())}

def train_model(inputs):
    source=inputs["validate"]
    return Ridge(alpha=1).fit(source["X"][:600],source["y"][:600])

def score(inputs):
    source=inputs["validate"]
    predictions=inputs["train"].predict(source["X"][600:])
    return {"prediction":predictions,"actual":source["y"][600:],"mae":mean_absolute_error(source["y"][600:],predictions)}



def publish(inputs):
    scored=inputs["score"]
    return {"rows":len(scored["prediction"]),"mae":scored["mae"],"profile":inputs["profile"]}

functions={"extract":extract,"validate":validate,"profile":profile,"train":train_model,"score":score,"publish":publish}
artifacts={}
traces=[]
pending={task.name:task for task in specifications}

while pending:
    ready=[task for task in pending.values() if all(dependency in artifacts for dependency in task.dependencies)]
    if not ready:
        raise ValueError("cycle_or_unresolved_dependency")
    def execute(task):
        started=time.perf_counter()
        for attempt in range(task.retries+1):
            try:
                result=functions[task.name]({dependency:artifacts[dependency] for dependency in task.dependencies})
                return task.name,result,{"task":task.name,"attempts":attempt+1,"status":"succeeded","seconds":time.perf_counter()-started}
            except Exception:
                if attempt==task.retries:
                    raise
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(execute,task) for task in ready]
        for future in as_completed(futures):
            name,artifact,trace=future.result()
            artifacts[name]=artifact
            traces.append(trace)
            del pending[name]

fabric_dag={"activities":[{"name":"profile","path":"01_data_profiling_and_quality","timeoutPerCellInSeconds":180,"args":{"SEED":SEED,"SAMPLE_SIZE":SAMPLE_SIZE}},{"name":"classification","path":"09_classification_model_benchmark","timeoutPerCellInSeconds":240,"args":{"SEED":SEED,"SAMPLE_SIZE":SAMPLE_SIZE},"dependencies":["profile"]}],"timeoutInSeconds":900,"concurrency":2}
RUN_FABRIC_DAG=os.environ.get("FABRIC_STARTER_RUN_REMOTE_NOTEBOOKS","0")=="1"
fabric_result=None
if RUN_FABRIC_DAG:
    import notebookutils
    fabric_result=notebookutils.notebook.runMultiple(fabric_dag)


capabilities={"python":sys.version.split()[0],"lakehouse_files_attached":lakehouse_files.is_dir(),"notebookutils_available":importlib.util.find_spec("notebookutils") is not None,"remote_execution_enabled":RUN_FABRIC_DAG,"local_dag_tasks":len(specifications)}
save_table(pd.DataFrame(traces),"dag_execution_trace")
save_table(pd.DataFrame({"actual":artifacts["score"]["actual"],"prediction":artifacts["score"]["prediction"]}),"dag_predictions")
save_json(fabric_dag,"fabric_notebook_dag")
save_json(capabilities,"runtime_capabilities")
save_json(artifacts["publish"],"published_metrics")
save_model(artifacts["train"],"dag_model")
assert len(artifacts)==len(specifications)
assert all(trace["status"]=="succeeded" for trace in traces)
fig,ax=plt.subplots(figsize=(9,4))
pd.DataFrame(traces).plot.barh(x="task",y="seconds",ax=ax,legend=False)
save_figure(fig,"dag_task_durations")
result=finish({"completed_tasks":len(artifacts),"mae":artifacts["publish"]["mae"],"remote_execution_enabled":RUN_FABRIC_DAG},[pd.DataFrame(traces)])
