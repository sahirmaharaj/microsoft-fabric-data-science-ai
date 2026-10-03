import os
import sys
import subprocess
import importlib.util
import importlib.metadata
import site

if site.ENABLE_USER_SITE:
    site.addsitedir(site.getusersitepackages())

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
importlib.invalidate_caches()
for thread_variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(thread_variable, "2")


SEED = 42
SAMPLE_SIZE = 1200
OUTPUT_ROOT = os.environ.get("FABRIC_STARTER_OUTPUT", "")
WORKFLOW = "150_capacitated_hungarian_task_assignment"


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


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

workers=9
tasks=20
capacity=np.array([2,2,3,1,2,2,3,2,1])
worker_location=rng.uniform(0,20,(workers,2))
task_location=rng.uniform(0,20,(tasks,2))
skills=rng.uniform(.3,1,(workers,3))
requirements=rng.integers(0,3,tasks)
base_distance=cdist(task_location,worker_location)
base_cost=base_distance+15*(1-skills[:,requirements].T)
qualified=skills[:,requirements].T>=.45
slots=np.repeat(np.arange(workers),capacity)
slot_number=np.concatenate([np.arange(value) for value in capacity])
real_cost=base_cost[:,slots]+slot_number[None,:]*1.5
real_cost=np.where(qualified[:,slots],real_cost,1e6)
penalties=rng.uniform(16,24,tasks)
cost=np.column_stack([real_cost,np.repeat(penalties[:,None],tasks,axis=1)])
row_indices,column_indices=linear_sum_assignment(cost)
assignments=[]
for task,slot in zip(row_indices,column_indices):
    assigned=slot<len(slots)
    worker=int(slots[slot]) if assigned else -1
    assignments.append({'task':int(task),'worker':worker,'slot':int(slot),'assigned':assigned,'required_skill':int(requirements[task]),'cost':float(cost[task,slot]),'unassigned_penalty':float(penalties[task])})
output=pd.DataFrame(assignments)
load=output.loc[output.assigned].groupby('worker').size().reindex(range(workers),fill_value=0)
assert (load.to_numpy()<=capacity).all()
assert output.task.is_unique
for row in assignments:
    if row['assigned']:
        assert qualified[row['task'],row['worker']]
scenarios=[]
for multiplier in [.6,1,1.5,2]:
    scenario_cost=np.column_stack([real_cost,np.repeat((penalties*multiplier)[:,None],tasks,axis=1)])
    rows,columns=linear_sum_assignment(scenario_cost)
    scenarios.append({'penalty_multiplier':multiplier,'assigned_tasks':int((columns<len(slots)).sum()),'objective_cost':float(scenario_cost[rows,columns].sum())})


workers_table=pd.DataFrame({'worker':range(workers),'capacity':capacity,'assigned_tasks':load.to_numpy(),'utilization':load.to_numpy()/capacity})
save_table(output,'optimal_task_assignments')
save_table(workers_table,'worker_capacity_utilization')
save_table(pd.DataFrame(scenarios),'unassigned_penalty_sensitivity')
save_table(pd.DataFrame(base_cost),'task_worker_costs')
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(*worker_location.T,marker='s',s=80)
ax.scatter(*task_location.T,s=25)
for row in assignments:
    if row['assigned']:
        pair=np.vstack([task_location[row['task']],worker_location[row['worker']]])
        ax.plot(*pair.T,alpha=.4)
save_figure(fig,'capacity_feasible_assignments')
finish({'assigned_tasks':int(output.assigned.sum()),'unassigned_tasks':int((~output.assigned).sum()),'objective_cost':float(output.cost.sum()),'capacities_and_qualifications_verified':True},[workers_table])
