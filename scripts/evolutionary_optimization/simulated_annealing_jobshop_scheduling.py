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
WORKFLOW = "154_simulated_annealing_jobshop_scheduling"


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


jobs=9
machines=4
machine_order=np.stack([rng.permutation(machines) for _ in range(jobs)])
durations=rng.integers(2,12,(jobs,machines))
sequence=np.repeat(np.arange(jobs),machines)
rng.shuffle(sequence)

def decode(order):
    next_operation=np.zeros(jobs,dtype=int)
    job_ready=np.zeros(jobs,dtype=int)
    machine_ready=np.zeros(machines,dtype=int)
    records=[]
    for job in order:
        operation=next_operation[job]
        machine=int(machine_order[job,operation])
        start=max(job_ready[job],machine_ready[machine])
        finish_at=start+int(durations[job,operation])
        records.append({'job':int(job),'operation':int(operation),'machine':machine,'start':int(start),'finish':int(finish_at),'duration':int(durations[job,operation])})
        job_ready[job]=machine_ready[machine]=finish_at
        next_operation[job]+=1
    return int(machine_ready.max()),records

current_cost,_=decode(sequence)
initial_cost=current_cost
best_sequence=sequence.copy()
best_cost=current_cost
history=[]
accepted=0
for iteration in range(7000):
    temperature=8*(.015/8)**(iteration/6999)
    proposal=sequence.copy()
    left,right=rng.choice(len(sequence),2,replace=False)
    proposal[left],proposal[right]=proposal[right],proposal[left]
    cost,_=decode(proposal)
    if cost<current_cost or np.log(rng.random())<(current_cost-cost)/temperature:
        sequence,current_cost=proposal,cost
        accepted+=1
    if current_cost<best_cost:
        best_sequence,best_cost=sequence.copy(),current_cost
    if iteration%100==0:
        history.append({'iteration':iteration,'temperature':temperature,'current_makespan':current_cost,'best_makespan':best_cost})


_,records=decode(best_sequence)
schedule=pd.DataFrame(records)
for machine,group in schedule.groupby('machine'):
    group=group.sort_values('start')
    assert np.all(group.start.to_numpy()[1:]>=group.finish.to_numpy()[:-1])
for job,group in schedule.groupby('job'):
    group=group.sort_values('operation')
    assert np.all(group.start.to_numpy()[1:]>=group.finish.to_numpy()[:-1])
lower_bound=max(durations.sum(axis=1).max(),max(durations[machine_order==machine].sum() for machine in range(machines)))
save_table(schedule,'jobshop_schedule')
save_table(pd.DataFrame(history),'annealing_convergence')
save_json({'machine_order':machine_order,'durations':durations,'best_job_sequence':best_sequence},'jobshop_inputs_and_solution')
fig,ax=plt.subplots(figsize=(12,4))
for row in records:
    ax.barh(row['machine'],row['duration'],left=row['start'],color=plt.cm.tab10(row['job']%10))
ax.set(xlabel='Time',ylabel='Machine')
save_figure(fig,'jobshop_machine_schedule')
finish({'initial_makespan':initial_cost,'best_makespan':best_cost,'lower_bound':int(lower_bound),'gap_to_lower_bound':best_cost/lower_bound-1,'feasibility_verified':True,'acceptance_rate':accepted/7000},[schedule])
