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
WORKFLOW = "153_pareto_multiobjective_evolution"


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


population_size=64
dimensions=8
population=rng.uniform(0,1,(population_size,dimensions))

def objectives(points):
    first=points[:,0]
    g=1+9*points[:,1:].mean(axis=1)
    return np.column_stack([first,g*(1-np.sqrt(first/g))])

def fronts(scores):
    domination=(scores[:,None,:]<=scores[None,:,:]).all(axis=2)&(scores[:,None,:]<scores[None,:,:]).any(axis=2)
    counts=domination.sum(axis=0)
    layers=[]
    remaining=np.ones(len(scores),dtype=bool)
    while remaining.any():
        layer=np.where(remaining&(counts==0))[0]
        layers.append(layer)
        remaining[layer]=False
        counts-=domination[layer].sum(axis=0)
    return layers

def crowding(scores,indices):
    result=np.zeros(len(indices))
    for objective in range(scores.shape[1]):
        order=np.argsort(scores[indices,objective])
        result[order[[0,-1]]]=np.inf
        span=np.ptp(scores[indices,objective])
        if span>0 and len(indices)>2:
            result[order[1:-1]]+=(scores[indices[order[2:]],objective]-scores[indices[order[:-2]],objective])/span
    return result

history=[]
for generation in range(100):
    parent_a=population[rng.integers(0,population_size,population_size)]
    parent_b=population[rng.integers(0,population_size,population_size)]
    blend=rng.uniform(-.2,1.2,(population_size,dimensions))
    children=blend*parent_a+(1-blend)*parent_b
    mutation=rng.random(children.shape)<1/dimensions
    children=np.clip(children+mutation*rng.normal(0,.08,children.shape),0,1)
    combined=np.vstack([population,children])
    scores=objectives(combined)
    selected=[]
    for layer in fronts(scores):
        available=population_size-len(selected)
        if len(layer)<=available:
            selected.extend(layer)
        else:
            selected.extend(layer[np.argsort(-crowding(scores,layer))[:available]])
            break
    population=combined[selected]
    current=objectives(population)
    history.append({'generation':generation,'nondominated':len(fronts(current)[0]),'mean_distance_to_true_front':float(np.mean(abs(current[:,1]-(1-np.sqrt(current[:,0])))))})


score=objectives(population)
front=fronts(score)[0]
output=pd.DataFrame(population[front],columns=[f'x{i}' for i in range(dimensions)])
output['objective_1']=score[front,0]
output['objective_2']=score[front,1]
assert ((population>=0)&(population<=1)).all()
save_table(output.sort_values('objective_1'),'pareto_solutions')
save_table(pd.DataFrame(history),'pareto_convergence')
fig,ax=plt.subplots(figsize=(8,5))
grid=np.linspace(0,1,200)
ax.plot(grid,1-np.sqrt(grid),label='analytic Pareto front')
ax.scatter(output.objective_1,output.objective_2,label='evolved solutions')
ax.legend()
save_figure(fig,'pareto_tradeoff_front')
finish({'nondominated_solutions':len(front),'final_mean_front_gap':history[-1]['mean_distance_to_true_front'],'global_convergence_guaranteed':False,'population':population_size,'generations':100},[output])
