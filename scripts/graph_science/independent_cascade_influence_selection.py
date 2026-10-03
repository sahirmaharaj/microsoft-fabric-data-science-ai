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
WORKFLOW = "147_independent_cascade_influence_selection"


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


nodes=48
edge_mask=rng.random((nodes,nodes))<.06
np.fill_diagonal(edge_mask,False)
probability=np.where(edge_mask,rng.uniform(.08,.35,(nodes,nodes)),0)
trials=140
reachability=[]
for trial in range(trials):
    reach=rng.random((nodes,nodes))<probability
    np.fill_diagonal(reach,True)
    for intermediate in range(nodes):
        reach|=reach[:,intermediate,None]&reach[None,intermediate,:]
    reachability.append(reach)
reachability=np.asarray(reachability)
selected=[]
covered=np.zeros((trials,nodes),dtype=bool)
history=[]
for budget in range(1,7):
    candidates=[node for node in range(nodes) if node not in selected]
    gains=[]
    for node in candidates:
        gain=float((reachability[:,node,:]&~covered).sum(axis=1).mean())
        gains.append((gain,node))
    gain,node=max(gains,key=lambda item:(item[0],-item[1]))
    selected.append(node)
    covered|=reachability[:,node,:]
    history.append({'budget':budget,'new_seed':node,'marginal_gain':gain,'expected_spread':float(covered.sum(axis=1).mean())})
evaluation=[]
independent_reach=[]
for trial in range(300):
    reach=rng.random((nodes,nodes))<probability
    np.fill_diagonal(reach,True)
    for intermediate in range(nodes):
        reach|=reach[:,intermediate,None]&reach[None,intermediate,:]
    independent_reach.append(reach)
independent_reach=np.asarray(independent_reach)


degree_order=np.argsort(-probability.sum(axis=1))
random_order=rng.permutation(nodes)
for strategy,seeds in [('greedy',selected),('weighted_degree',degree_order[:6]),('random',random_order[:6])]:
    spread=independent_reach[:,seeds,:].any(axis=1).sum(axis=1)
    evaluation.append({'strategy':strategy,'mean_spread':float(spread.mean()),'mc_standard_error':float(spread.std(ddof=1)/np.sqrt(len(spread))),'seeds':json.dumps(list(map(int,seeds)))})
assert len(set(selected))==6
assert np.diff([row['expected_spread'] for row in history]).min()>=0
save_table(pd.DataFrame(history),'greedy_seed_acquisition')
save_table(pd.DataFrame(evaluation),'independent_spread_evaluation')
save_table(pd.DataFrame([{'from_node':int(i),'to_node':int(j),'activation_probability':probability[i,j]} for i,j in zip(*np.nonzero(edge_mask))]),'cascade_network')
fig,ax=plt.subplots(figsize=(8,4))
ax.plot([row['budget'] for row in history],[row['expected_spread'] for row in history],marker='o')
ax.set(xlabel='Seed budget',ylabel='Expected activated nodes')
save_figure(fig,'influence_budget_curve')
finish({'selected_seeds':selected,'training_simulations':trials,'evaluation_simulations':300,'evaluation_uses_independent_live_edge_graphs':True},[pd.DataFrame(evaluation)])
