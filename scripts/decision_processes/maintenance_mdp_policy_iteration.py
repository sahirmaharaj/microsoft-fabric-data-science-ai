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
WORKFLOW = "157_maintenance_mdp_policy_iteration"


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


from scipy.linalg import solve

states=7
actions=3
discount=.95
transition=np.zeros((states,actions,states))
reward=np.zeros((states,actions))
for state in range(states):
    for increment,probability in [(0,.55),(1,.35),(2,.1)]:
        transition[state,0,min(state+increment,states-1)]+=probability
    reward[state,0]=12-2*state if state<states-1 else -35
    transition[state,1,max(0,state-2)]=.8
    transition[state,1,state]+=.2
    reward[state,1]=-6
    transition[state,2,0]=1
    reward[state,2]=-19
assert np.allclose(transition.sum(axis=2),1)
policy=np.zeros(states,dtype=int)
history=[]
for iteration in range(40):
    chosen=transition[np.arange(states),policy]
    values=solve(np.eye(states)-discount*chosen,reward[np.arange(states),policy])
    action_values=reward+discount*np.einsum('sak,k->sa',transition,values)
    improved=action_values.argmax(axis=1)
    history.append({'iteration':iteration,'state0_value':float(values[0]),'changed_actions':int((improved!=policy).sum())})
    if np.array_equal(improved,policy):
        break
    policy=improved
reference=np.zeros(states)
for iteration in range(2000):
    updated=np.max(reward+discount*np.einsum('sak,k->sa',transition,reference),axis=1)
    if np.max(abs(updated-reference))<1e-10:
        reference=updated
        break
    reference=updated


np.testing.assert_allclose(values,reference,atol=1e-7)
returns=[]
for episode in range(1800):
    state=0
    total=0.
    for step in range(220):
        action=int(policy[state])
        total+=discount**step*reward[state,action]
        state=int(rng.choice(states,p=transition[state,action]))
    returns.append(total)
output=pd.DataFrame({'condition_state':range(states),'action':policy,'action_name':np.array(['operate','service','replace'])[policy],'optimal_value':values})
for action in range(actions):
    output[f'action_value_{action}']=action_values[:,action]
save_table(output,'optimal_maintenance_policy')
save_table(pd.DataFrame(history),'policy_iteration_trace')
save_table(pd.DataFrame({'discounted_return':returns}),'policy_simulation_returns')
save_json({'transition':transition,'reward':reward,'discount':discount},'maintenance_mdp')
fig,ax=plt.subplots(figsize=(8,4))
for action in range(actions):
    ax.plot(range(states),action_values[:,action],label=['operate','service','replace'][action])
ax.legend()
save_figure(fig,'maintenance_action_values')
finish({'optimal_start_value':float(values[0]),'simulated_start_value':float(np.mean(returns)),'monte_carlo_standard_error':float(np.std(returns,ddof=1)/np.sqrt(len(returns))),'policy_value_iteration_agreement':True},[output])
