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
WORKFLOW = "42_q_learning_inventory_control"


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


capacity=20
max_order=8
unit_price=9.0
unit_cost=3.0
holding_cost=0.3
lost_sale_cost=4.0
demand_mean=5.0
gamma=0.95
q_values=np.zeros((capacity+1,max_order+1))
visits=np.zeros_like(q_values)

class InventoryEnvironment:
    def __init__(self,seed):
        self.generator=np.random.default_rng(seed)
        self.stock=10
    def reset(self):
        self.stock=10
        return self.stock
    def step(self,order):
        ordered=min(int(order),capacity-self.stock)
        available=self.stock+ordered
        demand=int(self.generator.poisson(demand_mean))
        sold=min(available,demand)
        remaining=available-sold
        lost=demand-sold
        reward=unit_price*sold-unit_cost*ordered-holding_cost*remaining-lost_sale_cost*lost
        self.stock=remaining
        return remaining,float(reward),{"demand":demand,"sales":sold,"lost":lost,"ordered":ordered,"remaining":remaining}

environment=InventoryEnvironment(SEED)
training=[]
for episode in range(900):
    state=environment.reset()
    total_reward=0.0
    epsilon=max(0.05,0.9*np.exp(-episode/250))
    for day in range(60):
        feasible=np.arange(min(max_order,capacity-state)+1)
        action=int(rng.choice(feasible)) if rng.random()<epsilon else int(feasible[np.argmax(q_values[state,feasible])])
        next_state,reward,details=environment.step(action)
        visits[state,action]+=1
        learning_rate=max(0.03,0.6/(1+0.01*visits[state,action]))
        next_actions=np.arange(min(max_order,capacity-next_state)+1)
        target=reward+gamma*np.max(q_values[next_state,next_actions])
        q_values[state,action]+=learning_rate*(target-q_values[state,action])
        total_reward+=reward
        state=next_state
    training.append({"episode":episode,"reward":total_reward,"epsilon":epsilon})


policy=[]
for stock in range(capacity+1):
    feasible=np.arange(min(max_order,capacity-stock)+1)
    action=int(feasible[np.argmax(q_values[stock,feasible])])
    policy.append({"stock":stock,"order":action,"state_value":float(q_values[stock,action])})
policy=pd.DataFrame(policy)
evaluation=[]
for policy_name in ["q_learning","order_up_to_10"]:
    evaluator=InventoryEnvironment(SEED+1000)
    state=evaluator.reset()
    for day in range(365):
        action=int(policy.loc[policy.stock.eq(state),"order"].iloc[0]) if policy_name=="q_learning" else min(max_order,max(0,10-state))
        state,reward,details=evaluator.step(action)
        evaluation.append({"policy":policy_name,"day":day,"reward":reward,**details})
evaluation=pd.DataFrame(evaluation)
metrics=evaluation.groupby("policy").agg(mean_daily_reward=("reward","mean"),lost_units=("lost","sum"),sold_units=("sales","sum"),average_inventory=("remaining","mean")).reset_index()
save_table(pd.DataFrame(training),"training_rewards")
save_table(policy,"inventory_policy")
save_table(evaluation,"policy_simulation")
save_table(metrics,"policy_comparison")
state_path=output_dir/"q_table.npz"
np.savez_compressed(state_path,q_values=q_values,visits=visits)
record_artifact(state_path)
assert (policy.stock+policy.order<=capacity).all()
assert evaluation.remaining.between(0,capacity).all()
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(pd.DataFrame(training).reward.rolling(30).mean())
policy.plot(x="stock",y="order",ax=axes[1],legend=False)
save_figure(fig,"inventory_learning")
result=finish({"episodes":len(training),"learned_mean_reward":float(metrics.loc[metrics.policy.eq("q_learning"),"mean_daily_reward"].iloc[0]),"baseline_mean_reward":float(metrics.loc[metrics.policy.eq("order_up_to_10"),"mean_daily_reward"].iloc[0])},[policy,metrics])
