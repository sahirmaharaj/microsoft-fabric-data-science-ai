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
WORKFLOW = "41_contextual_bandits_with_linucb"


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


from scipy.special import expit

arm_count,dimension,rounds=4,6,1800
true_weights=rng.normal(0,0.6,(arm_count,dimension))
contexts=rng.normal(size=(rounds,dimension))
contexts[:,0]=1
expected_rewards=expit(contexts@true_weights.T)
uniform_draws=rng.random((rounds,arm_count))

class LinUCB:
    def __init__(self,arms,features,alpha=0.8):
        self.alpha=alpha
        self.matrices=np.repeat(np.eye(features)[None,:,:],arms,axis=0)
        self.vectors=np.zeros((arms,features))
    def choose(self,context):
        scores=[]
        for arm in range(len(self.matrices)):
            solved=np.linalg.solve(self.matrices[arm],context)
            mean=context@np.linalg.solve(self.matrices[arm],self.vectors[arm])
            uncertainty=self.alpha*np.sqrt(max(context@solved,0))
            scores.append(mean+uncertainty)
        return int(np.argmax(scores))
    def update(self,arm,context,reward):
        self.matrices[arm]+=np.outer(context,context)
        self.vectors[arm]+=reward*context

policy=LinUCB(arm_count,dimension)
arm_totals=np.zeros(arm_count)
arm_counts=np.zeros(arm_count)
logs=[]
for step,context in enumerate(contexts):
    chosen=policy.choose(context)
    reward=float(uniform_draws[step,chosen]<expected_rewards[step,chosen])
    policy.update(chosen,context,reward)
    random_arm=int(rng.integers(arm_count))
    random_reward=float(uniform_draws[step,random_arm]<expected_rewards[step,random_arm])
    best_probability=float(expected_rewards[step].max())
    logs.append({"round":step,"action":chosen,"reward":reward,"random_action":random_arm,"random_reward":random_reward,"expected_regret":best_probability-expected_rewards[step,chosen],"random_expected_regret":best_probability-expected_rewards[step,random_arm]})
    arm_totals[chosen]+=reward
    arm_counts[chosen]+=1


logs=pd.DataFrame(logs)
logs["cumulative_reward"]=logs.reward.cumsum()
logs["cumulative_random_reward"]=logs.random_reward.cumsum()
logs["cumulative_regret"]=logs.expected_regret.cumsum()
logs["cumulative_random_regret"]=logs.random_expected_regret.cumsum()
arm_metrics=pd.DataFrame({"arm":np.arange(arm_count),"pulls":arm_counts,"reward":arm_totals,"reward_rate":np.divide(arm_totals,arm_counts,out=np.zeros(arm_count),where=arm_counts>0)})
holdout_context=rng.normal(size=(300,dimension))
holdout_context[:,0]=1
holdout_rewards=expit(holdout_context@true_weights.T)
actions=np.array([policy.choose(row) for row in holdout_context])
holdout_regret=holdout_rewards.max(axis=1)-holdout_rewards[np.arange(len(actions)),actions]
state_path=output_dir/"linucb_state.npz"
np.savez_compressed(state_path,matrices=policy.matrices,vectors=policy.vectors,alpha=policy.alpha)
record_artifact(state_path)
save_table(logs,"bandit_event_log")
save_table(arm_metrics,"arm_metrics")
save_json({"arms":arm_count,"features":dimension,"rounds":rounds,"reward":"bernoulli","evaluation":"synthetic_known_reward_function"},"bandit_configuration")
assert arm_counts.sum()==rounds
assert logs.expected_regret.ge(-1e-12).all()
fig,axes=plt.subplots(1,2,figsize=(11,4))
logs.plot(x="round",y=["cumulative_reward","cumulative_random_reward"],ax=axes[0])
logs.plot(x="round",y=["cumulative_regret","cumulative_random_regret"],ax=axes[1])
save_figure(fig,"bandit_learning")
result=finish({"reward_rate":float(logs.reward.mean()),"random_reward_rate":float(logs.random_reward.mean()),"cumulative_expected_regret":float(logs.expected_regret.sum()),"holdout_expected_regret":float(holdout_regret.mean())},[arm_metrics])
