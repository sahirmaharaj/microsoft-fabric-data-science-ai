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
WORKFLOW = "168_cross_fitted_off_policy_bandit_evaluation"


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


from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

count,actions,features=3200,4,5
X=rng.normal(size=(count,features))
reward_weights=rng.normal(0,.6,(features,actions))
true_reward=expit(X@reward_weights+np.array([-.4,.2,.5,-.2]))
behavior_logits=X@rng.normal(0,.5,(features,actions))
behavior=.2/actions+.8*softmax(behavior_logits,axis=1)
target=softmax(X@(reward_weights+rng.normal(0,.2,reward_weights.shape))*1.4,axis=1)
logged_action=np.array([rng.choice(actions,p=probability) for probability in behavior])
reward=rng.binomial(1,true_reward[np.arange(count),logged_action])
logged_propensity=behavior[np.arange(count),logged_action]
target_propensity=target[np.arange(count),logged_action]
importance=target_propensity/logged_propensity
predicted_reward=np.zeros((count,actions))
folds=KFold(n_splits=5,shuffle=True,random_state=SEED)
for train,test in folds.split(X):
    for action in range(actions):
        selected=train[logged_action[train]==action]
        model=LogisticRegression(C=1,max_iter=1000).fit(X[selected],reward[selected])
        predicted_reward[test,action]=model.predict_proba(X[test])[:,1]
direct=np.sum(target*predicted_reward,axis=1)
residual=reward-predicted_reward[np.arange(count),logged_action]
doubly_robust=direct+importance*residual
truth=float(np.mean(np.sum(target*true_reward,axis=1)))
estimates=[{'estimator':'IPS','value':float(np.mean(importance*reward))},{'estimator':'self_normalized_IPS','value':float(np.sum(importance*reward)/importance.sum())},{'estimator':'direct','value':float(direct.mean())},{'estimator':'cross_fitted_DR','value':float(doubly_robust.mean())}]
for row in estimates:
    row['synthetic_policy_value']=truth
    row['error']=row['value']-truth
bootstrap=[]
for iteration in range(500):
    index=rng.integers(0,count,count)
    bootstrap.append(float(doubly_robust[index].mean()))


clipping=[]
for cap in [1,2,5,10,20]:
    weight=np.minimum(importance,cap)
    clipping.append({'weight_cap':cap,'clipped_ips':float(np.mean(weight*reward)),'clipped_dr':float(np.mean(direct+weight*residual)),'clipped_fraction':float((importance>cap).mean())})
assert np.all(logged_propensity>=.2/actions-1e-10)
save_table(pd.DataFrame(estimates),'policy_value_estimators')
save_table(pd.DataFrame(clipping),'importance_weight_clipping_sensitivity')
save_table(pd.DataFrame({'action':logged_action,'reward':reward,'behavior_propensity':logged_propensity,'target_propensity':target_propensity,'importance_weight':importance,'dr_score':doubly_robust}),'logged_bandit_evaluation_rows')
save_table(pd.DataFrame({'dr_bootstrap_value':bootstrap}),'policy_value_bootstrap')
fig,ax=plt.subplots(figsize=(8,4))
ax.hist(importance,bins=40)
ax.set(xlabel='Importance weight',ylabel='Logged interactions')
save_figure(fig,'policy_overlap_diagnostics')
finish({'target_policy_value_truth':truth,'cross_fitted_dr':float(doubly_robust.mean()),'dr_bootstrap_interval':np.quantile(bootstrap,[.025,.975]),'importance_effective_sample_size':float(importance.sum()**2/np.sum(importance**2)),'behavior_propensities_known':True},[pd.DataFrame(estimates)])
