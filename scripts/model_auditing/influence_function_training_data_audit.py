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
WORKFLOW = "164_influence_function_training_data_audit"


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
from scipy.optimize import minimize
from scipy.linalg import solve

X=np.column_stack([np.ones(500),rng.normal(size=(500,5))])
true_weights=np.array([-.2,1.2,-1,.7,0,.3])
y=rng.binomial(1,expit(X@true_weights))
training=np.arange(350)
validation=np.arange(350,500)
corrupted=rng.choice(training,25,replace=False)
original=y.copy()
y[corrupted]=1-y[corrupted]
regularization=.03

def fit_model(extra_index=None,extra_weight=0.):
    def objective(beta):
        logits=X[training]@beta
        loss=np.mean(np.logaddexp(0,logits)-y[training]*logits)+.5*regularization*(beta@beta)
        gradient=X[training].T@(expit(logits)-y[training])/len(training)+regularization*beta
        if extra_index is not None:
            logit=X[extra_index]@beta
            loss+=extra_weight*(np.logaddexp(0,logit)-y[extra_index]*logit)
            gradient+=extra_weight*(expit(logit)-y[extra_index])*X[extra_index]
        return float(loss),gradient
    return minimize(objective,np.zeros(X.shape[1]),jac=True,method='BFGS',options={'gtol':1e-9}).x

beta=fit_model()
probability=expit(X[training]@beta)
hessian=X[training].T@(X[training]*(probability*(1-probability))[:,None])/len(training)+regularization*np.eye(X.shape[1])
validation_gradient=X[validation].T@(expit(X[validation]@beta)-y[validation])/len(validation)
inverse_direction=solve(hessian,validation_gradient,assume_a='pos')
training_gradient=X[training]*(probability-y[training])[:,None]
upweight_influence=-training_gradient@inverse_direction
scores=pd.DataFrame({'sample':training,'upweight_validation_loss_derivative':upweight_influence,'synthetically_corrupted':np.isin(training,corrupted),'training_probability':probability,'observed_label':y[training]})
scores=scores.sort_values('upweight_validation_loss_derivative',ascending=False)



def validation_loss(weights):
    logits=X[validation]@weights
    return float(np.mean(np.logaddexp(0,logits)-y[validation]*logits))

baseline=validation_loss(beta)
checks=[]
for index in np.r_[scores['sample'].head(5),scores['sample'].tail(5)]:
    epsilon=.001
    refit=fit_model(int(index),epsilon)
    actual=validation_loss(refit)-baseline
    approximate=epsilon*upweight_influence[int(index)]
    checks.append({'sample':int(index),'epsilon':epsilon,'actual_validation_loss_change':actual,'influence_approximation':approximate})
save_table(scores,'influential_training_records')
save_table(pd.DataFrame(checks),'finite_perturbation_reference')
save_json({'weights':beta,'regularization':regularization,'hessian':hessian},'influence_model_state')
fig,ax=plt.subplots(figsize=(7,5))
check_frame=pd.DataFrame(checks)
ax.scatter(check_frame.influence_approximation,check_frame.actual_validation_loss_change)
ax.set(xlabel='Influence approximation',ylabel='Exact refit loss change')
save_figure(fig,'influence_approximation_validation')
finish({'validation_log_loss':baseline,'corrupted_in_top25':int(scores.head(25).synthetically_corrupted.sum()),'hessian_condition_number':float(np.linalg.cond(hessian)),'positive_influence_means_harmful_upweighting':True},[check_frame])
