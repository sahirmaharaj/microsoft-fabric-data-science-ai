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
WORKFLOW = "138_compositional_logratio_classification"


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


from scipy.linalg import helmert
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import expit

samples,parts=1100,5
composition=rng.dirichlet([.6,1.1,1.4,.8,1.8],samples)
counts=np.stack([rng.multinomial(int(total),row) for total,row in zip(rng.integers(40,400,samples),composition)])
y=rng.binomial(1,expit(.8*np.log(composition[:,0]/composition[:,1])-.5*np.log(composition[:,2]/composition[:,4])))
basis=helmert(parts,full=False)
smoothed=(counts+.5)/(counts.sum(axis=1,keepdims=True)+.5*parts)
clr=np.log(smoothed)-np.log(smoothed).mean(axis=1,keepdims=True)
ilr=clr@basis.T
train,test=train_test_split(np.arange(samples),test_size=.3,stratify=y,random_state=SEED)
model=LogisticRegression(C=1,max_iter=1000).fit(ilr[train],y[train])
probability=model.predict_proba(ilr[test])[:,1]
log_contrast=basis.T@model.coef_.ravel()
assert abs(log_contrast.sum())<1e-10
assert np.allclose(basis@basis.T,np.eye(parts-1))
reconstructed=np.exp(ilr@basis)
reconstructed/=reconstructed.sum(axis=1,keepdims=True)
np.testing.assert_allclose(reconstructed,smoothed,atol=1e-10)
scaled=smoothed*rng.uniform(.1,100,(samples,1))
scaled/=scaled.sum(axis=1,keepdims=True)
np.testing.assert_allclose(np.log(scaled)@basis.T,ilr,atol=1e-10)
output=pd.DataFrame({'sample':test,'actual':y[test],'probability':probability,'count_total':counts[test].sum(axis=1)})
contrasts=pd.DataFrame({'part':range(parts),'log_contrast_weight':log_contrast})
perturbations=[]
for multiplier in [.5,.8,1,1.5,2]:
    scenario=smoothed[test].copy()
    scenario[:,0]*=multiplier
    scenario/=scenario.sum(axis=1,keepdims=True)
    p=model.predict_proba(np.log(scenario)@basis.T)[:,1]
    perturbations.append({'part0_multiplier':multiplier,'mean_predicted_probability':float(p.mean()),'mean_prediction_change':float((p-probability).mean())})


save_table(output,'composition_predictions')
save_table(contrasts,'zero_sum_log_contrast')
save_table(pd.DataFrame(perturbations),'relative_perturbation_scenarios')
save_model({'classifier':model,'ilr_basis':basis,'pseudocount':.5},'compositional_model')
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(contrasts.part,contrasts.log_contrast_weight)
ax.set(xlabel='Composition part',ylabel='Log contrast weight')
save_figure(fig,'compositional_effects')
finish({'heldout_auc':roc_auc_score(y[test],probability),'observed_zero_counts':int((counts==0).sum()),'closure_invariance_verified':True,'perturbation_effects_are_predictive':True},[contrasts])
