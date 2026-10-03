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
WORKFLOW = "129_variational_mixture_component_pruning"


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


from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score

centers=np.array([[-3,-2],[0,3],[3,-1]])
labels=rng.choice(3,1000,p=[.2,.5,.3])
X=centers[labels]+rng.normal(0,.65,(1000,2))
train,test=train_test_split(np.arange(len(X)),test_size=.3,random_state=SEED)
models={}
results=[]
for concentration in [.01,.1,1,10]:
    model=BayesianGaussianMixture(n_components=10,weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=concentration,max_iter=700,n_init=2,random_state=SEED,reg_covar=1e-5).fit(X[train])
    probabilities=model.predict_proba(X[test])
    results.append({'concentration':concentration,'active_components':int((model.weights_>.02).sum()),'heldout_log_density':model.score(X[test]),'heldout_ari':adjusted_rand_score(labels[test],model.predict(X[test])),'converged':bool(model.converged_)})
    models[concentration]=model
model=models[.1]
responsibilities=model.predict_proba(X[test])
entropy=-np.sum(responsibilities*np.log(np.maximum(responsibilities,1e-12)),axis=1)
component_table=pd.DataFrame({'component':range(10),'posterior_weight':model.weights_,'center_x':model.means_[:,0],'center_y':model.means_[:,1],'effective_training_mass':model.predict_proba(X[train]).sum(axis=0)})
assignment=pd.DataFrame({'sample':test,'x':X[test,0],'y':X[test,1],'component':responsibilities.argmax(axis=1),'assignment_entropy':entropy,'log_density':model.score_samples(X[test])})
generated,component=model.sample(500)
assert np.allclose(responsibilities.sum(axis=1),1)
assert np.isclose(model.weights_.sum(),1)
save_table(pd.DataFrame(results),'prior_concentration_sensitivity')
save_table(component_table,'posterior_mixture_components')
save_table(assignment,'heldout_soft_assignments')
save_table(pd.DataFrame({'x':generated[:,0],'y':generated[:,1],'component':component}),'posterior_predictive_samples')
save_model(model,'variational_mixture')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(X[test,0],X[test,1],c=responsibilities.argmax(axis=1),s=12)
axes[1].bar(component_table.component,component_table.posterior_weight)
save_figure(fig,'variational_component_pruning')
finish({'selected_prior_concentration':.1,'active_components':int((model.weights_>.02).sum()),'heldout_log_density':model.score(X[test]),'mean_assignment_entropy':float(entropy.mean()),'converged':bool(model.converged_)},[component_table])
