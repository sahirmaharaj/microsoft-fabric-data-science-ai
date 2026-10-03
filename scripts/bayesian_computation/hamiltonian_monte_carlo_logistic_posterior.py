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
WORKFLOW = "128_hamiltonian_monte_carlo_logistic_posterior"


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

X=np.column_stack([np.ones(300),rng.normal(size=(300,3))])
true_beta=np.array([-.3,1.1,-1.5,.7])
y=rng.binomial(1,expit(X@true_beta))
prior_precision=.25

def potential(beta):
    logits=X@beta
    return float(np.sum(np.logaddexp(0,logits)-y*logits)+.5*prior_precision*(beta@beta))

def gradient(beta):
    return X.T@(expit(X@beta)-y)+prior_precision*beta

def leapfrog(position,momentum,step_size,steps):
    position=position.copy()
    momentum=momentum.copy()-.5*step_size*gradient(position)
    for step in range(steps):
        position+=step_size*momentum
        if step!=steps-1:
            momentum-=step_size*gradient(position)
    momentum-=.5*step_size*gradient(position)
    return position,-momentum

chains=[]
acceptance=[]
for chain_id in range(3):
    random=np.random.default_rng(SEED+chain_id)
    beta=random.normal(0,.2,4)
    draws=[]
    accepted=0
    for iteration in range(2200):
        momentum=random.normal(size=4)
        proposal,final_momentum=leapfrog(beta,momentum,.045,16)
        log_ratio=potential(beta)+.5*(momentum@momentum)-potential(proposal)-.5*(final_momentum@final_momentum)
        if np.log(random.random())<log_ratio:
            beta=proposal
            accepted+=1
        if iteration>=600:
            draws.append(beta.copy())
    chains.append(draws)
    acceptance.append(accepted/2200)


chains=np.asarray(chains)
pooled=chains.reshape(-1,4)
posterior=pd.DataFrame({'coefficient':range(4),'true':true_beta,'mean':pooled.mean(axis=0),'lower':np.quantile(pooled,.025,axis=0),'upper':np.quantile(pooled,.975,axis=0)})
query=np.column_stack([np.ones(80),np.linspace(-2,2,80),np.zeros((80,2))])
probabilities=expit(pooled@query.T)
curve=pd.DataFrame({'x1':query[:,1],'mean_probability':probabilities.mean(axis=0),'lower':np.quantile(probabilities,.025,axis=0),'upper':np.quantile(probabilities,.975,axis=0)})
probe=rng.normal(size=4)
numerical=np.array([(potential(probe+np.eye(4)[i]*1e-5)-potential(probe-np.eye(4)[i]*1e-5))/2e-5 for i in range(4)])
np.testing.assert_allclose(gradient(probe),numerical,rtol=1e-5)
save_table(posterior,'bayesian_logistic_coefficients')
save_table(curve,'posterior_probability_curve')
save_table(pd.DataFrame(pooled,columns=['intercept','x1','x2','x3']),'hmc_draws')
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(curve.x1,curve.mean_probability)
ax.fill_between(curve.x1,curve.lower,curve.upper,alpha=.3)
save_figure(fig,'hmc_prediction_uncertainty')
finish({'acceptance_rates':acceptance,'retained_draws':len(pooled),'analytic_gradient_verified':True,'step_size':.045,'leapfrog_steps':16},[posterior])
