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
WORKFLOW = "137_functional_data_basis_regression"


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


from scipy.integrate import trapezoid
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

subjects=500
grid=np.linspace(0,1,90)
latent=rng.normal(size=(subjects,6))
basis=np.stack([np.sin(np.pi*(index+1)*grid) for index in range(6)])
curves=latent@basis+rng.normal(0,.12,(subjects,len(grid)))
true_coefficient=2*np.sin(np.pi*grid)-np.sin(3*np.pi*grid)+.5*np.sin(5*np.pi*grid)
y=1+trapezoid(curves*true_coefficient,grid,axis=1)+rng.normal(0,.25,subjects)
train,test=train_test_split(np.arange(subjects),test_size=.3,random_state=SEED)
spline=SplineTransformer(n_knots=10,degree=3,include_bias=False)
functional_basis=spline.fit_transform(grid[:,None])
features=np.stack([trapezoid(curves*functional_basis[:,column],grid,axis=1) for column in range(functional_basis.shape[1])],axis=1)
model=RidgeCV(alphas=np.logspace(-5,2,20),cv=5).fit(features[train],y[train])
coefficient=functional_basis@model.coef_
prediction=model.predict(features[test])
contribution=curves[test]*coefficient
reconstructed=model.intercept_+trapezoid(contribution,grid,axis=1)
np.testing.assert_allclose(reconstructed,prediction,atol=1e-9)
bootstrap=[]
for iteration in range(120):
    selected=rng.choice(train,len(train),replace=True)
    from sklearn.linear_model import Ridge
    estimate=Ridge(alpha=model.alpha_).fit(features[selected],y[selected])
    bootstrap.append(functional_basis@estimate.coef_)
bootstrap=np.asarray(bootstrap)
coefficient_table=pd.DataFrame({'time':grid,'estimated_coefficient':coefficient,'true_coefficient':true_coefficient,'pointwise_lower':np.quantile(bootstrap,.025,axis=0),'pointwise_upper':np.quantile(bootstrap,.975,axis=0)})
output=pd.DataFrame({'subject':test,'actual':y[test],'prediction':prediction})
segments=[]
for left,right in [(0,.25),(.25,.5),(.5,.75),(.75,1)]:
    selected=(grid>=left)&(grid<=right)
    segments.append({'start':left,'end':right,'mean_absolute_integrated_contribution':float(np.mean(abs(trapezoid(contribution[:,selected],grid[selected],axis=1))))})


save_table(coefficient_table,'functional_coefficient_curve')
save_table(output,'functional_regression_predictions')
save_table(pd.DataFrame(segments),'interval_contribution_summary')
save_model({'regression':model,'functional_basis':functional_basis,'grid':grid,'coefficient':coefficient},'functional_regression_bundle')
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(grid,true_coefficient,label='truth')
ax.plot(grid,coefficient,label='estimate')
ax.fill_between(grid,coefficient_table.pointwise_lower,coefficient_table.pointwise_upper,alpha=.2)
ax.legend()
save_figure(fig,'functional_response_coefficient')
finish({'heldout_rmse':float(np.sqrt(np.mean((prediction-y[test])**2))),'regularization':float(model.alpha_),'integrated_decomposition_verified':True},[output])
