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
WORKFLOW = "136_additive_spline_response_decomposition"


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


from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X=rng.uniform(-3,3,(1000,3))
contributions=np.column_stack([2*np.sin(X[:,0]),.6*X[:,1]**2,np.tanh(2*X[:,2])])
y=3+contributions.sum(axis=1)+rng.normal(0,.4,len(X))
train,test=train_test_split(np.arange(len(y)),test_size=.3,random_state=SEED)
spline=SplineTransformer(n_knots=7,degree=3,include_bias=False,extrapolation='linear')
model=make_pipeline(spline,RidgeCV(alphas=np.logspace(-3,3,15),cv=5)).fit(X[train],y[train])
baseline=LinearRegression().fit(X[train],y[train])
prediction=model.predict(X[test])
reference=np.median(X[train],axis=0)
reference_prediction=float(model.predict(reference.reshape(1,-1))[0])
curves=[]
for feature in range(3):
    grid=np.linspace(-3,3,100)
    query=np.tile(reference,(len(grid),1))
    query[:,feature]=grid
    response=model.predict(query)-reference_prediction
    truth_function=[lambda z:2*np.sin(z),lambda z:.6*z**2,lambda z:np.tanh(2*z)][feature]
    true_effect=truth_function(grid)-truth_function(reference[feature])
    for value,effect,truth in zip(grid,response,true_effect):
        curves.append({'feature':feature,'value':value,'estimated_centered_effect':effect,'true_centered_effect':truth})
curves=pd.DataFrame(curves)
contribution_matrix=np.zeros((len(test),3))
for feature in range(3):
    query=np.tile(reference,(len(test),1))
    query[:,feature]=X[test,feature]
    contribution_matrix[:,feature]=model.predict(query)-reference_prediction
reconstructed=reference_prediction+contribution_matrix.sum(axis=1)
np.testing.assert_allclose(reconstructed,prediction,atol=1e-9)
output=pd.DataFrame({'actual':y[test],'prediction':prediction,'baseline':baseline.predict(X[test])})


for feature in range(3):
    output[f'contribution_{feature}']=contribution_matrix[:,feature]
save_table(output,'additive_predictions')
save_table(curves,'feature_response_curves')
save_model({'model':model,'reference':reference,'reference_prediction':reference_prediction},'additive_spline_model')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for feature,ax in enumerate(axes):
    frame=curves.loc[curves.feature.eq(feature)]
    ax.plot(frame.value,frame.estimated_centered_effect)
    ax.plot(frame.value,frame.true_centered_effect,linestyle='--')
save_figure(fig,'additive_nonlinear_effects')
finish({'heldout_rmse':float(np.sqrt(mean_squared_error(y[test],prediction))),'linear_rmse':float(np.sqrt(mean_squared_error(y[test],baseline.predict(X[test])))),'selected_regularization':float(model[-1].alpha_),'exact_additive_decomposition_verified':True},[output])
