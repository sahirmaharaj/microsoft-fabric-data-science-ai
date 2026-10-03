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
WORKFLOW = "30_cross_fitted_doubly_robust_estimation"


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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

n=SAMPLE_SIZE*2
X=rng.normal(size=(n,6))
true_propensity=expit(0.5*X[:,0]-0.4*X[:,1])
treatment=rng.binomial(1,true_propensity)
true_effect=2+0.5*X[:,2]
base=3*X[:,0]+2*X[:,1]+X[:,3]**2
outcome=base+treatment*true_effect+rng.normal(0,1,n)
propensity=np.zeros(n)
mu0=np.zeros(n)
mu1=np.zeros(n)
fold_id=np.zeros(n,dtype=int)
folds=StratifiedKFold(5,shuffle=True,random_state=SEED)
for fold,(train,test) in enumerate(folds.split(X,treatment)):
    prop_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
    prop_model.fit(X[train],treatment[train])
    propensity[test]=np.clip(prop_model.predict_proba(X[test])[:,1],0.05,0.95)
    for arm,output in [(0,mu0),(1,mu1)]:
        subset=train[treatment[train]==arm]
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=12,l2_regularization=2,random_state=SEED)
        model.fit(X[subset],outcome[subset])
        output[test]=model.predict(X[test])
    fold_id[test]=fold
influence=mu1-mu0+treatment*(outcome-mu1)/propensity-(1-treatment)*(outcome-mu0)/(1-propensity)
ate=float(influence.mean())
standard_error=float(influence.std(ddof=1)/np.sqrt(n))
naive=float(outcome[treatment==1].mean()-outcome[treatment==0].mean())
ipw=float(np.mean(treatment*outcome/propensity-(1-treatment)*outcome/(1-propensity)))
weights=treatment/propensity+(1-treatment)/(1-propensity)


effective_sample_size=float(weights.sum()**2/np.square(weights).sum())
output=pd.DataFrame({"treatment":treatment,"outcome":outcome,"propensity":propensity,"mu0":mu0,"mu1":mu1,"aipw_score":influence,"fold":fold_id,"true_effect":true_effect})
output["effect_stratum"]=pd.qcut(X[:,2],4,labels=False)
strata=output.groupby("effect_stratum").agg(estimated_effect=("aipw_score","mean"),true_effect=("true_effect","mean"),rows=("outcome","size")).reset_index()
balance=[]
for index in range(X.shape[1]):
    treated_mean=np.average(X[treatment==1,index],weights=weights[treatment==1])
    control_mean=np.average(X[treatment==0,index],weights=weights[treatment==0])
    balance.append({"feature":index,"raw_smd":(X[treatment==1,index].mean()-X[treatment==0,index].mean())/X[:,index].std(),"weighted_smd":(treated_mean-control_mean)/X[:,index].std()})
balance=pd.DataFrame(balance)
save_table(output,"cross_fitted_scores")
save_table(strata,"effect_heterogeneity")
save_table(balance,"covariate_balance")
save_json({"estimate":ate,"standard_error":standard_error,"lower":ate-1.96*standard_error,"upper":ate+1.96*standard_error,"naive":naive,"ipw":ipw,"synthetic_truth":float(true_effect.mean())},"treatment_effect")
assert np.isfinite(influence).all()
assert np.all((propensity>=0.05)&(propensity<=0.95))
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(propensity[treatment==0],bins=25,alpha=0.5,label="Control")
axes[0].hist(propensity[treatment==1],bins=25,alpha=0.5,label="Treated")
axes[0].legend()
balance.plot.bar(x="feature",y=["raw_smd","weighted_smd"],ax=axes[1])
save_figure(fig,"causal_diagnostics")
result=finish({"aipw_ate":ate,"true_ate":float(true_effect.mean()),"naive_difference":naive,"ci_lower":ate-1.96*standard_error,"ci_upper":ate+1.96*standard_error,"effective_sample_size":effective_sample_size},[strata,balance])
