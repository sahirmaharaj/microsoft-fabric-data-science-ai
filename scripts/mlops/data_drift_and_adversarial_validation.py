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
WORKFLOW = "45_data_drift_and_adversarial_validation"


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


from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

reference=pd.DataFrame(rng.normal(size=(SAMPLE_SIZE,8)),columns=[f"feature_{i}" for i in range(8)])
current=pd.DataFrame(rng.normal(size=(SAMPLE_SIZE,8)),columns=reference.columns)
current["feature_0"]+=0.7
current["feature_1"]*=1.7
current.loc[rng.choice(SAMPLE_SIZE,100,replace=False),"feature_2"]=np.nan

def population_stability(reference_values,current_values,bins=10):
    baseline=np.asarray(reference_values.dropna())
    observed=np.asarray(current_values.dropna())
    edges=np.unique(np.quantile(baseline,np.linspace(0,1,bins+1)))
    edges[0],edges[-1]=-np.inf,np.inf
    baseline_counts=np.histogram(baseline,edges)[0]+0.5
    observed_counts=np.histogram(observed,edges)[0]+0.5
    baseline_probability=baseline_counts/baseline_counts.sum()
    observed_probability=observed_counts/observed_counts.sum()
    contributions=(observed_probability-baseline_probability)*np.log(observed_probability/baseline_probability)
    return float(contributions.sum()),pd.DataFrame({"bin":np.arange(len(contributions)),"reference_probability":baseline_probability,"current_probability":observed_probability,"psi_contribution":contributions})

rows=[]
bin_rows=[]
for column in reference:
    psi,bins=population_stability(reference[column],current[column])
    test=ks_2samp(reference[column].dropna(),current[column].dropna())
    rows.append({"feature":column,"psi":psi,"ks_statistic":test.statistic,"pvalue":test.pvalue,"reference_missing":reference[column].isna().mean(),"current_missing":current[column].isna().mean()})
    bins["feature"]=column
    bin_rows.append(bins)
metrics=pd.DataFrame(rows)
metrics["bonferroni_pvalue"]=np.minimum(metrics.pvalue*len(metrics),1)
metrics["alert"]=(metrics.psi>0.2)|((metrics.bonferroni_pvalue<0.05)&(metrics.ks_statistic>0.1))|(metrics.current_missing-metrics.reference_missing>0.05)
combined=pd.concat([reference,current],ignore_index=True)


labels=np.r_[np.zeros(len(reference)),np.ones(len(current))]
X_train,X_test,y_train,y_test=train_test_split(combined,labels,stratify=labels,test_size=0.3,random_state=SEED)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
adversary=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),RandomForestClassifier(n_estimators=120,max_depth=6,min_samples_leaf=10,n_jobs=2,random_state=SEED))
adversary.fit(X_train,y_train)
p=adversary.predict_proba(X_test)[:,1]
auc=roc_auc_score(y_test,p)
bootstrap=[]
for iteration in range(300):
    indices=rng.integers(0,len(y_test),len(y_test))
    bootstrap.append(roc_auc_score(y_test[indices],p[indices]))
interval=np.quantile(bootstrap,[0.025,0.975])
save_table(metrics,"feature_drift")
save_table(pd.concat(bin_rows,ignore_index=True),"psi_bins")
save_table(pd.DataFrame({"domain":y_test,"current_probability":p}),"adversarial_predictions")
save_model(adversary,"drift_discriminator")
save_json({"reference_rows":len(reference),"current_rows":len(current),"adversarial_auc":auc,"auc_interval":interval,"reference_quantiles":reference.quantile([0,0.1,0.5,0.9,1]).to_dict()},"drift_baseline")
fig,axes=plt.subplots(1,2,figsize=(11,4))
metrics.plot.bar(x="feature",y="psi",ax=axes[0],legend=False)
axes[1].hist(p[y_test==0],bins=20,alpha=0.5,label="Reference")
axes[1].hist(p[y_test==1],bins=20,alpha=0.5,label="Current")
axes[1].legend()
save_figure(fig,"drift_monitoring")
result=finish({"drifted_features":int(metrics.alert.sum()),"adversarial_auc":auc,"auc_lower":interval[0],"auc_upper":interval[1]},[metrics])
