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
WORKFLOW = "29_bootstrap_permutation_and_multiple_testing"


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


from scipy.stats import ttest_ind

n = 700
control = rng.lognormal(3,0.8,n)
variant = rng.lognormal(3.08,0.8,n)
observed_difference=float(variant.mean()-control.mean())
bootstrap=[]
for iteration in range(1500):
    a=rng.choice(control,n,replace=True)
    b=rng.choice(variant,n,replace=True)
    bootstrap.append({"iteration":iteration,"mean_difference":float(b.mean()-a.mean()),"median_difference":float(np.median(b)-np.median(a)),"relative_lift":float(b.mean()/a.mean()-1)})
bootstrap=pd.DataFrame(bootstrap)
pooled=np.r_[control,variant]
permuted=[]
for iteration in range(1500):
    shuffled=rng.permutation(pooled)
    permuted.append(float(shuffled[n:].mean()-shuffled[:n].mean()))
permutation_p=(1+np.sum(np.abs(permuted)>=abs(observed_difference)))/(1+len(permuted))
metric_rows=[]
for index in range(24):
    baseline=rng.normal(size=(n,))
    comparison=rng.normal(0.25 if index<5 else 0,1,n)
    test=ttest_ind(comparison,baseline,equal_var=False)
    metric_rows.append({"metric":f"metric_{index:02d}","effect":float(comparison.mean()-baseline.mean()),"pvalue":float(test.pvalue),"synthetic_effect_present":index<5})
metrics=pd.DataFrame(metric_rows)
order=np.argsort(metrics.pvalue.to_numpy())
sorted_p=metrics.pvalue.to_numpy()[order]
scaled=sorted_p*len(metrics)/np.arange(1,len(metrics)+1)
adjusted_sorted=np.minimum.accumulate(scaled[::-1])[::-1]
adjusted=np.empty(len(metrics))
adjusted[order]=np.minimum(adjusted_sorted,1)
metrics["fdr_adjusted_pvalue"]=adjusted
metrics["reject_fdr_005"]=metrics.fdr_adjusted_pvalue<=0.05
metrics["reject_bonferroni_005"]=metrics.pvalue<=0.05/len(metrics)
intervals=[]


for column in ["mean_difference","median_difference","relative_lift"]:
    lower,upper=bootstrap[column].quantile([0.025,0.975])
    intervals.append({"metric":column,"lower":lower,"upper":upper,"bootstrap_mean":bootstrap[column].mean()})
intervals=pd.DataFrame(intervals)
save_table(bootstrap,"bootstrap_draws")
save_table(pd.DataFrame({"permuted_difference":permuted}),"permutation_null")
save_table(metrics,"multiple_testing")
save_table(intervals,"confidence_intervals")
assert metrics.fdr_adjusted_pvalue.between(0,1).all()
assert np.all(np.diff(adjusted[order])>=-1e-12)
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(permuted,bins=35)
axes[0].axvline(observed_difference,color="red")
axes[1].scatter(metrics.pvalue,metrics.fdr_adjusted_pvalue)
axes[1].set(xlabel="Raw p-value",ylabel="FDR adjusted p-value")
save_figure(fig,"statistical_inference")
result=finish({"observed_mean_difference":observed_difference,"permutation_pvalue":float(permutation_p),"fdr_discoveries":int(metrics.reject_fdr_005.sum()),"bonferroni_discoveries":int(metrics.reject_bonferroni_005.sum())},[intervals,metrics])
