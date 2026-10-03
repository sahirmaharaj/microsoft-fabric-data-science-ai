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
WORKFLOW = "27_survival_analysis_and_retention"


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


from scipy.stats import chi2

n = SAMPLE_SIZE
group = rng.integers(0, 2, n)
tenure_score = rng.normal(size=n)
hazard = 0.025 * np.exp(-0.45*group + 0.25*tenure_score)
true_time = rng.exponential(1/hazard)
censor_time = rng.uniform(10, 70, n)
data = pd.DataFrame({"duration": np.minimum(true_time,censor_time), "event": (true_time<=censor_time).astype(int), "group": group, "score": tenure_score})

def kaplan_meier(frame):
    survival, greenwood = 1.0, 0.0
    rows = [{"time": 0.0, "survival": 1.0, "lower": 1.0, "upper": 1.0, "at_risk": len(frame), "events": 0}]
    for event_time in np.sort(frame.loc[frame.event.eq(1), "duration"].unique()):
        at_risk = int(frame.duration.ge(event_time).sum())
        events = int((frame.duration.eq(event_time)&frame.event.eq(1)).sum())
        survival *= 1-events/at_risk
        if at_risk > events:
            greenwood += events/(at_risk*(at_risk-events))
        error = survival*np.sqrt(greenwood)
        rows.append({"time": event_time, "survival": survival, "lower": max(0,survival-1.96*error), "upper": min(1,survival+1.96*error), "at_risk": at_risk, "events": events})
    return pd.DataFrame(rows)

curves = []
for label, frame in data.groupby("group"):
    curve = kaplan_meier(frame)
    curve["group"] = label
    curves.append(curve)
curves = pd.concat(curves,ignore_index=True)
observed_minus_expected, variance = 0.0, 0.0
for event_time in np.sort(data.loc[data.event.eq(1),"duration"].unique()):
    risk = data.loc[data.duration.ge(event_time)]
    deaths = data.loc[data.duration.eq(event_time)&data.event.eq(1)]
    total, n1, d, d1 = len(risk), int(risk.group.eq(1).sum()), len(deaths), int(deaths.group.eq(1).sum())
    observed_minus_expected += d1-d*n1/total
    if total>1:
        variance += n1*(total-n1)*d*(total-d)/(total*total*(total-1))


statistic = observed_minus_expected**2/max(variance,1e-12)
retention_rows=[]
for label, curve in curves.groupby("group"):
    for day in [7,14,30,45,60]:
        retained=curve.loc[curve.time.le(day)].iloc[-1]
        retention_rows.append({"group":label,"day":day,"retention":retained.survival,"lower":retained.lower,"upper":retained.upper})
retention=pd.DataFrame(retention_rows)
bootstrap_differences=[]
for iteration in range(250):
    probabilities=[]
    for label in [0,1]:
        subset=data.loc[data.group.eq(label)]
        sample=subset.iloc[rng.integers(0,len(subset),len(subset))]
        curve=kaplan_meier(sample)
        probabilities.append(float(curve.loc[curve.time.le(30)].iloc[-1].survival))
    bootstrap_differences.append(probabilities[1]-probabilities[0])
interval=np.quantile(bootstrap_differences,[0.025,0.975])
save_table(data,"survival_data")
save_table(curves,"kaplan_meier")
save_table(retention,"retention_estimates")
fig,ax=plt.subplots(figsize=(10,4))
for label,curve in curves.groupby("group"):
    ax.step(curve.time,curve.survival,where="post",label=f"group_{label}")
    ax.fill_between(curve.time,curve.lower,curve.upper,alpha=0.15,step="post")
ax.legend()
ax.set(xlabel="Duration",ylabel="Survival")
save_figure(fig,"survival_curves")
result=finish({"events":int(data.event.sum()),"censoring_rate":float(1-data.event.mean()),"logrank_pvalue":float(chi2.sf(statistic,1)),"day30_difference_lower":interval[0],"day30_difference_upper":interval[1]},[retention])
