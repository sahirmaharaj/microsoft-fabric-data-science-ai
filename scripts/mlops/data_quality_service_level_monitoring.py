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
WORKFLOW = "49_data_quality_service_level_monitoring"


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


from scipy.stats import beta

batch_count=90
rows=[]
for day in range(batch_count):
    total=int(rng.integers(800,1400))
    invalid_rate=0.007 if day<60 else 0.035
    if day in [42,76]:
        invalid_rate=0.12
    rows.append({"date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=day),"rows":total,"invalid":int(rng.binomial(total,invalid_rate)),"duplicates":int(rng.binomial(total,0.002)),"freshness_minutes":float(rng.gamma(2,7))+(50 if day>75 else 0),"runtime_seconds":float(rng.lognormal(3.5,0.15))})
batches=pd.DataFrame(rows)
batches["valid"]=batches.rows-batches.invalid
batches["invalid_rate"]=batches.invalid/batches.rows
batches["duplicate_rate"]=batches.duplicates/batches.rows
baseline=batches.iloc[:30]
mean_rate=baseline.invalid.sum()/baseline.rows.sum()
batches["upper_control_limit"]=mean_rate+3*np.sqrt(mean_rate*(1-mean_rate)/batches.rows)
batches["quality_breach"]=batches.invalid_rate>batches.upper_control_limit
batches["freshness_breach"]=batches.freshness_minutes>60
batches["availability_slo"]=~(batches.quality_breach|batches.freshness_breach)
batches["posterior_invalid_mean"]=(batches.invalid+1)/(batches.rows+2)
batches["posterior_invalid_p95"]=beta.ppf(0.95,batches.invalid+1,batches.valid+1)
batches["rolling_7d_invalid_rate"]=batches.invalid.rolling(7,min_periods=1).sum()/batches.rows.rolling(7,min_periods=1).sum()
allowed_invalid_rate=0.02
batches["burn_rate"]=batches.rolling_7d_invalid_rate/allowed_invalid_rate
batches["paging_alert"]=batches.burn_rate.gt(2)&batches.burn_rate.shift(1,fill_value=0).gt(2)
incident_ids=[]
incident=0
previous=False
for breached in (batches.quality_breach|batches.freshness_breach):
    if breached and not previous:
        incident+=1
    incident_ids.append(incident if breached else 0)
    previous=bool(breached)
batches["incident_id"]=incident_ids


incidents=batches.loc[batches.incident_id.gt(0)].groupby("incident_id").agg(start=("date","min"),end=("date","max"),days=("date","size"),invalid_rows=("invalid","sum"),max_freshness=("freshness_minutes","max")).reset_index()
monthly=batches.groupby(batches.date.dt.to_period("M").astype(str)).agg(rows=("rows","sum"),invalid=("invalid","sum"),slo_attainment=("availability_slo","mean"),freshness_p95=("freshness_minutes",lambda values:values.quantile(0.95))).reset_index()
monthly["row_validity"]=1-monthly.invalid/monthly.rows
save_table(batches,"batch_quality_history")
save_table(incidents,"incident_register")
save_table(monthly,"monthly_service_levels")
save_json({"baseline_invalid_rate":mean_rate,"allowed_invalid_rate":allowed_invalid_rate,"freshness_limit_minutes":60,"baseline_batches":30},"monitoring_policy")
assert batches.valid.add(batches.invalid).equals(batches.rows)
assert batches.posterior_invalid_p95.between(0,1).all()
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
axes[0].plot(batches.date,batches.invalid_rate,label="Invalid rate")
axes[0].plot(batches.date,batches.upper_control_limit,label="Control limit")
axes[0].legend()
axes[1].plot(batches.date,batches.freshness_minutes)
axes[1].axhline(60,color="red")
save_figure(fig,"quality_monitoring")
result=finish({"batches":len(batches),"incidents":len(incidents),"slo_attainment":float(batches.availability_slo.mean()),"invalid_rows":int(batches.invalid.sum()),"paging_days":int(batches.paging_alert.sum())},[monthly,incidents])
