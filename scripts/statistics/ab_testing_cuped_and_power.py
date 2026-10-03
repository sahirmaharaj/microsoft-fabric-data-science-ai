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
WORKFLOW = "28_ab_testing_cuped_and_power"


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


from scipy.stats import ttest_ind, norm

n = SAMPLE_SIZE * 4
treatment = rng.integers(0, 2, n)
preperiod = rng.gamma(3, 20, n)
noise = rng.normal(0, 12, n)
outcome = 25 + 0.7 * preperiod + 3 * treatment + noise
frame = pd.DataFrame({"treatment": treatment, "preperiod": preperiod, "outcome": outcome})
theta = np.cov(frame.preperiod, frame.outcome, ddof=1)[0,1] / np.var(frame.preperiod, ddof=1)
frame["cuped"] = frame.outcome-theta*(frame.preperiod-frame.preperiod.mean())

def difference_analysis(frame, column):
    control = frame.loc[frame.treatment.eq(0),column].to_numpy()
    variant = frame.loc[frame.treatment.eq(1),column].to_numpy()
    effect = variant.mean()-control.mean()
    standard_error = np.sqrt(variant.var(ddof=1)/len(variant)+control.var(ddof=1)/len(control))
    test = ttest_ind(variant,control,equal_var=False)
    return {"metric":column,"control_mean":control.mean(),"treatment_mean":variant.mean(),"effect":effect,"standard_error":standard_error,"lower":effect-1.96*standard_error,"upper":effect+1.96*standard_error,"pvalue":test.pvalue}

results = pd.DataFrame([difference_analysis(frame,column) for column in ["outcome","cuped"]])
variance_reduction = 1-frame.cuped.var()/frame.outcome.var()
power_rows=[]
for minimum_effect in [1,2,3,4,5]:
    for power in [0.8,0.9]:
        for metric in ["outcome","cuped"]:
            variance=float(frame[metric].var())
            per_arm=int(np.ceil(2*variance*(norm.ppf(0.975)+norm.ppf(power))**2/minimum_effect**2))
            power_rows.append({"metric":metric,"minimum_effect":minimum_effect,"power":power,"required_per_arm":per_arm})
power_table=pd.DataFrame(power_rows)
srm_counts=np.bincount(treatment,minlength=2)
from scipy.stats import chisquare
srm_pvalue=float(chisquare(srm_counts).pvalue)
bootstrap=[]
for _ in range(400):
    values=[]
    for arm in [0,1]:
        arm_values=frame.loc[frame.treatment.eq(arm),"cuped"].to_numpy()
        values.append(float(rng.choice(arm_values,len(arm_values),replace=True).mean()))
    bootstrap.append(values[1]-values[0])


bootstrap_interval=np.quantile(bootstrap,[0.025,0.975])
save_table(frame,"experiment_data")
save_table(results,"effect_estimates")
save_table(power_table,"sample_size_planning")
save_json({"theta":theta,"srm_counts":srm_counts,"srm_pvalue":srm_pvalue,"bootstrap_interval":bootstrap_interval},"diagnostics")
assert abs(results.iloc[0].effect-results.iloc[1].effect)<10
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].scatter(frame.preperiod.iloc[:500],frame.outcome.iloc[:500],s=10,alpha=0.5)
for metric,subset in power_table.loc[power_table.power.eq(0.8)].groupby("metric"):
    axes[1].plot(subset.minimum_effect,subset.required_per_arm,label=metric)
axes[1].legend()
save_figure(fig,"experiment_analysis")
result=finish({"cuped_effect":float(results.iloc[1].effect),"cuped_pvalue":float(results.iloc[1].pvalue),"variance_reduction":float(variance_reduction),"sample_ratio_pvalue":srm_pvalue},[results,power_table])
