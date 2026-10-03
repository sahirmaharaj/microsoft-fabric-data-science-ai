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
WORKFLOW = "155_risk_averse_newsvendor_cvar"


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


training_demand=np.maximum(0,rng.normal(110,25,1500)+rng.binomial(1,.08,1500)*70)
heldout_demand=np.maximum(0,rng.normal(110,25,6000)+rng.binomial(1,.08,6000)*70)
underage=9.
overage=3.
alpha=.9
quantities=np.arange(40,241)

def cost_matrix(demand):
    return overage*np.maximum(quantities[:,None]-demand[None,:],0)+underage*np.maximum(demand[None,:]-quantities[:,None],0)

def empirical_cvar(costs,confidence):
    threshold=np.quantile(costs,confidence,axis=1)
    return threshold+np.maximum(costs-threshold[:,None],0).mean(axis=1)/(1-confidence)

training_cost=cost_matrix(training_demand)
expected=training_cost.mean(axis=1)
tail=empirical_cvar(training_cost,alpha)
curves=pd.DataFrame({'quantity':quantities,'expected_cost':expected,'cvar90':tail})
heldout_cost=cost_matrix(heldout_demand)
policies=[]
for risk_weight in [0,.25,.5,.75,1]:
    objective=(1-risk_weight)*expected+risk_weight*tail
    index=int(np.argmin(objective))
    evaluation=heldout_cost[index]
    value_at_risk=float(np.quantile(evaluation,alpha))
    tail_cost=value_at_risk+float(np.maximum(evaluation-value_at_risk,0).mean())/(1-alpha)
    policies.append({'risk_weight':risk_weight,'quantity':int(quantities[index]),'training_objective':float(objective[index]),'heldout_expected_cost':float(evaluation.mean()),'heldout_cvar90':tail_cost,'heldout_service_level':float((heldout_demand<=quantities[index]).mean())})
critical_fractile=underage/(underage+overage)
analytic_quantity=float(np.quantile(training_demand,critical_fractile))
bootstrap=[]
for iteration in range(180):
    sample=rng.choice(training_demand,len(training_demand),replace=True)
    bootstrap.append(float(np.quantile(sample,critical_fractile)))
assert abs(policies[0]['quantity']-analytic_quantity)<2
assert np.all(tail>=expected)


save_table(curves,'quantity_cost_tradeoffs')
save_table(pd.DataFrame(policies),'out_of_sample_inventory_policies')
save_table(pd.DataFrame({'bootstrap_risk_neutral_quantity':bootstrap}),'order_quantity_sampling_uncertainty')
save_table(pd.DataFrame({'training_demand':training_demand}),'training_demand_scenarios')
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(quantities,expected,label='expected cost')
ax.plot(quantities,tail,label='CVaR 90')
ax.axvline(analytic_quantity,color='gray',linestyle='--')
ax.legend()
save_figure(fig,'newsvendor_risk_tradeoff')
finish({'critical_fractile':critical_fractile,'empirical_quantile_order':analytic_quantity,'quantity_interval':np.quantile(bootstrap,[.025,.975]),'evaluation_scenarios':len(heldout_demand),'test_demand_used_for_policy_selection':False},[pd.DataFrame(policies)])
