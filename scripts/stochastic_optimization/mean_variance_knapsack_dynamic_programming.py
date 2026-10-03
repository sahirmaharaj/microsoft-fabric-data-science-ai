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
WORKFLOW = "156_mean_variance_knapsack_dynamic_programming"


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


items=45
budget=160
cost=rng.integers(4,25,items)
mean=rng.uniform(8,45,items)
standard_deviation=rng.uniform(2,15,items)

def optimize(risk_aversion):
    utility=mean-risk_aversion*standard_deviation**2
    table=np.zeros((items+1,budget+1))
    take=np.zeros((items+1,budget+1),dtype=bool)
    for item in range(1,items+1):
        table[item]=table[item-1]
        weight=cost[item-1]
        candidate=table[item-1,:budget+1-weight]+utility[item-1]
        improve=candidate>table[item,weight:]+1e-12
        table[item,weight:]=np.where(improve,candidate,table[item,weight:])
        take[item,weight:]=improve
    remaining=budget
    chosen=[]
    for item in range(items,0,-1):
        if take[item,remaining]:
            chosen.append(item-1)
            remaining-=int(cost[item-1])
    chosen=sorted(chosen)
    assert sum(cost[chosen])<=budget
    assert np.isclose(sum(utility[chosen]),table[-1,budget])
    return chosen,float(table[-1,budget])

outcomes=rng.normal(mean,standard_deviation,(12000,items))
policies=[]
assignments=[]
for risk_aversion in [0,.02,.05,.1,.2]:
    selected,objective=optimize(risk_aversion)
    total=outcomes[:,selected].sum(axis=1)
    policies.append({'risk_aversion':risk_aversion,'selected_items':len(selected),'used_budget':int(cost[selected].sum()),'objective':objective,'analytic_mean':float(mean[selected].sum()),'analytic_sd':float(np.sqrt(np.sum(standard_deviation[selected]**2))),'simulated_mean':float(total.mean()),'simulated_p05':float(np.quantile(total,.05)),'simulated_sd':float(total.std(ddof=1))})
    for item in selected:
        assignments.append({'risk_aversion':risk_aversion,'item':item,'cost':int(cost[item]),'mean_payoff':mean[item],'payoff_sd':standard_deviation[item]})


policy_table=pd.DataFrame(policies)
item_table=pd.DataFrame({'item':range(items),'cost':cost,'mean_payoff':mean,'payoff_sd':standard_deviation})
save_table(policy_table,'knapsack_risk_policies')
save_table(pd.DataFrame(assignments),'selected_items_by_policy')
save_table(item_table,'candidate_items')
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(policy_table.analytic_sd,policy_table.analytic_mean,s=70)
for row in policies:
    ax.annotate(str(row['risk_aversion']),(row['analytic_sd'],row['analytic_mean']))
ax.set(xlabel='Aggregate payoff standard deviation',ylabel='Expected aggregate payoff')
save_figure(fig,'knapsack_mean_variance_tradeoff')
finish({'items':items,'budget':budget,'policies':len(policies),'independent_item_payoffs_assumed':True,'dynamic_program_objectives_verified':True},[policy_table])
