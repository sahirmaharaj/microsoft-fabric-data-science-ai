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
WORKFLOW = "50_capacity_planning_and_resource_allocation"


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


from scipy.optimize import linprog

products=["analytics","engineering","ai_models","support"]
resources=["analyst_hours","engineer_hours","compute_units"]
requirements=np.array([[3,1,2,1],[1,4,3,0.5],[1,2,5,0.5]],dtype=float)
capacity=np.array([160,200,180],dtype=float)
contribution=np.array([500,800,1100,220],dtype=float)
minimum_demand=np.array([8,10,6,12],dtype=float)
maximum_demand=np.array([50,40,30,60],dtype=float)
bounds=list(zip(minimum_demand,maximum_demand))

@dataclass
class AllocationResult:
    quantity: np.ndarray
    contribution: float
    utilization: np.ndarray
    shadow_prices: np.ndarray

def optimize_allocation(available):
    solution=linprog(-contribution,A_ub=requirements,b_ub=available,bounds=bounds,method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    return AllocationResult(solution.x,float(contribution@solution.x),requirements@solution.x,-np.asarray(solution.ineqlin.marginals))

baseline=optimize_allocation(capacity)
allocation=pd.DataFrame({"work_type":products,"quantity":baseline.quantity,"unit_contribution":contribution,"total_contribution":baseline.quantity*contribution,"minimum":minimum_demand,"maximum":maximum_demand})
resource_table=pd.DataFrame({"resource":resources,"capacity":capacity,"used":baseline.utilization,"slack":capacity-baseline.utilization,"utilization":baseline.utilization/capacity,"shadow_price":baseline.shadow_prices})
scenarios=[]
for resource_index,resource in enumerate(resources):
    for increase in [0,10,20,40]:
        scenario_capacity=capacity.copy()
        scenario_capacity[resource_index]+=increase
        solution=optimize_allocation(scenario_capacity)
        scenarios.append({"resource":resource,"additional_units":increase,"total_contribution":solution.contribution,"incremental_contribution":solution.contribution-baseline.contribution})
scenarios=pd.DataFrame(scenarios)


robust_capacity=capacity*0.9
robust=optimize_allocation(robust_capacity)
simulation=[]
for iteration in range(1000):
    realized_capacity=capacity*rng.uniform(0.85,1.05,len(resources))
    baseline_shortage=np.maximum(baseline.utilization-realized_capacity,0)
    robust_shortage=np.maximum(robust.utilization-realized_capacity,0)
    simulation.append({"simulation":iteration,"baseline_feasible":bool((baseline_shortage<=1e-8).all()),"robust_feasible":bool((robust_shortage<=1e-8).all()),"baseline_shortage":float(baseline_shortage.sum()),"robust_shortage":float(robust_shortage.sum())})
simulation=pd.DataFrame(simulation)
integer_quantity=np.floor(baseline.quantity)
assert (requirements@integer_quantity<=capacity+1e-8).all()
assert (baseline.quantity>=minimum_demand-1e-8).all()
assert (baseline.utilization<=capacity+1e-8).all()
allocation["rounded_down_quantity"]=integer_quantity
save_table(allocation,"optimal_allocation")
save_table(resource_table,"resource_utilization")
save_table(scenarios,"capacity_sensitivity")
save_table(simulation,"capacity_uncertainty")
save_json({"requirements":requirements,"products":products,"resources":resources,"continuous_objective":baseline.contribution,"robust_objective":robust.contribution,"rounded_down_objective":float(contribution@integer_quantity)},"optimization_model")
fig,axes=plt.subplots(1,2,figsize=(12,4))
allocation.plot.bar(x="work_type",y="quantity",ax=axes[0],legend=False)
for resource,frame in scenarios.groupby("resource"):
    axes[1].plot(frame.additional_units,frame.incremental_contribution,label=resource)
axes[1].legend()
save_figure(fig,"resource_planning")
result=finish({"optimal_contribution":baseline.contribution,"robust_contribution":robust.contribution,"baseline_feasibility_rate":float(simulation.baseline_feasible.mean()),"robust_feasibility_rate":float(simulation.robust_feasible.mean())},[allocation,resource_table])
