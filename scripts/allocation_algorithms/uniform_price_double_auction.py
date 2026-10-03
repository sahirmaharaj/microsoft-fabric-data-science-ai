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
WORKFLOW = "152_uniform_price_double_auction"


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


buyers=[]
sellers=[]
for buyer in range(18):
    base=rng.uniform(35,90)
    quantity=int(rng.integers(1,5))
    for unit in range(quantity):
        buyers.append({'participant':buyer,'unit':unit,'value':float(base-unit*rng.uniform(2,6))})
for seller in range(15):
    base=rng.uniform(15,65)
    quantity=int(rng.integers(1,5))
    for unit in range(quantity):
        sellers.append({'participant':seller,'unit':unit,'cost':float(base+unit*rng.uniform(1,5))})

def clear_market(demand_multiplier=1.):
    demand=sorted([{**row,'value':row['value']*demand_multiplier} for row in buyers],key=lambda row:(-row['value'],row['participant'],row['unit']))
    supply=sorted(sellers,key=lambda row:(row['cost'],row['participant'],row['unit']))
    quantity=0
    for bid,ask in zip(demand,supply):
        if bid['value']<ask['cost']:
            break
        quantity+=1
    if quantity==0:
        return [],{'quantity':0,'price':None,'total_surplus':0.}
    price=(demand[quantity-1]['value']+supply[quantity-1]['cost'])/2
    trades=[]
    for bid,ask in zip(demand[:quantity],supply[:quantity]):
        trades.append({'buyer':bid['participant'],'buyer_unit':bid['unit'],'seller':ask['participant'],'seller_unit':ask['unit'],'bid':bid['value'],'ask':ask['cost'],'price':price,'buyer_surplus':bid['value']-price,'seller_surplus':price-ask['cost']})
    assert all(row['buyer_surplus']>=-1e-10 and row['seller_surplus']>=-1e-10 for row in trades)
    surplus=sum(row['bid']-row['ask'] for row in trades)
    theoretical=sum(max(bid['value']-ask['cost'],0) for bid,ask in zip(demand,supply))
    assert np.isclose(surplus,theoretical)
    return trades,{'quantity':quantity,'price':price,'total_surplus':surplus}

trades,metrics=clear_market()
output=pd.DataFrame(trades)


scenarios=[]
for multiplier in [.6,.8,1,1.2,1.4]:
    _,result=clear_market(multiplier)
    scenarios.append({'demand_multiplier':multiplier,**result})
buyer_summary=output.groupby('buyer').agg(units=('buyer_unit','size'),payment=('price','sum'),surplus=('buyer_surplus','sum')).reset_index()
seller_summary=output.groupby('seller').agg(units=('seller_unit','size'),revenue=('price','sum'),surplus=('seller_surplus','sum')).reset_index()
assert np.isclose(buyer_summary.payment.sum(),seller_summary.revenue.sum())
save_table(output,'cleared_trades')
save_table(buyer_summary,'buyer_settlement')
save_table(seller_summary,'seller_settlement')
save_table(pd.DataFrame(scenarios),'market_demand_scenarios')
save_table(pd.DataFrame(buyers),'submitted_unit_bids')
save_table(pd.DataFrame(sellers),'submitted_unit_asks')
fig,ax=plt.subplots(figsize=(9,5))
ax.step(range(len(buyers)),sorted([row['value'] for row in buyers],reverse=True),label='bids')
ax.step(range(len(sellers)),sorted([row['cost'] for row in sellers]),label='asks')
ax.axhline(metrics['price'],color='gray')
ax.legend()
save_figure(fig,'auction_supply_demand')
finish({**metrics,'budget_balanced':True,'truthful_bidding_guaranteed':False,'settlement_is_synthetic':True},[pd.DataFrame(scenarios)])
