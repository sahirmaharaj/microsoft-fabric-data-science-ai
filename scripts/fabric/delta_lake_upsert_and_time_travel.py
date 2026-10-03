import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4', 'pyarrow': 'pyarrow>=14,<30', 'deltalake': 'deltalake>=0.25,<2'}
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
WORKFLOW = "54_delta_lake_upsert_and_time_travel"


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


import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

initial=pd.DataFrame({"customer_id":np.arange(100,dtype=np.int64),"segment":rng.choice(["bronze","silver","gold"],100),"balance":rng.uniform(0,1000,100).round(2),"update_sequence":np.ones(100,dtype=np.int64)})
schema=pa.schema([("customer_id",pa.int64()),("segment",pa.string()),("balance",pa.float64()),("update_sequence",pa.int64())])
table_path=working_dir/"customer_delta"
write_deltalake(str(table_path),pa.Table.from_pandas(initial,schema=schema,preserve_index=False),mode="error")
initial_version=DeltaTable(str(table_path)).version()
updates=initial.iloc[20:45].copy()
updates["balance"]+=150
updates["segment"]="gold"
updates["update_sequence"]=2
new_rows=pd.DataFrame({"customer_id":np.arange(100,115,dtype=np.int64),"segment":["silver"]*15,"balance":rng.uniform(100,400,15).round(2),"update_sequence":np.ones(15,dtype=np.int64)})
stale=initial.iloc[20:25].copy()
stale["balance"]=-999.0
source=pd.concat([updates,new_rows,stale],ignore_index=True)
source=source.sort_values("update_sequence").drop_duplicates("customer_id",keep="last")

def merge_source(source_frame):
    table=DeltaTable(str(table_path))
    source_arrow=pa.Table.from_pandas(source_frame,schema=schema,preserve_index=False)
    return (
        table.merge(source=source_arrow,predicate="target.customer_id = source.customer_id",source_alias="source",target_alias="target")
        .when_matched_update_all(predicate="source.update_sequence > target.update_sequence")
        .when_not_matched_insert_all()
        .execute()
    )

first_merge=merge_source(source)
after_first=DeltaTable(str(table_path)).to_pandas().sort_values("customer_id").reset_index(drop=True)
second_merge=merge_source(source)
after_second=DeltaTable(str(table_path)).to_pandas().sort_values("customer_id").reset_index(drop=True)
historical=DeltaTable(str(table_path),version=initial_version).to_pandas().sort_values("customer_id").reset_index(drop=True)
pd.testing.assert_frame_equal(after_first,after_second)
pd.testing.assert_frame_equal(historical,initial.sort_values("customer_id").reset_index(drop=True))


assert len(after_second)==115
assert after_second.customer_id.is_unique
assert after_second.balance.ge(0).all()
assert after_second.loc[after_second.customer_id.between(20,44),"update_sequence"].eq(2).all()
current_table=DeltaTable(str(table_path))
history=pd.DataFrame(current_table.history())
comparison=initial.merge(after_second,on="customer_id",suffixes=("_before","_after"),how="outer")
comparison["change_type"]=np.where(comparison.balance_before.isna(),"insert",np.where(comparison.balance_before.eq(comparison.balance_after),"unchanged","update"))
save_table(after_second,"current_snapshot")
save_table(historical,"version_zero_snapshot")
save_table(comparison,"change_reconciliation")
save_table(history,"delta_history")
save_json({"first_merge":first_merge,"replayed_merge":second_merge,"initial_version":initial_version,"current_version":current_table.version()},"merge_metrics")
import shutil
published_path=output_dir/"customer_delta"
shutil.copytree(table_path,published_path)
save_json({"delta_path":str(published_path),"registered_fabric_table":False,"primary_key":"customer_id","ordering_column":"update_sequence"},"delta_contract")
fig,ax=plt.subplots(figsize=(9,4))
comparison.change_type.value_counts().plot.bar(ax=ax)
save_figure(fig,"delta_changes")
result=finish({"initial_rows":len(initial),"current_rows":len(after_second),"updated_rows":int(comparison.change_type.eq("update").sum()),"inserted_rows":int(comparison.change_type.eq("insert").sum()),"replay_changed_rows":0},[comparison,history])
