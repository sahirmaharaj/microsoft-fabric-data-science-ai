import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4', 'pyarrow': 'pyarrow>=14,<30', 'duckdb': 'duckdb>=1.1,<2'}
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
WORKFLOW = "53_lakehouse_partitioned_parquet_and_arrow"


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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import duckdb

schema=pa.schema([("event_id",pa.int64()),("event_date",pa.string()),("region",pa.string()),("customer_id",pa.int64()),("revenue",pa.float64()),("units",pa.int64())])
frame=pd.DataFrame({
    "event_id":np.arange(SAMPLE_SIZE*3,dtype=np.int64),
    "event_date":rng.choice(pd.date_range("2025-01-01",periods=12).strftime("%Y-%m-%d"),SAMPLE_SIZE*3),
    "region":rng.choice(["north","south","west"],SAMPLE_SIZE*3),
    "customer_id":rng.integers(1,500,SAMPLE_SIZE*3,dtype=np.int64),
    "revenue":rng.gamma(3,70,SAMPLE_SIZE*3),
    "units":rng.integers(1,10,SAMPLE_SIZE*3,dtype=np.int64)
})
table=pa.Table.from_pandas(frame,schema=schema,preserve_index=False)
dataset_path=output_dir/"partitioned_events"
partition_schema=pa.schema([("event_date",pa.string()),("region",pa.string())])
partitioning=ds.partitioning(partition_schema,flavor="hive")
write_options=ds.ParquetFileFormat().make_write_options(compression="zstd")
ds.write_dataset(table,dataset_path,format="parquet",partitioning=partitioning,file_options=write_options,max_rows_per_file=500,max_rows_per_group=250,existing_data_behavior="error")
dataset=ds.dataset(dataset_path,format="parquet",partitioning="hive")
filter_expression=(ds.field("region")=="north")&(ds.field("event_date")>="2025-01-07")
selected=dataset.to_table(columns=["event_id","revenue","units","region","event_date"],filter=filter_expression).to_pandas()
expected=frame.loc[frame.region.eq("north")&frame.event_date.ge("2025-01-07")]
assert len(selected)==len(expected)
assert np.isclose(selected.revenue.sum(),expected.revenue.sum())
scanner=dataset.scanner(columns=["revenue","units"],batch_size=256)
scanned_rows=0
revenue_total=0.0
unit_total=0
for batch in scanner.to_batches():
    scanned_rows+=batch.num_rows
    revenue_total+=float(pa.compute.sum(batch.column("revenue")).as_py())
    unit_total+=int(pa.compute.sum(batch.column("units")).as_py())
connection=duckdb.connect()


connection.register("events_arrow",dataset)
summary=connection.execute("SELECT region,event_date,COUNT(*) AS events,SUM(revenue) AS revenue,SUM(units) AS units FROM events_arrow GROUP BY 1,2 ORDER BY 1,2").df()
connection.close()
file_rows=[]
for path in sorted(dataset_path.rglob("*.parquet")):
    metadata=pq.read_metadata(path)
    file_rows.append({"path":str(path.relative_to(output_dir)),"rows":metadata.num_rows,"row_groups":metadata.num_row_groups,"bytes":path.stat().st_size})
file_inventory=pd.DataFrame(file_rows)
save_table(summary,"partition_summary")
save_table(selected,"filtered_events")
save_table(file_inventory,"parquet_inventory")
save_json({"schema":str(schema),"partition_columns":["event_date","region"],"compression":"zstd","dataset_path":str(dataset_path),"lakehouse_attached":lakehouse_files.is_dir()},"dataset_contract")
assert scanned_rows==len(frame)
assert np.isclose(revenue_total,frame.revenue.sum())
assert unit_total==int(frame.units.sum())
fig,axes=plt.subplots(1,2,figsize=(11,4))
summary.groupby("region").revenue.sum().plot.bar(ax=axes[0])
axes[1].hist(file_inventory["bytes"],bins=15)
save_figure(fig,"parquet_analytics")
result=finish({"rows":scanned_rows,"parquet_files":len(file_inventory),"filtered_rows":len(selected),"revenue":revenue_total,"parquet_bytes":int(file_inventory["bytes"].sum())},[summary,file_inventory])
