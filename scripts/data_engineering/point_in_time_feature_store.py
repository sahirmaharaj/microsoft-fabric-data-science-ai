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
WORKFLOW = "05_point_in_time_feature_store"


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


customer_count = 80
calendar = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
transactions = pd.DataFrame({
    "customer_id": rng.integers(1, customer_count + 1, SAMPLE_SIZE * 3),
    "event_time": rng.choice(calendar.to_numpy(), SAMPLE_SIZE * 3),
    "amount": rng.gamma(2, 40, SAMPLE_SIZE * 3)
})
transactions["event_time"] = pd.to_datetime(transactions.event_time, utc=True)
labels = pd.DataFrame({
    "customer_id": np.repeat(np.arange(1, customer_count + 1), 3),
    "label_time": np.tile(pd.to_datetime(["2025-03-01", "2025-04-01", "2025-04-25"], utc=True), customer_count)
})

def build_snapshots(events, snapshot_times):
    records = []
    for snapshot in snapshot_times:
        history = events.loc[events.event_time.lt(snapshot)].copy()
        recent = history.loc[history.event_time.ge(snapshot - pd.Timedelta(days=30))]
        all_time = history.groupby("customer_id").agg(lifetime_spend=("amount", "sum"), lifetime_orders=("amount", "size"), last_event=("event_time", "max"))
        short_term = recent.groupby("customer_id").agg(spend_30d=("amount", "sum"), orders_30d=("amount", "size"))
        features = all_time.join(short_term, how="left").reset_index()
        features[["spend_30d", "orders_30d"]] = features[["spend_30d", "orders_30d"]].fillna(0)
        features["days_since_order"] = (snapshot - features.last_event).dt.total_seconds() / 86400
        features["feature_time"] = snapshot
        features["available_at"] = snapshot + pd.Timedelta(hours=2)
        records.append(features)
    return pd.concat(records, ignore_index=True)

snapshots = build_snapshots(transactions, calendar[1:])
features = pd.merge_asof(
    labels.sort_values("label_time"),
    snapshots.sort_values("available_at"),
    left_on="label_time",
    right_on="available_at",
    by="customer_id",
    direction="backward",
    allow_exact_matches=True
)


assert features.available_at.le(features.label_time).all()
assert features.last_event.lt(features.feature_time).all()
assert not features[["customer_id", "label_time"]].duplicated().any()
future_targets = []
for row in features.itertuples(index=False):
    future = transactions.loc[transactions.customer_id.eq(row.customer_id) & transactions.event_time.ge(row.label_time) & transactions.event_time.lt(row.label_time + pd.Timedelta(days=5))]
    future_targets.append(float(future.amount.sum()))
features["future_spend_5d"] = future_targets
train = features.loc[features.label_time.lt(pd.Timestamp("2025-04-25", tz="UTC"))]
test = features.loc[features.label_time.ge(pd.Timestamp("2025-04-25", tz="UTC"))]
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
columns = ["lifetime_spend", "lifetime_orders", "spend_30d", "orders_30d", "days_since_order"]
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=8, random_state=SEED, n_jobs=2)
model.fit(train[columns], train.future_spend_5d)
predictions = test[["customer_id", "label_time", "future_spend_5d"]].copy()
predictions["prediction"] = model.predict(test[columns])
lineage = pd.DataFrame({"feature": columns, "source": "transactions", "join_key": "customer_id", "temporal_key": "available_at"})
save_table(features, "training_set")
save_table(predictions, "predictions")
save_table(lineage, "feature_lineage")
save_model(model, "spend_model")
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(predictions.future_spend_5d, predictions.prediction, alpha=0.6)
ax.set(xlabel="Actual future spend", ylabel="Predicted future spend")
save_figure(fig, "predictions")
result = finish({"feature_snapshots": len(snapshots), "train_rows": len(train), "test_rows": len(test), "mae": mean_absolute_error(test.future_spend_5d, predictions.prediction)}, [predictions])
