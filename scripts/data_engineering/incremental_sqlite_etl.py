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
WORKFLOW = "03_incremental_sqlite_etl"


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


import sqlite3

connection = sqlite3.connect(working_dir / "warehouse.sqlite")
connection.executescript("""
CREATE TABLE events(event_id TEXT PRIMARY KEY, entity_id INTEGER NOT NULL, event_time TEXT NOT NULL, value REAL NOT NULL, payload_hash TEXT NOT NULL);
CREATE TABLE watermarks(pipeline TEXT PRIMARY KEY, last_event_time TEXT NOT NULL);
CREATE TABLE audit(batch_id TEXT PRIMARY KEY, input_rows INTEGER, inserted_rows INTEGER, duplicates INTEGER, mismatches INTEGER);
""")

def event_digest(row):
    data = {"entity_id": int(row.entity_id), "event_time": str(row.event_time), "value": float(row.value)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def ingest_batch(frame, batch_id):
    inserted, duplicate, mismatches = 0, 0, []
    with connection:
        for row in frame.itertuples(index=False):
            digest = event_digest(row)
            previous = connection.execute("SELECT payload_hash FROM events WHERE event_id=?", (row.event_id,)).fetchone()
            if previous:
                if previous[0] == digest:
                    duplicate += 1
                else:
                    mismatches.append({"event_id": row.event_id, "existing_hash": previous[0], "incoming_hash": digest})
                continue
            connection.execute("INSERT INTO events VALUES(?,?,?,?,?)", (row.event_id, int(row.entity_id), str(row.event_time), float(row.value), digest))
            inserted += 1
        maximum = connection.execute("SELECT MAX(event_time) FROM events").fetchone()[0]
        connection.execute("INSERT INTO watermarks VALUES(?,?) ON CONFLICT(pipeline) DO UPDATE SET last_event_time=excluded.last_event_time", ("events", maximum))
        connection.execute("INSERT INTO audit VALUES(?,?,?,?,?)", (batch_id, len(frame), inserted, duplicate, len(mismatches)))
    return pd.DataFrame(mismatches, columns=["event_id", "existing_hash", "incoming_hash"])

initial = pd.DataFrame({
    "event_id": [f"event-{i}" for i in range(SAMPLE_SIZE)],
    "entity_id": rng.integers(1, 100, SAMPLE_SIZE),
    "event_time": pd.date_range("2025-01-01", periods=SAMPLE_SIZE, freq="min").astype(str),
    "value": rng.normal(100, 15, SAMPLE_SIZE).round(3)
})


ingest_batch(initial, "initial")
replayed = initial.sample(100, random_state=SEED).copy()
late = initial.iloc[:20].copy()
late["event_id"] = [f"late-{i}" for i in range(len(late))]
conflict = initial.iloc[[0]].copy()
conflict["value"] += 99
second = pd.concat([replayed, late, conflict], ignore_index=True)
conflicts = ingest_batch(second, "replay_and_late")
audit = pd.read_sql_query("SELECT * FROM audit ORDER BY rowid", connection)
current = pd.read_sql_query("""
SELECT entity_id, event_id, event_time, value FROM (
SELECT *, ROW_NUMBER() OVER(PARTITION BY entity_id ORDER BY event_time DESC, event_id DESC) AS position
FROM events
) WHERE position=1 ORDER BY entity_id
""", connection)
counts = pd.read_sql_query("SELECT entity_id, COUNT(*) AS events, AVG(value) AS mean_value FROM events GROUP BY entity_id", connection)
watermark = pd.read_sql_query("SELECT * FROM watermarks", connection)
assert audit.inserted_rows.sum() == SAMPLE_SIZE + 20
assert audit.duplicates.sum() == 100
assert len(conflicts) == 1
assert current.entity_id.is_unique
save_table(audit, "audit")
save_table(conflicts, "conflicts")
save_table(current, "latest_state")
save_table(counts, "entity_metrics")
save_table(watermark, "watermarks")
connection.close()
__import__("shutil").copy2(working_dir / "warehouse.sqlite", output_dir / "warehouse.sqlite")
record_artifact(output_dir / "warehouse.sqlite")
fig, ax = plt.subplots(figsize=(9, 4))
counts.plot.scatter(x="events", y="mean_value", ax=ax)
save_figure(fig, "entity_distribution")
result = finish({"stored_events": int(audit.inserted_rows.sum()), "ignored_duplicates": int(audit.duplicates.sum()), "conflicts": len(conflicts)}, [audit, current])
