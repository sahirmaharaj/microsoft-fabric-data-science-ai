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
WORKFLOW = "02_schema_contracts_and_quarantine"


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


from decimal import Decimal, InvalidOperation

@dataclass(frozen=True)
class OrderContract:
    minimum_amount: float = 0.0
    maximum_amount: float = 100000.0
    currencies: tuple = ("ZAR", "USD", "EUR")
    schema_version: int = 1

contract = OrderContract()
raw = pd.DataFrame({
    "event_id": [f"evt-{i:05d}" for i in range(SAMPLE_SIZE)],
    "customer_id": rng.integers(1, 250, SAMPLE_SIZE).astype(str),
    "amount": np.round(rng.uniform(5, 5000, SAMPLE_SIZE), 2).astype(str),
    "currency": rng.choice(contract.currencies, SAMPLE_SIZE),
    "event_time": pd.date_range("2025-01-01", periods=SAMPLE_SIZE, freq="h").astype(str)
})
raw.loc[3, "amount"] = "invalid"
raw.loc[5, "amount"] = "-50"
raw.loc[7, "currency"] = "UNKNOWN"
raw.loc[9, "event_time"] = "not-a-date"
raw.loc[11, "customer_id"] = ""
raw = pd.concat([raw, raw.iloc[[20]]], ignore_index=True)

def validate_batch(frame, contract):
    required = {"event_id", "customer_id", "amount", "currency", "event_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(sorted(missing))
    output = frame.copy()
    output["amount"] = pd.to_numeric(output.amount, errors="coerce")
    output["customer_id"] = pd.to_numeric(output.customer_id, errors="coerce").astype("Int64")
    output["event_time"] = pd.to_datetime(output.event_time, errors="coerce", utc=True)
    checks = pd.DataFrame({
        "empty_event_id": output.event_id.isna() | output.event_id.astype(str).str.strip().eq(""),
        "invalid_customer": output.customer_id.isna() | output.customer_id.le(0),
        "invalid_amount": output.amount.isna() | ~np.isfinite(output.amount),
        "out_of_range_amount": output.amount.lt(contract.minimum_amount) | output.amount.gt(contract.maximum_amount),
        "invalid_currency": ~output.currency.isin(contract.currencies),
        "invalid_time": output.event_time.isna(),
        "duplicate_event_id": output.event_id.duplicated(keep="first")
    }).fillna(True)
    output["violations"] = checks.apply(lambda row: "|".join(row.index[row]), axis=1)
    output["schema_version"] = contract.schema_version
    valid = output.loc[~checks.any(axis=1)].drop(columns="violations").copy()
    quarantine = output.loc[checks.any(axis=1)].copy()
    report = checks.sum().rename("violations").rename_axis("rule").reset_index()
    return valid, quarantine, report



def canonical_fingerprint(frame):
    ordered = frame.sort_values("event_id").copy()
    for column in ordered:
        ordered[column] = ordered[column].astype(str)
    payload = ordered.to_json(orient="records")
    return hashlib.sha256(payload.encode()).hexdigest()

valid, quarantine, report = validate_batch(raw, contract)
assert len(valid) + len(quarantine) == len(raw)
assert valid.event_id.is_unique
assert valid.amount.ge(0).all()
assert len(quarantine) == 6
fingerprint = canonical_fingerprint(valid)
repeat_valid, repeat_quarantine, _ = validate_batch(raw, contract)
assert fingerprint == canonical_fingerprint(repeat_valid)
reconciled = valid.groupby("currency").agg(rows=("event_id", "size"), amount=("amount", "sum")).reset_index()
save_table(valid, "validated_orders")
save_table(quarantine, "quarantine")
save_table(report, "rule_failures")
save_table(reconciled, "reconciliation")
save_json(asdict(contract), "schema_contract")
save_json({"input_rows": len(raw), "accepted": len(valid), "rejected": len(quarantine), "fingerprint": fingerprint}, "batch_receipt")
fig, ax = plt.subplots(figsize=(9, 4))
report.plot.barh(x="rule", y="violations", ax=ax, legend=False)
save_figure(fig, "contract_failures")
result = finish({"accepted_rows": len(valid), "quarantined_rows": len(quarantine), "acceptance_rate": len(valid) / len(raw)}, [report, quarantine])
