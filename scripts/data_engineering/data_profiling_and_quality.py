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
WORKFLOW = "01_data_profiling_and_quality"


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


from scipy.stats import skew, kurtosis, entropy

def make_orders(size):
    frame = pd.DataFrame({
        "order_id": np.arange(size),
        "customer_id": rng.integers(1000, 1250, size),
        "region": rng.choice(["Cape Town", "Durban", "Johannesburg"], size),
        "amount": rng.lognormal(4.2, 0.8, size),
        "quantity": rng.integers(1, 12, size),
        "created_at": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, size), unit="D")
    })
    frame.loc[rng.choice(size, size // 20, replace=False), "amount"] = np.nan
    frame.loc[rng.choice(size, size // 25, replace=False), "region"] = None
    frame.loc[0, "amount"] = 25000.0
    return pd.concat([frame, frame.iloc[:8]], ignore_index=True)

def profile_columns(frame):
    records = []
    for column in frame:
        values = frame[column]
        counts = values.value_counts(dropna=True)
        record = {
            "column": column,
            "dtype": str(values.dtype),
            "rows": len(values),
            "missing": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()),
            "unique": int(values.nunique()),
            "unique_rate": float(values.nunique() / max(len(values), 1)),
            "entropy": float(entropy(counts.to_numpy())) if len(counts) else 0.0
        }
        if pd.api.types.is_numeric_dtype(values):
            clean = values.dropna()
            q1, q3 = clean.quantile([0.25, 0.75])
            spread = q3 - q1
            record.update({
                "minimum": float(clean.min()),
                "maximum": float(clean.max()),
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "std": float(clean.std()),
                "skew": float(skew(clean)),
                "kurtosis": float(kurtosis(clean)),
                "iqr_outlier_rate": float(((clean < q1 - 1.5 * spread) | (clean > q3 + 1.5 * spread)).mean())
            })
        records.append(record)
    return pd.DataFrame(records)



orders = make_orders(SAMPLE_SIZE)
profile = profile_columns(orders)
duplicate_rows = orders.loc[orders.duplicated("order_id", keep=False)].sort_values("order_id")
numeric = orders.select_dtypes(include="number")
correlations = numeric.corr(method="spearman")
pairs = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack().reset_index()
pairs.columns = ["feature_a", "feature_b", "spearman"]
quality_rules = pd.DataFrame([
    {"rule": "unique_order_id", "passed": not orders.order_id.duplicated().any(), "violations": int(orders.order_id.duplicated().sum())},
    {"rule": "nonnegative_amount", "passed": bool(orders.amount.dropna().ge(0).all()), "violations": int(orders.amount.lt(0).sum())},
    {"rule": "complete_amount", "passed": bool(orders.amount.notna().all()), "violations": int(orders.amount.isna().sum())}
])
regional = orders.groupby("region", dropna=False).agg(orders=("order_id", "size"), revenue=("amount", "sum"), median_basket=("amount", "median")).reset_index()
for name, table in [("column_profile", profile), ("duplicates", duplicate_rows), ("correlations", pairs), ("quality_rules", quality_rules), ("regional_summary", regional)]:
    save_table(table, name)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
profile.plot.bar(x="column", y="missing_rate", ax=axes[0], legend=False)
axes[0].set_ylabel("Missing fraction")
axes[1].hist(np.log1p(orders.amount.dropna()), bins=30)
axes[1].set_xlabel("Log amount")
save_figure(fig, "profile")
assert len(profile) == len(orders.columns)
assert len(duplicate_rows) == 16
result = finish({"rows": len(orders), "duplicate_keys": int(orders.order_id.duplicated().sum()), "passed_rules": int(quality_rules.passed.sum())}, [profile, quality_rules])
