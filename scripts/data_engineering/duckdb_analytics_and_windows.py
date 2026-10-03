import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4', 'duckdb': 'duckdb>=1.1,<2'}
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
WORKFLOW = "04_duckdb_analytics_and_windows"


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


import duckdb

orders = pd.DataFrame({
    "order_id": np.arange(SAMPLE_SIZE),
    "customer_id": rng.integers(1, 180, SAMPLE_SIZE),
    "region": rng.choice(["west", "east", "north"], SAMPLE_SIZE),
    "order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 180, SAMPLE_SIZE), unit="D"),
    "revenue": rng.gamma(3, 90, SAMPLE_SIZE).round(2)
})
orders["cost"] = (orders.revenue * rng.uniform(0.3, 0.8, len(orders))).round(2)
connection = duckdb.connect(str(working_dir / "analytics.duckdb"))
connection.register("incoming", orders)
connection.execute("CREATE TABLE orders AS SELECT * FROM incoming")
connection.execute("CREATE INDEX customer_lookup ON orders(customer_id)")
monthly = connection.execute("""
WITH monthly AS (
 SELECT date_trunc('month', order_date) AS month, region,
 SUM(revenue) AS revenue, SUM(revenue-cost) AS profit, COUNT(*) AS orders
 FROM orders GROUP BY 1, 2
), changes AS (
 SELECT *, LAG(revenue) OVER(PARTITION BY region ORDER BY month) AS previous_revenue,
 SUM(revenue) OVER(PARTITION BY region ORDER BY month ROWS UNBOUNDED PRECEDING) AS cumulative_revenue
 FROM monthly
)
SELECT *, (revenue-previous_revenue)/NULLIF(previous_revenue,0) AS growth FROM changes ORDER BY region, month
""").df()
rfm = connection.execute("""
WITH reference AS (SELECT MAX(order_date)+INTERVAL 1 DAY AS cutoff FROM orders),
customer AS (
 SELECT customer_id, date_diff('day', MAX(order_date), (SELECT cutoff FROM reference)) AS recency,
 COUNT(*) AS frequency, SUM(revenue) AS monetary FROM orders GROUP BY 1
)
SELECT *, NTILE(5) OVER(ORDER BY recency DESC) AS recency_score,
NTILE(5) OVER(ORDER BY frequency) AS frequency_score,
NTILE(5) OVER(ORDER BY monetary) AS monetary_score FROM customer
""").df()


rfm["rfm_score"] = rfm[["recency_score", "frequency_score", "monetary_score"]].sum(axis=1)
rfm["segment"] = pd.cut(rfm.rfm_score, [0, 6, 10, 15], labels=["develop", "retain", "champion"])
basket = connection.execute("""
SELECT region, COUNT(*) AS orders, AVG(revenue) AS average_revenue,
quantile_cont(revenue,0.5) AS median_revenue,
quantile_cont(revenue,0.9) AS p90_revenue,
SUM(revenue-cost)/NULLIF(SUM(revenue),0) AS margin
FROM orders GROUP BY region ORDER BY margin DESC
""").df()
cohorts = connection.execute("""
WITH customer_cohort AS (
 SELECT customer_id, date_trunc('month', MIN(order_date)) AS cohort FROM orders GROUP BY 1
), active AS (
 SELECT DISTINCT o.customer_id, c.cohort,
 date_diff('month', c.cohort, date_trunc('month', o.order_date)) AS age
 FROM orders o JOIN customer_cohort c USING(customer_id)
), sizes AS (SELECT cohort, COUNT(*) AS cohort_size FROM customer_cohort GROUP BY 1)
SELECT a.cohort, age, COUNT(*) AS active_customers, MAX(cohort_size) AS cohort_size,
COUNT(*)::DOUBLE/MAX(cohort_size) AS retention
FROM active a JOIN sizes s USING(cohort) GROUP BY 1,2 ORDER BY 1,2
""").df()
assert abs(monthly.revenue.sum() - orders.revenue.sum()) < 1e-6
assert rfm.customer_id.is_unique
assert cohorts.retention.between(0, 1).all()
for name, frame in [("monthly", monthly), ("rfm", rfm), ("basket", basket), ("cohorts", cohorts)]:
    save_table(frame, name)
connection.close()
__import__("shutil").copy2(working_dir / "analytics.duckdb", output_dir / "analytics.duckdb")
record_artifact(output_dir / "analytics.duckdb")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
monthly.pivot(index="month", columns="region", values="revenue").plot(ax=axes[0])
rfm.groupby("segment", observed=True).size().plot.bar(ax=axes[1])
save_figure(fig, "analytics")
result = finish({"revenue": float(orders.revenue.sum()), "customers": len(rfm), "cohort_cells": len(cohorts)}, [monthly, rfm])
