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
WORKFLOW = "23_rolling_origin_forecast_backtesting"


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


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

periods = 560
dates = pd.date_range("2024-01-01", periods=periods, freq="D")
t = np.arange(periods)
y = 100 + 0.08 * t + 12 * np.sin(2 * np.pi * t / 7) + 5 * np.cos(2 * np.pi * t / 30) + rng.normal(0, 4, periods)
series = pd.DataFrame({"date": dates, "demand": y})

def feature_frame(frame):
    output = frame.copy()
    for lag in [1, 2, 7, 14, 28]:
        output[f"lag_{lag}"] = output.demand.shift(lag)
    for window in [7, 14, 28]:
        output[f"rolling_mean_{window}"] = output.demand.shift(1).rolling(window).mean()
        output[f"rolling_std_{window}"] = output.demand.shift(1).rolling(window).std()
    output["weekday_sin"] = np.sin(2 * np.pi * output.date.dt.dayofweek / 7)
    output["weekday_cos"] = np.cos(2 * np.pi * output.date.dt.dayofweek / 7)
    output["trend"] = (output.date - dates[0]).dt.days
    return output

features = feature_frame(series).dropna().reset_index(drop=True)
columns = [column for column in features if column not in ["date", "demand"]]

backtest_rows = []
for cutoff in [350, 390, 430, 470, 510]:
    train = features.loc[features.date.lt(dates[cutoff])]
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=12, l2_regularization=3, random_state=SEED)
    model.fit(train[columns], train.demand)
    history = series.iloc[:cutoff].copy()
    recursive_predictions = []
    for horizon in range(1, 15):
        future_date = dates[cutoff + horizon - 1]
        temporary = pd.concat([history, pd.DataFrame({"date": [future_date], "demand": [np.nan]})], ignore_index=True)
        row = feature_frame(temporary).iloc[[-1]]
        predicted = float(model.predict(row[columns])[0])
        seasonal = float(series.iloc[cutoff - 7 + (horizon - 1) % 7].demand)
        backtest_rows.append({"cutoff": dates[cutoff], "date": future_date, "horizon": horizon, "actual": float(series.iloc[cutoff+horizon-1].demand), "prediction": predicted, "seasonal_naive": seasonal})
        history = pd.concat([history, pd.DataFrame({"date": [future_date], "demand": [predicted]})], ignore_index=True)


backtest = pd.DataFrame(backtest_rows)
backtest["absolute_error"] = abs(backtest.actual-backtest.prediction)
backtest["naive_error"] = abs(backtest.actual-backtest.seasonal_naive)
by_horizon = backtest.groupby("horizon").agg(mae=("absolute_error", "mean"), naive_mae=("naive_error", "mean"), origins=("cutoff", "nunique")).reset_index()
by_origin = backtest.groupby("cutoff").agg(mae=("absolute_error", "mean"), naive_mae=("naive_error", "mean")).reset_index()
final_model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=12, random_state=SEED).fit(features[columns], features.demand)
future_rows = []
history = series.copy()
for horizon in range(1, 15):
    future_date = dates[-1] + pd.Timedelta(days=horizon)
    temporary = pd.concat([history, pd.DataFrame({"date": [future_date], "demand": [np.nan]})], ignore_index=True)
    prediction = float(final_model.predict(feature_frame(temporary).iloc[[-1]][columns])[0])
    future_rows.append({"date": future_date, "horizon": horizon, "prediction": prediction})
    history = pd.concat([history, pd.DataFrame({"date": [future_date], "demand": [prediction]})], ignore_index=True)
forecast = pd.DataFrame(future_rows)
save_table(backtest, "backtest_predictions")
save_table(by_horizon, "horizon_metrics")
save_table(by_origin, "origin_metrics")
save_table(forecast, "future_forecast")
save_model({"model": final_model, "features": columns, "history": series.tail(28)}, "forecast_bundle")
assert backtest.groupby("cutoff").horizon.max().eq(14).all()
assert forecast.date.gt(series.date.max()).all()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
by_horizon.plot(x="horizon", y=["mae", "naive_mae"], ax=axes[0])
axes[1].plot(series.date.tail(60), series.demand.tail(60))
axes[1].plot(forecast.date, forecast.prediction)
save_figure(fig, "forecast_backtest")
result = finish({"backtest_mae": float(backtest.absolute_error.mean()), "naive_mae": float(backtest.naive_error.mean()), "forecast_steps": len(forecast)}, [by_horizon, forecast])
