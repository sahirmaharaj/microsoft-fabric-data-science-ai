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
WORKFLOW = "08_mixed_type_feature_pipeline"


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


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

frame = pd.DataFrame({
    "age": rng.integers(18, 75, SAMPLE_SIZE).astype(float),
    "tenure": rng.uniform(0, 10, SAMPLE_SIZE),
    "channel": rng.choice(["web", "store", "partner"], SAMPLE_SIZE),
    "region": rng.choice(["WC", "GP", "KZN"], SAMPLE_SIZE),
    "signup": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1800, SAMPLE_SIZE), unit="D")
})
frame["spend"] = 80 + 0.03 * (frame.age - 40) ** 2 + 30 * np.sqrt(frame.tenure + 1) + 20 * frame.channel.eq("web") + rng.normal(0, 8, SAMPLE_SIZE)
frame.loc[rng.choice(len(frame), 60, replace=False), "age"] = np.nan
frame["signup_month_sin"] = np.sin(2 * np.pi * frame.signup.dt.month / 12)
frame["signup_month_cos"] = np.cos(2 * np.pi * frame.signup.dt.month / 12)
X = frame.drop(columns=["spend", "signup"])
y = frame.spend
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
numeric_columns = ["age", "tenure"]
cyclic_columns = ["signup_month_sin", "signup_month_cos"]
categorical_columns = ["channel", "region"]
numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("spline", SplineTransformer(n_knots=5, degree=3)), ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns), ("cyclic", "passthrough", cyclic_columns)], remainder="drop")
pipeline = Pipeline([("features", preprocessor), ("model", Ridge())])
search = GridSearchCV(pipeline, {"model__alpha": [0.1, 1.0, 10.0], "features__numeric__spline__n_knots": [4, 6]}, cv=4, scoring="neg_mean_absolute_error", n_jobs=2)
search.fit(X_train, y_train)
best = search.best_estimator_
prediction = best.predict(X_test)
feature_names = best.named_steps["features"].get_feature_names_out()
coefficients = pd.DataFrame({"feature": feature_names, "coefficient": best.named_steps["model"].coef_}).sort_values("coefficient", key=abs, ascending=False)


unseen = X_test.head(5).copy()
unseen["region"] = "NEW_REGION"
unseen_prediction = best.predict(unseen)
assert np.isfinite(unseen_prediction).all()
model_path = save_model(best, "feature_pipeline")
restored = joblib.load(model_path)
assert np.allclose(restored.predict(X_test), prediction)
predictions = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": prediction})
predictions["residual"] = predictions.actual - predictions.predicted
search_results = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
save_table(coefficients, "coefficients")
save_table(search_results, "search")
save_table(predictions, "predictions")
save_json({"raw_columns": list(X.columns), "numeric": numeric_columns, "categorical": categorical_columns, "engineered_features": list(feature_names), "best_parameters": search.best_params_}, "feature_contract")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(predictions.actual, predictions.predicted, alpha=0.5)
axes[0].set(xlabel="Actual", ylabel="Prediction")
axes[1].hist(predictions.residual, bins=25)
save_figure(fig, "regression_diagnostics")
result = finish({"mae": mean_absolute_error(y_test, prediction), "r2": r2_score(y_test, prediction), "engineered_features": len(feature_names)}, [predictions, coefficients])
