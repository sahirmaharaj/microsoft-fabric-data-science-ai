import sys
import subprocess
import importlib.util
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
packages = ["numpy", "pandas", "matplotlib", "sklearn", "scipy"]
missing = [p for p in packages if importlib.util.find_spec("sklearn" if p == "sklearn" else p) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
np.random.seed(26176)
rng = np.random.default_rng(26176)
base_dir = Path.cwd()
artifact_dir = base_dir / "outputs" / "1538_legal_clinic_backlog_scorecard"
artifact_dir.mkdir(parents=True, exist_ok=True)
rows = 268
t = np.arange(rows)
dates = pd.date_range("2026-01-01", periods=rows, freq="D")
segments = rng.choice(["alpha", "beta", "gamma", "delta"], rows, p=[0.31, 0.27, 0.24, 0.18])
regions = rng.choice(["north", "south", "east", "west", "central"], rows)
channels = rng.choice(["digital", "field", "partner", "direct", "automated"], rows)
season = np.sin(t / 9.0) + 0.6 * np.cos(t / 23.0)
segment_effect = pd.Series(segments).map({"alpha": 1.15, "beta": 0.55, "gamma": -0.1, "delta": 0.25}).to_numpy()
region_effect = pd.Series(regions).map({"north": 0.2, "south": 0.55, "east": 0.9, "west": 0.35, "central": 0.7}).to_numpy()
channel_effect = pd.Series(channels).map({"digital": 0.75, "field": 1.0, "partner": 0.45, "direct": 0.6, "automated": 0.25}).to_numpy()
x1 = rng.normal(0, 1, rows) + 0.3 * season + 0.2 * segment_effect
x2 = rng.gamma(2.2, 1.15, rows) + 0.35 * region_effect
x3 = rng.beta(2.5, 4.5, rows) + 0.06 * channel_effect
x4 = rng.poisson(3.2 + np.maximum(season, -0.4), rows)
raw = -1.1 + 0.8 * x1 + 0.24 * x2 + 1.2 * x3 + 0.16 * x4 + 0.45 * segment_effect + 0.28 * region_effect + 0.18 * channel_effect + 0.4 * season + rng.normal(0, 0.7, rows)
probability = 1 / (1 + np.exp(-raw))
target = rng.binomial(1, np.clip(probability, 0.03, 0.97))
value = 900 + 240 * x2 + 700 * target + 320 * segment_effect + 160 * season + rng.normal(0, 230, rows)
duration = np.maximum(1, rng.lognormal(2.0 + 0.1 * target + 0.03 * x2, 0.38, rows))
phrases = np.array(["priority signal", "operational variance", "quality exception", "capacity pressure", "forecast shift", "policy review", "demand surge", "risk event"])
text = [str(phrases[(i + 1538) % len(phrases)]) + " backlog " + str(segments[i]) + " " + str(regions[i]) for i in range(rows)]
df = pd.DataFrame({"date": dates, "segment": segments, "region": regions, "channel": channels, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "probability": probability, "target": target, "value": value, "duration": duration, "text": text})
df["workflow"] = "Legal Clinic Backlog Scorecard"
df["family"] = "legal aid analytics"
df["motif"] = "backlog"
df["rolling_value"] = df["value"].rolling(14, min_periods=1).mean()
df["lag_value"] = df["value"].shift(1).bfill()
df["risk_label"] = ((df["probability"] > df["probability"].quantile(0.7)) | (df["value"] > df["rolling_value"] + df["value"].std() * 0.35)).astype(int)
df.to_csv(artifact_dir / "synthetic_input.csv", index=False)
numeric_cols = ["x1", "x2", "x3", "x4", "lag_value", "rolling_value", "duration"]
categorical_cols = ["segment", "region", "channel"]
features = numeric_cols + categorical_cols
preprocess = ColumnTransformer([("num", StandardScaler(), numeric_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)])
train_df, test_df = train_test_split(df, test_size=0.28, random_state=26176, stratify=df["risk_label"])
classifier = Pipeline([("prep", preprocess), ("clf", GradientBoostingClassifier(random_state=26176))])
classifier.fit(train_df[features], train_df["risk_label"])
proba = classifier.predict_proba(test_df[features])[:, 1]
pred = (proba >= 0.5).astype(int)
regressor = Pipeline([("prep", preprocess), ("reg", RandomForestRegressor(n_estimators=36, min_samples_leaf=4, random_state=26176, n_jobs=-1))])
regressor.fit(train_df[features], train_df["value"])
value_pred = regressor.predict(test_df[features])
encoded = preprocess.fit_transform(df[features])
encoded = encoded.toarray() if hasattr(encoded, "toarray") else np.asarray(encoded)
embedding = PCA(n_components=2, random_state=26176).fit_transform(encoded)
df["cluster"] = KMeans(n_clusters=4, random_state=26176, n_init=10).fit_predict(encoded)
df["anomaly_score"] = -IsolationForest(contamination=0.08, random_state=26176).fit(encoded).score_samples(encoded)
metrics = {"workflow_number": 1538, "workflow_name": "legal_clinic_backlog_scorecard", "domain": "legal aid analytics", "accuracy": float(accuracy_score(test_df["risk_label"], pred)), "roc_auc": float(roc_auc_score(test_df["risk_label"], proba)), "average_precision": float(average_precision_score(test_df["risk_label"], proba)), "value_mae": float(mean_absolute_error(test_df["value"], value_pred)), "value_rmse": float(np.sqrt(mean_squared_error(test_df["value"], value_pred))), "anomaly_p95": float(df["anomaly_score"].quantile(0.95))}
(artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
df.groupby(["segment", "region"], observed=True).agg(records=("target", "size"), risk_rate=("risk_label", "mean"), value_mean=("value", "mean"), duration_mean=("duration", "mean"), anomaly_mean=("anomaly_score", "mean")).reset_index().to_csv(artifact_dir / "segment_region_scorecard.csv", index=False)
threshold_rows = []
for cutoff in np.linspace(0.1, 0.9, 17):
    selected = proba >= cutoff
    profit = np.sum(selected * test_df["risk_label"].to_numpy() * 450) - np.sum(selected * 55)
    threshold_rows.append({"threshold": float(cutoff), "selected": int(selected.sum()), "profit": float(profit), "selection_rate": float(selected.mean())})
pd.DataFrame(threshold_rows).to_csv(artifact_dir / "threshold_profit_curve.csv", index=False)
df.sort_values("anomaly_score", ascending=False).head(25).to_csv(artifact_dir / "top_anomalies.csv", index=False)
timeline = df.groupby("date").agg(value=("value", "mean"), risk=("risk_label", "mean"), anomaly=("anomaly_score", "mean")).reset_index()
timeline["ewma_value"] = timeline["value"].ewm(span=18).mean()
timeline.to_csv(artifact_dir / "monitoring_timeline.csv", index=False)
corr = df[numeric_cols + ["value", "probability", "anomaly_score"]].corr()
corr.to_csv(artifact_dir / "correlation_matrix.csv")
df.groupby("cluster").agg(records=("target", "size"), value_mean=("value", "mean"), risk_rate=("risk_label", "mean"), anomaly_mean=("anomaly_score", "mean")).reset_index().to_csv(artifact_dir / "cluster_profile.csv", index=False)
text_vector = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(df["text"])
text_strength = np.asarray(text_vector.mean(axis=1)).ravel()
pd.DataFrame({"date": df["date"], "text": df["text"], "text_strength": text_strength, "risk_label": df["risk_label"]}).sort_values("text_strength", ascending=False).head(40).to_csv(artifact_dir / "text_signal_rankings.csv", index=False)
scenario = []
for shift in np.linspace(-0.3, 0.3, 13):
    adjusted = np.clip(proba + shift, 0.001, 0.999)
    scenario.append({"shift": float(shift), "positive_rate": float((adjusted >= 0.5).mean()), "expected_value": float(np.mean(adjusted * test_df["value"].to_numpy()))})
pd.DataFrame(scenario).to_csv(artifact_dir / "scenario_sensitivity.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], c=df["cluster"], cmap="viridis", s=16, alpha=0.75)
axes[0, 0].set_title("Legal Clinic Backlog Scorecard")
axes[0, 1].plot(timeline["date"], timeline["value"], linewidth=1.1)
axes[0, 1].plot(timeline["date"], timeline["ewma_value"], linewidth=2.0)
axes[1, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(corr.columns)))
axes[1, 0].set_xticklabels(corr.columns, rotation=90, fontsize=7)
axes[1, 0].set_yticks(range(len(corr.index)))
axes[1, 0].set_yticklabels(corr.index, fontsize=7)
curve = pd.DataFrame(threshold_rows)
axes[1, 1].plot(curve["threshold"], curve["profit"], marker="o")
axes[1, 1].set_title("Threshold Profit Curve")
fig.tight_layout()
fig.savefig(artifact_dir / "workflow_diagnostics.png", dpi=150)
plt.close(fig)
print(json.dumps(metrics))
