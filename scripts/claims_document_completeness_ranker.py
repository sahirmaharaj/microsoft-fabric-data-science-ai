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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, QuantileRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import OneClassSVM
np.random.seed(16467)
base_dir = Path.cwd()
artifact_dir = base_dir / "outputs" / "1264_claims_document_completeness_ranker"
artifact_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(16467)
rows = 284
t = np.arange(rows)
calendar = pd.date_range("2026-01-01", periods=rows, freq="D")
segments = rng.choice(["enterprise", "midmarket", "startup", "public_sector"], size=rows, p=[0.28, 0.34, 0.27, 0.11])
regions = rng.choice(["cape_town", "durban", "johannesburg", "remote"], size=rows, p=[0.32, 0.18, 0.36, 0.14])
channels = rng.choice(["organic", "paid", "partner", "email", "field"], size=rows)
season = np.sin(t / 11.0) + 0.45 * np.cos(t / 29.0)
trend = t / rows
noise = rng.normal(0, 0.65, rows)
segment_effect = pd.Series(segments).map({"enterprise": 1.8, "midmarket": 0.8, "startup": -0.1, "public_sector": 0.35}).to_numpy()
region_effect = pd.Series(regions).map({"cape_town": 0.65, "durban": 0.35, "johannesburg": 0.9, "remote": 0.15}).to_numpy()
channel_effect = pd.Series(channels).map({"organic": 0.55, "paid": 0.75, "partner": 1.0, "email": 0.45, "field": 1.25}).to_numpy()
x1 = rng.normal(0, 1, rows) + 0.25 * season + 0.15 * segment_effect
x2 = rng.gamma(2.5, 1.2, rows) + 0.4 * region_effect
x3 = rng.beta(2.0, 5.0, rows) + 0.08 * channel_effect
x4 = rng.poisson(3.5 + 1.2 * np.maximum(season, -0.5), rows)
score = -1.35 + 0.95 * x1 + 0.33 * x2 + 1.45 * x3 + 0.12 * x4 + 0.55 * segment_effect + 0.3 * region_effect + 0.2 * channel_effect + 0.55 * season + noise
probability = 1 / (1 + np.exp(-score))
target = rng.binomial(1, np.clip(probability, 0.02, 0.98))
value = 1200 + 280 * x2 + 950 * target + 420 * segment_effect + 180 * season + rng.normal(0, 280, rows)
duration = np.maximum(1, rng.lognormal(2.2 + 0.15 * target + 0.04 * x2, 0.42, rows))
text_pool = np.array(["urgent model review", "fabric lakehouse refresh", "customer churn signal", "security policy exception", "forecast variance spike", "agent answer audit", "capacity pressure rising", "invoice delay risk", "route telemetry drift", "feature quality warning"])
text = [str(text_pool[(i + 1264) % len(text_pool)]) + " " + str(channels[i]) + " " + str(regions[i]) for i in range(rows)]
df = pd.DataFrame({"date": calendar, "segment": segments, "region": regions, "channel": channels, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "score": score, "probability": probability, "target": target, "value": value, "duration": duration, "text": text})
df["workflow"] = "Claims Document Completeness Ranker"
df["family"] = "funeral service operations analytics"
df["motif"] = "documents"
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["lag_value"] = df["value"].shift(1).bfill()
df["rolling_value"] = df["value"].rolling(14, min_periods=1).mean()
df["change_signal"] = (df["value"] - df["rolling_value"]) / df["rolling_value"].std()
df["risk_label"] = np.where((df["probability"] > np.quantile(df["probability"], 0.72)) | (df["change_signal"] > 1.0), 1, 0)
df.to_csv(artifact_dir / "synthetic_input.csv", index=False)
numeric_cols = ["x1", "x2", "x3", "x4", "lag_value", "rolling_value", "duration"]
categorical_cols = ["segment", "region", "channel"]
preprocess = ColumnTransformer([("num", StandardScaler(), numeric_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)])
model = Pipeline([("prep", preprocess), ("clf", GradientBoostingClassifier(random_state=16467))])
train_df, test_df = train_test_split(df, test_size=0.28, random_state=16467, stratify=df["risk_label"])
model.fit(train_df[numeric_cols + categorical_cols], train_df["risk_label"])
pred = model.predict(test_df[numeric_cols + categorical_cols])
proba = model.predict_proba(test_df[numeric_cols + categorical_cols])[:, 1]
metrics = {"accuracy": float(accuracy_score(test_df["risk_label"], pred)), "roc_auc": float(roc_auc_score(test_df["risk_label"], proba)), "average_precision": float(average_precision_score(test_df["risk_label"], proba)), "brier": float(brier_score_loss(test_df["risk_label"], proba))}
regressor = Pipeline([("prep", preprocess), ("reg", RandomForestRegressor(n_estimators=80, min_samples_leaf=4, random_state=16467, n_jobs=-1))])
regressor.fit(train_df[numeric_cols + categorical_cols], train_df["value"])
value_pred = regressor.predict(test_df[numeric_cols + categorical_cols])
metrics["value_mae"] = float(mean_absolute_error(test_df["value"], value_pred))
metrics["value_rmse"] = float(np.sqrt(mean_squared_error(test_df["value"], value_pred)))
encoded = preprocess.fit_transform(df[numeric_cols + categorical_cols])
if hasattr(encoded, "toarray"):
    encoded_dense = encoded.toarray()
else:
    encoded_dense = np.asarray(encoded)
pca = PCA(n_components=2, random_state=16467)
embedding = pca.fit_transform(encoded_dense)
kmeans = KMeans(n_clusters=4, random_state=16467, n_init=10)
clusters = kmeans.fit_predict(encoded_dense)
df["cluster"] = clusters
metrics["silhouette"] = float(silhouette_score(encoded_dense, clusters))
outlier = IsolationForest(contamination=0.08, random_state=16467)
df["anomaly_score"] = -outlier.fit(encoded_dense).score_samples(encoded_dense)
precision, recall, thresholds = precision_recall_curve(test_df["risk_label"], proba)
gain = []
for cutoff in np.linspace(0.1, 0.9, 17):
    chosen = proba >= cutoff
    benefit = np.sum(chosen * test_df["risk_label"].to_numpy() * 500) - np.sum(chosen * 65)
    gain.append((float(cutoff), float(benefit), int(chosen.sum())))
gain_df = pd.DataFrame(gain, columns=["threshold", "profit", "selected"])
best_threshold = gain_df.sort_values("profit", ascending=False).iloc[0].to_dict()
metrics["best_threshold"] = float(best_threshold["threshold"])
metrics["best_profit"] = float(best_threshold["profit"])
group_table = df.groupby(["segment", "region"], observed=True).agg(records=("target", "size"), risk_rate=("risk_label", "mean"), value_mean=("value", "mean"), anomaly_mean=("anomaly_score", "mean")).reset_index()
text_model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)), ("svd", TruncatedSVD(n_components=5, random_state=16467))])
text_vectors = text_model.fit_transform(df["text"])
df["text_component_1"] = text_vectors[:, 0]
df["text_component_2"] = text_vectors[:, 1]
quantile_low = QuantileRegressor(quantile=0.1, alpha=0.02, solver="highs")
quantile_high = QuantileRegressor(quantile=0.9, alpha=0.02, solver="highs")
scaled_numeric = StandardScaler().fit_transform(df[numeric_cols])
quantile_low.fit(scaled_numeric, df["value"])
quantile_high.fit(scaled_numeric, df["value"])
df["value_p10"] = quantile_low.predict(scaled_numeric)
df["value_p90"] = quantile_high.predict(scaled_numeric)
coverage = np.mean((df["value"] >= df["value_p10"]) & (df["value"] <= df["value_p90"]))
metrics["interval_coverage"] = float(coverage)
drift_ref = df.iloc[: rows // 2][numeric_cols].mean()
drift_cur = df.iloc[rows // 2:][numeric_cols].mean()
drift = ((drift_cur - drift_ref).abs() / (df[numeric_cols].std() + 1e-9)).sort_values(ascending=False)
metrics["largest_drift_feature"] = str(drift.index[0])
metrics["largest_drift_score"] = float(drift.iloc[0])
corr = df[numeric_cols + ["value", "probability", "anomaly_score"]].corr()
distance = cdist(group_table[["risk_rate", "value_mean", "anomaly_mean"]], group_table[["risk_rate", "value_mean", "anomaly_mean"]])
assignment_rows, assignment_cols = linear_sum_assignment(distance + np.eye(distance.shape[0]) * 1e6)
pair_df = pd.DataFrame({"source": assignment_rows, "matched": assignment_cols, "distance": distance[assignment_rows, assignment_cols]}).sort_values("distance").head(10)
timeline = df.groupby("date").agg(value=("value", "mean"), risk=("risk_label", "mean"), anomaly=("anomaly_score", "mean")).reset_index()
timeline["ewma_value"] = timeline["value"].ewm(span=18).mean()
timeline["control_high"] = timeline["ewma_value"] + 2.4 * timeline["value"].rolling(21, min_periods=5).std().bfill()
timeline["control_low"] = timeline["ewma_value"] - 2.4 * timeline["value"].rolling(21, min_periods=5).std().bfill()
scenario = []
for shift in np.linspace(-0.35, 0.35, 15):
    adjusted = np.clip(proba + shift, 0.001, 0.999)
    scenario.append({"shift": float(shift), "positive_rate": float(np.mean(adjusted >= metrics["best_threshold"])), "expected_value": float(np.mean(adjusted * test_df["value"].to_numpy()))})
scenario_df = pd.DataFrame(scenario)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], c=df["cluster"], cmap="viridis", s=18, alpha=0.78)
axes[0, 0].set_title("Claims Document Completeness Ranker")
axes[0, 1].plot(timeline["date"], timeline["value"], color="#2563eb", linewidth=1.3)
axes[0, 1].plot(timeline["date"], timeline["ewma_value"], color="#111827", linewidth=2)
axes[0, 1].fill_between(timeline["date"], timeline["control_low"], timeline["control_high"], color="#93c5fd", alpha=0.3)
axes[1, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(corr.columns)))
axes[1, 0].set_xticklabels(corr.columns, rotation=90, fontsize=7)
axes[1, 0].set_yticks(range(len(corr.index)))
axes[1, 0].set_yticklabels(corr.index, fontsize=7)
axes[1, 1].plot(gain_df["threshold"], gain_df["profit"], marker="o", color="#16a34a")
axes[1, 1].axvline(metrics["best_threshold"], color="#dc2626", linewidth=1.5)
fig.tight_layout()
fig.savefig(artifact_dir / "workflow_diagnostics.png", dpi=150)
plt.close(fig)
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
axes2[0].bar(drift.index, drift.values, color="#7c3aed")
axes2[0].tick_params(axis="x", rotation=45)
axes2[1].plot(scenario_df["shift"], scenario_df["positive_rate"], color="#0891b2", marker="s")
axes2[1].set_ylim(0, 1)
axes2[2].scatter(df["value_p10"], df["value_p90"], c=df["risk_label"], cmap="plasma", s=20, alpha=0.72)
axes2[2].plot([df["value_p10"].min(), df["value_p90"].max()], [df["value_p10"].min(), df["value_p90"].max()], color="#111827", linewidth=1)
fig2.tight_layout()
fig2.savefig(artifact_dir / "scenario_and_interval_analysis.png", dpi=150)
plt.close(fig2)
df.sort_values("anomaly_score", ascending=False).head(35).to_csv(artifact_dir / "top_anomalies.csv", index=False)
group_table.to_csv(artifact_dir / "segment_region_scorecard.csv", index=False)
gain_df.to_csv(artifact_dir / "threshold_profit_curve.csv", index=False)
scenario_df.to_csv(artifact_dir / "scenario_sensitivity.csv", index=False)
pair_df.to_csv(artifact_dir / "nearest_operational_matches.csv", index=False)
timeline.to_csv(artifact_dir / "monitoring_timeline.csv", index=False)
with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps({"workflow": "1264_claims_document_completeness_ranker", "metrics": metrics, "artifacts": len(list(artifact_dir.iterdir()))}, indent=2))
