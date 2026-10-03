import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4', 'networkx': 'networkx>=3.2,<4'}
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
WORKFLOW = "51_transaction_network_and_community_risk"


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


import networkx as nx

nodes=110
transactions=[]
for index in range(SAMPLE_SIZE):
    sender=int(rng.integers(0,nodes))
    receiver=int(rng.integers(0,nodes-1))
    receiver=receiver+1 if receiver>=sender else receiver
    transactions.append({"transaction_id":index,"sender":sender,"receiver":receiver,"amount":float(rng.gamma(2,50)),"day":int(rng.integers(1,31))})
for repeat in range(10):
    for sender,receiver in [(0,1),(1,2),(2,3),(3,0)]:
        transactions.append({"transaction_id":len(transactions),"sender":sender,"receiver":receiver,"amount":2000.0,"day":repeat+1})
frame=pd.DataFrame(transactions)
aggregated=frame.groupby(["sender","receiver"]).agg(amount=("amount","sum"),transfers=("transaction_id","size")).reset_index()
graph=nx.DiGraph()
for row in aggregated.itertuples():
    graph.add_edge(row.sender,row.receiver,weight=row.amount,transfers=row.transfers,distance=1/max(row.amount,1))
pagerank=nx.pagerank(graph,weight="weight")
betweenness=nx.betweenness_centrality(graph,k=40,weight="distance",seed=SEED)
undirected=graph.to_undirected()
communities=list(nx.community.greedy_modularity_communities(undirected,weight="weight"))
community_map={node:index for index,community in enumerate(communities) for node in community}
records=[]
for node in graph.nodes:
    incoming=sum(data["weight"] for _,_,data in graph.in_edges(node,data=True))
    outgoing=sum(data["weight"] for _,_,data in graph.out_edges(node,data=True))
    reciprocal=sum(graph.has_edge(neighbor,node) for neighbor in graph.successors(node))
    records.append({"account":node,"community":community_map[node],"in_degree":graph.in_degree(node),"out_degree":graph.out_degree(node),"incoming":incoming,"outgoing":outgoing,"flow_balance":abs(incoming-outgoing)/max(incoming+outgoing,1),"reciprocal_neighbors":reciprocal,"pagerank":pagerank[node],"betweenness":betweenness[node]})
accounts=pd.DataFrame(records)
accounts["volume_percentile"]=(accounts.incoming+accounts.outgoing).rank(pct=True)
accounts["centrality_percentile"]=accounts.betweenness.rank(pct=True)
accounts["review_score"]=0.5*accounts.volume_percentile+0.3*accounts.centrality_percentile+0.2*(1-accounts.flow_balance)
accounts=accounts.sort_values("review_score",ascending=False)
community_summary=accounts.groupby("community").agg(accounts=("account","size"),incoming=("incoming","sum"),outgoing=("outgoing","sum"),mean_review_score=("review_score","mean")).reset_index()
heavy=nx.DiGraph((a,b,data) for a,b,data in graph.edges(data=True) if data["weight"]>5000)


cycles=list(nx.simple_cycles(heavy))
cycle_rows=[{"cycle_id":index,"length":len(cycle),"accounts":"|".join(map(str,cycle))} for index,cycle in enumerate(cycles)]
save_table(frame,"transactions")
save_table(accounts,"account_graph_features")
save_table(community_summary,"community_summary")
save_table(pd.DataFrame(cycle_rows,columns=["cycle_id","length","accounts"]),"high_value_cycles")
graph_path=output_dir/"transaction_graph.graphml"
nx.write_graphml(graph,graph_path)
record_artifact(graph_path)
assert np.isclose(sum(pagerank.values()),1)
fig,ax=plt.subplots(figsize=(10,7))
positions=nx.spring_layout(undirected,seed=SEED,iterations=40)
nx.draw_networkx_nodes(graph,positions,node_color=[community_map[node] for node in graph.nodes],node_size=[80+3000*pagerank[node] for node in graph.nodes],ax=ax,cmap="tab20")
nx.draw_networkx_edges(graph,positions,alpha=0.08,arrows=False,ax=ax)
ax.axis("off")
save_figure(fig,"transaction_network")
result=finish({"accounts":graph.number_of_nodes(),"edges":graph.number_of_edges(),"communities":len(communities),"high_value_cycles":len(cycles)},[accounts,community_summary])
