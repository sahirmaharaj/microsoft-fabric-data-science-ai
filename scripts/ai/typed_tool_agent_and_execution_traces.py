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
WORKFLOW = "43_typed_tool_agent_and_execution_traces"


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


import ast
import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

orders=pd.DataFrame({"region":rng.choice(["north","south","west"],SAMPLE_SIZE),"revenue":rng.gamma(3,80,SAMPLE_SIZE),"cost":rng.gamma(2,40,SAMPLE_SIZE)})

@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict
    call_id: str

@dataclass
class AgentState:
    query: str
    status: str="created"
    calls: list=field(default_factory=list)
    observations: list=field(default_factory=list)
    answer: object=None

operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}

def calculate(expression):
    if len(expression)>200:
        raise ValueError("expression_length")
    tree=ast.parse(expression,mode="eval")
    if sum(1 for _ in ast.walk(tree))>50:
        raise ValueError("expression_complexity")
    def evaluate(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            value=float(node.value)
        elif isinstance(node,ast.BinOp) and type(node.op) in operations:
            value=operations[type(node.op)](evaluate(node.left),evaluate(node.right))
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            value=-evaluate(node.operand)
        else:
            raise ValueError("unsupported_expression")
        if not np.isfinite(value) or abs(value)>1e15:
            raise ValueError("numeric_range")
        return value
    return {"value":evaluate(tree.body)}



def aggregate_orders(region,metric):
    if region not in {"north","south","west","all"}:
        raise ValueError("invalid_region")
    if metric not in {"revenue","cost","profit","count"}:
        raise ValueError("invalid_metric")
    frame=orders if region=="all" else orders.loc[orders.region.eq(region)]
    value=len(frame) if metric=="count" else (frame.revenue-frame.cost).sum() if metric=="profit" else frame[metric].sum()
    return {"region":region,"metric":metric,"value":float(value),"rows":len(frame)}

def describe_schema():
    return {"columns":orders.dtypes.astype(str).to_dict(),"regions":sorted(orders.region.unique().tolist()),"rows":len(orders)}

tools_registry={"calculate":calculate,"aggregate_orders":aggregate_orders,"describe_schema":describe_schema}
cache={}
traces=[]

def execute_call(call):
    if call.tool not in tools_registry:
        raise ValueError("unregistered_tool")
    key=hashlib.sha256(json.dumps({"tool":call.tool,"arguments":call.arguments},sort_keys=True).encode()).hexdigest()
    cached=key in cache
    started=time.perf_counter()
    if not cached:
        cache[key]=tools_registry[call.tool](**call.arguments)
    observation=cache[key]
    traces.append({"call_id":call.call_id,"tool":call.tool,"cached":cached,"elapsed_ms":1000*(time.perf_counter()-started),"result_hash":hashlib.sha256(json.dumps(observation,sort_keys=True).encode()).hexdigest()})
    return observation

def plan_query(query):
    if query.startswith("calculate:"):
        return [ToolCall("calculate",{"expression":query.split(":",1)[1].strip()},uuid.uuid4().hex)]
    if query=="schema":
        return [ToolCall("describe_schema",{},uuid.uuid4().hex)]
    parts=query.lower().split()
    if len(parts)==2 and parts[0] in {"revenue","cost","profit","count"}:
        return [ToolCall("aggregate_orders",{"metric":parts[0],"region":parts[1]},uuid.uuid4().hex)]
    return []



def run_agent(query):
    state=AgentState(query=query)
    calls=plan_query(query)
    state.calls=[asdict(call) for call in calls]
    state.status="executing" if calls else "unsupported_query"
    for call in calls[:4]:
        state.observations.append(execute_call(call))
    state.answer=state.observations[-1] if state.observations else {"supported_commands":["schema","revenue north","profit all","calculate: (12+3)*4"]}
    if calls:
        state.status="completed"
    return asdict(state)

queries=["schema","revenue north","profit all","calculate: (12+3)*4","revenue north","unknown request"]
results=[run_agent(query) for query in queries]
rejections=[]
for expression in ["open('file')","__import__('os')","2**10000"]:
    try:
        calculate(expression)
    except ValueError as error:
        rejections.append({"expression":expression,"error":str(error)})
assert len(rejections)==3
assert results[3]["answer"]["value"]==60
assert any(trace["cached"] for trace in traces)
save_json(results,"agent_results")
save_table(pd.DataFrame(traces),"execution_trace")
save_table(pd.DataFrame(rejections),"rejected_expressions")
save_json({"planner":"deterministic_command_router","registered_tools":list(tools_registry),"max_steps":4,"arbitrary_code_execution":False},"agent_configuration")
result=finish({"queries":len(queries),"completed":sum(item["status"]=="completed" for item in results),"tool_calls":len(traces),"cache_hits":sum(item["cached"] for item in traces)},[pd.DataFrame(traces)])
