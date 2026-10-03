from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
scripts = sorted((root / "scripts").glob("*.py"))
failures = []
for script in scripts:
    print(f"RUNNING {script.name}", flush=True)
    completed = subprocess.run([sys.executable, str(script)], cwd=root, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        failures.append({"script": script.name, "returncode": completed.returncode, "stderr": completed.stderr[-4000:]})
if failures:
    print(json.dumps({"scripts_checked": len(scripts), "status": "failed", "failures": failures}, indent=2))
    sys.exit(1)
print(json.dumps({"scripts_checked": len(scripts), "status": "passed"}, indent=2))
