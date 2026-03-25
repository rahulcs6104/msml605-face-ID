import json
import os
import subprocess
from datetime import datetime, timezone


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def log_run(runs_dir: str, run_id: str, config_name: str,split: str, data_version: str, threshold: float,metrics: dict, note: str = "") :
    os.makedirs(runs_dir, exist_ok=True)
    record = {"run_id":run_id,"timestamp": datetime.now(timezone.utc).isoformat(), "commit":_git_hash(),"config":config_name,"split":split,"data_version": data_version,"threshold":threshold,"metrics":metrics,"note":note,}
    with open(os.path.join(runs_dir,f"{run_id}.json"),"w") as f:
        json.dump(record, f, indent=2)

    summary_path = os.path.join(runs_dir, "run_summary.json")

    if os.path.exists(summary_path):
        file = open(summary_path, "r")
        summary = json.load(file)
        file.close()
    else:
        summary = []

    summary.append(record)
    with open(summary_path, "w") as f:
        json.dump(summary,f,indent=2)

    print(f"[tracking] =>'{run_id}' and logged => {runs_dir}/{run_id}.json")
    return record

def load_summary(runs_dir):
    path = os.path.join(runs_dir,"run_summary.json")
    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
        return data
    else:
        return []