"""Download every subject-0 MEG run (h5) and events tsv from pnpl/LibriBrain and pnpl/LibriBrain2.
Files land under data/raw/<RepoName>/<repo path>; existing files with matching size are skipped."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
REPOS = {"pnpl/LibriBrain": "LibriBrain", "pnpl/LibriBrain2": "LibriBrain2"}

def listing(repo):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    out = []
    while url:
        r = urllib.request.urlopen(url); out += json.load(r)
        link = r.headers.get("Link"); url = None
        if link and 'rel="next"' in link:
            url = link.split("<")[1].split(">")[0]
    return out

jobs = []
for repo, local in REPOS.items():
    for f in listing(repo):
        p = f["path"]
        if f.get("type") != "file" or "sub-0_" not in p:
            continue
        if not (p.endswith(".h5") or p.endswith("_events.tsv")):
            continue
        jobs.append((repo, local, p, f.get("size", 0)))
print(len(jobs), "files", round(sum(j[3] for j in jobs) / 1e9, 1), "GB", flush=True)

def get(job):
    repo, local, p, size = job
    dest = ROOT / local / p
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    for attempt in range(5):
        try:
            hf_hub_download(repo_id=repo, filename=p, repo_type="dataset", local_dir=str(ROOT / local))
            assert dest.stat().st_size == size, (dest, dest.stat().st_size, size)
            return "ok"
        except Exception as e:
            print("retry", p, attempt, repr(e)[:200], flush=True); time.sleep(10)
    return "FAIL " + p

t0 = time.time(); done = 0
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(get, sorted(jobs, key=lambda j: (not j[2].endswith(".tsv"), j[2]))):
        done += 1
        if r.startswith("FAIL") or done % 10 == 0:
            print(done, r, round(time.time() - t0), "s", flush=True)
print("DONE", round(time.time() - t0), "s", flush=True)
