"""Copy a data folder from the Drive mount to local disk and verify it (Drive FUSE copies can come back truncated).
Every file is compared by size with the source and re-copied up to 5 times; every .npz/.npy must then open.
  python safe_copy.py <src_dir> <dst_dir>      exit code 1 if anything is still bad
"""
import os, shutil, sys, time, zipfile
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
files = [os.path.relpath(os.path.join(r, f), src) for r, _, fs in os.walk(src) for f in fs]


def ok(rel):
    s, d = os.path.join(src, rel), os.path.join(dst, rel)
    if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s):
        return False
    try:
        if d.endswith(".npz"):
            with zipfile.ZipFile(d) as z:
                return z.testzip() is None
        if d.endswith(".npy"):
            np.load(d, mmap_mode="r")
    except Exception:
        return False
    return True


bad = files
for attempt in range(5):
    for rel in bad:
        d = os.path.join(dst, rel); os.makedirs(os.path.dirname(d), exist_ok=True)
        try:
            shutil.copyfile(os.path.join(src, rel), d)
        except Exception as e:
            print("copy error", rel, e, flush=True)
    bad = [r for r in bad if not ok(r)]
    print(f"pass {attempt + 1}: {len(files)} files, {len(bad)} bad", flush=True)
    if not bad:
        break
    time.sleep(30)
gb = sum(os.path.getsize(os.path.join(dst, r)) for r in files) / 1e9
print(f"copied {src} -> {dst}: {len(files)} files {gb:.1f} GB, bad {bad[:5]}", flush=True)
sys.exit(1 if bad else 0)
