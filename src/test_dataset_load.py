import fcntl as _fcntl
_orig_flock = _fcntl.flock
def _safe_flock(fd, op):
    try:
        _orig_flock(fd, op)
    except OSError as e:
        if e.errno in (11, 13, 37):
            pass
        else:
            raise
_fcntl.flock = _safe_flock
import filelock as _filelock
_filelock.FileLock = _filelock.SoftFileLock

import os
import time

print("HF_HOME:", os.environ.get("HF_HOME", "not set"))
print("HF_DATASETS_OFFLINE:", os.environ.get("HF_DATASETS_OFFLINE", "not set"))
print()

from datasets import load_dataset

print("=== 测试 gsm8k ===")
t0 = time.time()
try:
    dataset = load_dataset('gsm8k', 'main', split='test')
    print(f"OK，耗时 {time.time()-t0:.1f}s，样本数: {len(dataset)}")
except Exception as e:
    print(f"FAIL ({time.time()-t0:.1f}s): {e}")

print()
print("=== 测试 mmlu ===")
t0 = time.time()
try:
    dataset = load_dataset('cais/mmlu', 'all', split='test')
    print(f"OK，耗时 {time.time()-t0:.1f}s，样本数: {len(dataset)}")
except Exception as e:
    print(f"FAIL ({time.time()-t0:.1f}s): {e}")
