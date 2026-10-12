"""Append one reproducibility record per run to results/runs.jsonl (tracked in git).
Fields: time, script, argv (exact command), seed, derived seeds, git commit (+dirty flag), host, device, library versions, outputs."""
import sys, os, json, time, socket, subprocess, platform
def _git(*a):
    try: return subprocess.check_output(['git', *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def log_run(seed, outputs=(), extra=None, path='results/runs.jsonl'):
    import numpy, scipy, sklearn, torch
    rec = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), script=sys.argv[0], argv=sys.argv[1:], command='python ' + ' '.join(sys.argv),
               seed=seed, git_commit=_git('rev-parse', 'HEAD'), git_dirty=bool(_git('status', '--porcelain', '--', 'code')),
               host=socket.gethostname(), python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, torch=torch.__version__, cuda=torch.cuda.is_available(),
               outputs=list(outputs), **(extra or {}))
    with open(path, 'a') as f: f.write(json.dumps(rec) + '\n')
