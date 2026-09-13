"""Resolve every path once, from the environment.

WHY THIS EXISTS. Scripts in this project used to open with

    CODE = '/Users/someone/Desktop/Research/.../code'
    BASE = f'{CODE}/scenefun3d 2'

which meant they ran on exactly one machine, and -- worse -- that the package
under `BASE` was a *copy* living inside a gitignored data directory. That copy
drifted from the repository and silently reverted two bug fixes. Paths come from
the environment now so there is one package and one data root, both nameable.

    SF3D_ROOT     data root (scans, caches, exemplar DB). Required.
    SF3D_TOOLKIT  the official SceneFun3D toolkit, for its evaluator. Optional;
                  only the scripts that call evaluate() need it.
    ARTICULATE3D_ROOT  the Articulate3D release. Optional; only the cross-dataset
                  transfer scripts need it.
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)                       # so `import sf3d` works
_HERE = os.path.join(REPO, 'scripts')
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)                      # so scripts import each other

DATA_ROOT = os.environ.get('SF3D_ROOT', os.path.expanduser('~/scenefun3d'))
TOOLKIT   = os.environ.get('SF3D_TOOLKIT', '')
A3D_ROOT  = os.environ.get('ARTICULATE3D_ROOT',
                           os.path.expanduser('~/articulate3d'))
A3D_VAL   = os.path.join(A3D_ROOT, 'processed',
                         'articulate3d_challenge_mov', 'validation')
DOCS      = os.path.join(REPO, 'docs')
FIGURES   = os.path.join(DOCS, 'figures')

def require_data(*subpaths):
    """Fail with the fix, not with a traceback forty lines later."""
    missing = [p for p in (os.path.join(DATA_ROOT, s) for s in subpaths)
               if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            'Missing data under SF3D_ROOT=%s\n  %s\n\n'
            'Set SF3D_ROOT to your data root, or see README.md "Data layout".'
            % (DATA_ROOT, '\n  '.join(missing)))
