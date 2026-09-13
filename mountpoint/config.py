"""Paths and tunables. Every constant in the method lives here, and nowhere else.

Paths resolve from the environment so the package runs anywhere:
    MOUNTPOINT_DATA       SceneFun3D root (required)
    SCENEFUN3D_TOOLKIT    official evaluator (segmentation metrics only)
    ARTICULATE3D_DATA     cross-dataset transfer only
"""
import os

ROOT    = os.environ.get('MOUNTPOINT_DATA', os.path.expanduser('~/scenefun3d'))
DATA    = f'{ROOT}/data'            # <visit>/<visit>_laser_scan.ply, _annotations.json
FRAMES  = f'{ROOT}/frames'          # posed RGB-D, 2D stage only
CACHE   = f'{ROOT}/cache'           # per-element points + neighbourhoods
DB_PATH = f'{ROOT}/exemplar_db.pkl'
CKPT    = f'{ROOT}/dinov3_vitl16.pth'
TOOLKIT = os.environ.get('SCENEFUN3D_TOOLKIT', '')   # official evaluator
A3D     = os.environ.get('ARTICULATE3D_DATA', '')    # cross-dataset transfer

CLIP_CACHE = f'{ROOT}/clip_embs.npz'
HUB_ID     = 'facebook/dinov3-vitl16-pretrain-lvd1689m'

def require(*paths):
    """Fail with the fix, not with a traceback forty lines later."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit('Missing:\n  ' + '\n  '.join(missing) +
                         f'\n\nMOUNTPOINT_DATA is {ROOT!r}. See README.')

# ---- measured; see notebook header for evidence ----
LAYER       = 1        # tied set {1,2,16}
CENTER      = False    # costs 8 pts of peak-in-GT
BANK_SHORT  = 518
QUERY_SHORT = 518      # deliberately equal: the 518/1036 mismatch was an accident
K_EX        = 3
TOPQ        = 0.10
BOX_PAD     = 0.25     # the only padding; hit tests use pad=0
GD_TOPK     = 8
GD_CONF     = 0.25
MULTIBOX_K  = 2
MAX_AREA_FRAC = 0.02   # masks larger than this are parent objects, not elements
SAM_AREA_MULT = 15.0   # NOT redundant -- that claim assumed SAM_BOX_PROMPT=True, now rejected
SAM_BOX_PROMPT = False # MEASURED NEGATIVE on CUDA: n=80 full IoU 0.101 (on) vs 0.175 (off),
                       # ceiling 0.143 vs 0.251. The MPS n=30 signal did not transfer. See header.

MEDIAN_AREA = None     # filled by pipeline.init_stats(db); see notebook

PARENT_HINT = {
  'hook_pull' : 'cabinet. drawer. door. fridge. oven.',
  'pinch_pull': 'drawer. cabinet. handle.',
  'hook_turn' : 'door. window. lever.',
  'rotate'    : 'knob. dial. valve. faucet.',
  'key_press' : 'switch. button. keypad.',
  'tip_push'  : 'button. switch. panel.',
  'foot_push' : 'pedal. bin. floor.',
  'unplug'    : 'socket. outlet. plug.',
  'plug_in'   : 'socket. outlet. plug.',
}
DEFAULT_HINT = 'handle. knob. switch. button.'

FAMILY = {'hook_pull':'pull','pinch_pull':'pull','unplug':'pull','plug_in':'pull',
          'rotate':'turn','hook_turn':'turn',
          'key_press':'press','tip_push':'press','foot_push':'press'}