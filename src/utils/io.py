"""Reading and writing results, and the record of how each result was produced.

Conventions
-----------
* Numbers go to JSON, arrays go to compressed `.npz`.
* Keys that start with an underscore (for example '_net', '_weights') are in-memory
  handles and are never written.
* Every JSON written by an experiment carries a 'meta' block from `run_metadata`:
  time, package versions, device, git commit and the study configuration.
"""
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import jax

import config as cfg


# ---- JSON -----------------------------------------------------------------------------------
def to_serialisable(obj):
    """Convert nested results (NumPy/JAX arrays and scalars, tuples, paths) to plain JSON types.

    Non-finite numbers become the strings 'inf', '-inf' and 'nan' so the files stay
    valid JSON; an infinite condition number is a result, not an error.
    """
    if isinstance(obj, dict):
        return {str(key): to_serialisable(value) for key, value in obj.items() if not str(key).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(value) for value in obj]
    if isinstance(obj, (np.ndarray, jax.Array)):
        return to_serialisable(np.asarray(obj).tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else str(float(obj))
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, obj):
    """Write a result dict as indented JSON, creating the folder if needed. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serialisable(obj), indent=2))
    return path


def _restore_non_finite(obj):
    if isinstance(obj, dict):
        return {key: _restore_non_finite(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_restore_non_finite(value) for value in obj]
    return float(obj) if obj in ('inf', '-inf', 'nan') else obj


def load_json(path):
    """Read a JSON result; 'inf', '-inf' and 'nan' become floats again."""
    return _restore_non_finite(json.loads(Path(path).read_text()))


# ---- arrays ---------------------------------------------------------------------------------
def save_arrays(path, **arrays):
    """Write named arrays to a compressed .npz file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in arrays.items()})
    return path


def load_arrays(path):
    """Read a .npz file into a dict of arrays."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def save_weights(path, weights):
    """Write trained network weights as one flat vector (.npz)."""
    from jax.flatten_util import ravel_pytree
    return save_arrays(path, flat=np.asarray(ravel_pytree(weights)[0]))


def load_weights(path, net):
    """Read weights written by `save_weights` into the structure of `net`."""
    from jax.flatten_util import ravel_pytree
    _, unravel = ravel_pytree(net.init(0))
    return unravel(jax.numpy.asarray(load_arrays(path)['flat']))


# ---- provenance -----------------------------------------------------------------------------
def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=cfg.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_name():
    """The device JAX computes on, e.g. 'gpu (NVIDIA A100-SXM4-40GB)' or 'cpu'."""
    device = jax.devices()[0]
    kind = getattr(device, 'device_kind', '')
    return f'{device.platform} ({kind})' if kind and kind != device.platform else device.platform


def run_metadata(**extra):
    """What is needed to trace a result back to the code and settings that produced it."""
    import flax
    import optax
    study = {name: getattr(cfg, name) for name in (
        'TRUE_PARAMS', 'TRAIN_DOSES', 'HELDOUT_DOSE', 'NOISE_SD', 'N_OBS', 'T_WINDOW', 'L', 'T', 'SIGMA', 'C0',
        'NX_FD', 'DT_FD', 'DT_STORE', 'LAYERS', 'W_INVERSE', 'W_FORWARD', 'STAGE_SCHEDULE')}
    return {'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'git_commit': _git_commit(), 'device': device_name(), 'python': platform.python_version(),
            'versions': {'jax': jax.__version__, 'flax': flax.__version__, 'optax': optax.__version__,
                         'numpy': np.__version__},
            'float64': bool(jax.config.jax_enable_x64), 'config': study, **extra}
