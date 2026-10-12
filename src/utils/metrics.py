"""Error measures and summaries over repeated noise realizations."""
import numpy as np


# ---- field errors -----------------------------------------------------------------------
def relative_l2(estimate, reference):
    """||estimate - reference||_2 / ||reference||_2 over all entries."""
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(reference)) / np.linalg.norm(reference))


def max_abs(estimate, reference):
    """L-infinity error: largest absolute difference."""
    return float(np.max(np.abs(np.asarray(estimate) - np.asarray(reference))))


def field_errors(estimate, reference):
    """Relative L2, absolute L-infinity, and L-infinity relative to max |reference|."""
    return {'rel_l2': relative_l2(estimate, reference), 'linf': max_abs(estimate, reference),
            'rel_linf': max_abs(estimate, reference) / float(np.max(np.abs(reference)))}


# ---- parameter errors ---------------------------------------------------------------------
def percent_error(estimate, truth):
    """Signed relative error in percent."""
    return float(100.0 * (estimate - truth) / truth)


def parameter_errors(estimate, truth, names=None):
    """{name: signed percent error} for the parameters in `names` (default: all in estimate)."""
    return {name: percent_error(estimate[name], truth[name]) for name in (names or estimate)}


# ---- repeated realizations ----------------------------------------------------------------
def summarise(values):
    """Mean, SD, median, quartiles and range of a sample; NaN entries are ignored."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'n': 0}
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {'n': int(values.size), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if values.size > 1 else 0.0,
            'median': float(q50), 'q25': float(q25), 'q75': float(q75),
            'min': float(values.min()), 'max': float(values.max()),
            'mean_abs': float(np.abs(values).mean())}


def summarise_errors(error_dicts):
    """[{name: error}, ...] over realizations -> {name: summarise(errors)}."""
    names = error_dicts[0].keys()
    return {name: summarise([errors[name] for errors in error_dicts]) for name in names}


def interval_coverage(intervals, truth):
    """Fraction of realizations whose interval (lower, upper) contains the true value."""
    inside = [lower <= truth <= upper for lower, upper in intervals]
    return float(np.mean(inside))
