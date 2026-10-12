"""Figures of the paper.

One function per figure of the manuscript plan; each takes plain arrays or the result
dicts of `src/`, returns the Matplotlib figure and, if `path` is given, writes it as
PNG and PDF.

    plot_convergence               solver verification (supplement)
    plot_forward_agreement         Figure 2  forward solution, FD against PINN, per dose
    plot_identifiability           Figure 3  sensitivity spectra and correlation maps
    plot_parameter_recovery        Figure 4  recovery over noise realizations
    plot_profiles                  Figure 5  profile likelihoods per design
    plot_counterfactual_maps       Figure 6  held-out effect maps
    plot_counterfactual_uncertainty  Figure 7  effect bands under parameter ambiguity

Figure 1 (the workflow schematic) is drawn by hand and is not produced here.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from src.model.parameters import LABELS, PARAM_NAMES

# Colour-blind-safe palette (Okabe & Ito).
COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']

STYLE = {'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
         'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight'}


def _finish(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix('.png'))
        fig.savefig(path.with_suffix('.pdf'))
    return fig


def _extent(solver):
    half = solver.L / 2
    return (-half, half, -half, half)


def _map(ax, field, solver, **kwargs):
    """Draw field[ix, iy] with x horizontal and y vertical."""
    return ax.imshow(np.asarray(field).T, origin='lower', extent=_extent(solver), **kwargs)


def plot_convergence(spatial, temporal, path=None):
    """Self-convergence errors of the reference solver with second-order guide lines."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), layout='constrained')
        for ax, steps, errors, label in ((axes[0], spatial['dx'][:-1], spatial['error'], r'$\Delta x$ ($\mu$m)'),
                                         (axes[1], temporal['dt'], temporal['error'], r'$\Delta t$ (ms)')):
            steps, errors = np.asarray(steps), np.asarray(errors)
            ax.loglog(steps, errors, 'o-', color=COLOURS[0], label='solver')
            ax.loglog(steps, errors[0] * (steps / steps[0])**2, '--', color='0.5', label='second order')
            ax.set(xlabel=label, ylabel='relative $L^2$ error')
            ax.legend(frameon=False)
    return _finish(fig, path)


def plot_forward_agreement(reference, network, solver, t=5.0, time_stride=1, path=None):
    """Rows: finite difference, PINN, absolute difference. Columns: doses.

    reference, network: {dose: field (n_times, nx, nx)} on the same frames.
    """
    doses = sorted(reference)
    i = int(round(t / (solver.dt_store * time_stride)))
    vmax = max(float(reference[A][i].max()) for A in doses)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, len(doses), figsize=(1.7 * len(doses) + 0.8, 5.0), sharex=True, sharey=True,
                                 squeeze=False, layout='constrained')
        for j, A in enumerate(doses):
            top = _map(axes[0, j], reference[A][i], solver, vmin=0, vmax=vmax, cmap='viridis')
            _map(axes[1, j], network[A][i], solver, vmin=0, vmax=vmax, cmap='viridis')
            diff = _map(axes[2, j], np.abs(network[A][i] - reference[A][i]), solver, cmap='magma')
            axes[0, j].set_title(f'$A = {A:g}$')
            fig.colorbar(diff, ax=axes[2, j], orientation='horizontal', pad=0.25, format='%.0e')
        for ax, label in zip(axes[:, 0], ('FD', 'PINN', '|difference|')):
            ax.set_ylabel(f'{label}\n$y$ ($\\mu$m)')
        for ax in axes[-1]:
            ax.set_xlabel(r'$x$ ($\mu$m)')
        fig.colorbar(top, ax=axes[:2, :], label=rf'$C$ ($\mu$M) at $t = {t:g}$ ms', shrink=0.8)
    return _finish(fig, path)


def plot_identifiability(spectra, correlations, names=PARAM_NAMES, path=None):
    """Left: singular values of each design. Right: one sensitivity-correlation map per design.

    spectra: {label: singular values}; correlations: {label: (p, p) matrix}. A design
    with fewer parameters than `names` (the baseline) is padded with blanks.
    """
    labels = list(correlations)
    ticks = [LABELS[name] for name in names]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 1 + len(labels), figsize=(3.0 + 2.2 * len(labels), 2.8),
                                 gridspec_kw={'width_ratios': [1.4] + [1] * len(labels)}, layout='constrained')
        for k, (label, values) in enumerate(spectra.items()):
            values = np.asarray(values, float)
            axes[0].semilogy(np.arange(1, len(values) + 1), np.where(values > 0, values, np.nan), 'o-',
                             color=COLOURS[k % len(COLOURS)], label=label)
        axes[0].set(xlabel='index', ylabel='singular value', xticks=range(1, len(names) + 1))
        axes[0].legend(frameon=False)
        for ax, label in zip(axes[1:], labels):
            matrix = np.full((len(names), len(names)), np.nan)
            block = np.asarray(correlations[label], float)
            matrix[:block.shape[0], :block.shape[1]] = block
            image = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set(title=label, xticks=range(len(names)), yticks=range(len(names)))
            ax.set_xticklabels(ticks, rotation=45)
            ax.set_yticklabels(ticks if ax is axes[1] else [])
            ax.spines[:].set_visible(False)
        fig.colorbar(image, ax=axes[1:], label='sensitivity correlation', shrink=0.85)
    return _finish(fig, path)


def plot_parameter_recovery(errors, names=PARAM_NAMES, path=None):
    """Signed percent error of each parameter over noise realizations, one box per method.

    errors: {method: {name: [percent errors over realizations]}}.
    """
    methods = list(errors)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(names), figsize=(1.6 * len(names) + 0.6, 2.8), squeeze=False,
                                 layout='constrained')
        for ax, name in zip(axes[0], names):
            for k, method in enumerate(methods):
                values = np.asarray(errors[method].get(name, []), float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                colour = COLOURS[k % len(COLOURS)]
                ax.boxplot(values, positions=[k], widths=0.5, showfliers=False,
                           medianprops={'color': colour}, boxprops={'color': colour},
                           whiskerprops={'color': colour}, capprops={'color': colour})
                jitter = np.random.default_rng(0).uniform(-0.15, 0.15, values.size)
                ax.plot(k + jitter, values, '.', color=colour, alpha=0.6, markersize=4)
            ax.axhline(0.0, color='0.5', linewidth=0.8, zorder=0)
            ax.set(title=LABELS[name], xticks=range(len(methods)), xlim=(-0.6, len(methods) - 0.4))
            ax.set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].set_ylabel('error (%)')
    return _finish(fig, path)


def plot_profiles(profiles, truth=None, path=None):
    """Profile likelihoods; one panel per parameter, one curve per design.

    profiles: {design label: {name: result of `profile_likelihood`}}.
    """
    names = [name for name in PARAM_NAMES if any(name in by_name for by_name in profiles.values())]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(names), figsize=(2.0 * len(names) + 0.4, 2.6), sharey=True, squeeze=False,
                                 layout='constrained')
        for ax, name in zip(axes[0], names):
            for k, (label, by_name) in enumerate(profiles.items()):
                if name in by_name:
                    profile = by_name[name]
                    ax.plot(profile['values'], profile['delta_chi2'], 'o-', markersize=3,
                            color=COLOURS[k % len(COLOURS)], label=label)
                    threshold = profile['threshold']
            ax.axhline(threshold, color='0.5', linestyle='--', linewidth=0.8)
            if truth is not None:
                ax.axvline(truth[name], color='0.5', linestyle=':', linewidth=0.8)
            if name != 'eta':
                ax.set_xscale('log')
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set(xlabel=LABELS[name], ylim=(0, 4 * threshold))
        axes[0, 0].set_ylabel(r'$\Delta\chi^2$')
        axes[0, 0].legend(frameon=False)
    return _finish(fig, path)


def plot_counterfactual_maps(tau_reference, tau_estimate, solver, times=(2.0, 10.0, 30.0), time_stride=1,
                             dose=0.5, path=None):
    """Rows: reference effect, estimated effect, difference. Columns: selected times."""
    indices = [int(round(t / (solver.dt_store * time_stride))) for t in times]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, len(times), figsize=(2.1 * len(times) + 0.8, 5.4), sharex=True, sharey=True,
                                 squeeze=False, layout='constrained')
        for j, (i, t) in enumerate(zip(indices, times)):
            vmax = float(max(np.max(np.abs(tau_reference[i])), np.max(np.abs(tau_estimate[i]))))
            for row, field in enumerate((tau_reference[i], tau_estimate[i])):
                image = _map(axes[row, j], field, solver, vmin=0, vmax=vmax, cmap='viridis')
            fig.colorbar(image, ax=axes[:2, j], orientation='horizontal', pad=0.08, format='%.1e')
            difference = tau_estimate[i] - tau_reference[i]
            bound = float(np.max(np.abs(difference))) or 1.0
            image = _map(axes[2, j], difference, solver, vmin=-bound, vmax=bound, cmap='RdBu_r')
            fig.colorbar(image, ax=axes[2, j], orientation='horizontal', pad=0.25, format='%.0e')
            axes[0, j].set_title(f'$t = {t:g}$ ms')
        for ax, label in zip(axes[:, 0], (rf'reference $\tau_{{{dose:g}}}$', rf'estimated $\tau_{{{dose:g}}}$',
                                          'difference')):
            ax.set_ylabel(f'{label}\n$y$ ($\\mu$m)')
        for ax in axes[-1]:
            ax.set_xlabel(r'$x$ ($\mu$m)')
    return _finish(fig, path)


def plot_counterfactual_uncertainty(summaries, times, reference_ate=None, dose=0.5, path=None):
    """Left: bands of ATE(t) per design. Right: interval of the global effect per design.

    summaries: {design label: result of `reliability_summary`}.
    """
    labels = list(summaries)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.7, 1]},
                                 layout='constrained')
        for k, label in enumerate(labels):
            band, colour = summaries[label]['ATE_band'], COLOURS[k % len(COLOURS)]
            axes[0].fill_between(times, band['lower'], band['upper'], color=colour, alpha=0.25, linewidth=0)
            axes[0].plot(times, band['median'], color=colour, label=label)
            gte = summaries[label]['GTE']
            axes[1].errorbar(k, gte['median'], yerr=[[gte['median'] - gte['lower']], [gte['upper'] - gte['median']]],
                             fmt='o', color=colour, capsize=3)
        if reference_ate is not None:
            axes[0].plot(times, reference_ate, 'k--', linewidth=1, label='truth')
            axes[1].axhline(np.trapezoid(reference_ate, times) / (times[-1] - times[0]), color='k', linestyle='--',
                            linewidth=1)
        axes[0].set(xlabel='$t$ (ms)', ylabel=rf'ATE$_{{{dose:g}}}(t)$ ($\mu$M)')
        axes[0].legend(frameon=False)
        axes[1].set(xticks=range(len(labels)), xlim=(-0.6, len(labels) - 0.4), ylabel=rf'GTE$_{{{dose:g}}}$ ($\mu$M)')
        axes[1].set_xticklabels(labels, rotation=30, ha='right')
    return _finish(fig, path)
