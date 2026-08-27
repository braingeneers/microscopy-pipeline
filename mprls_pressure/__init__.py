"""Smoothing and pressure-gradient analysis for MPRLS bioreactor pressure waveforms.

Three MPRLS sensors record a pulsatile-perfused bioreactor -- an **input** and an
**output** flow-through channel (joined externally by a long U-bend) plus a
**midpoint** channel. The recordings are buried in broadband vibration noise, but
the pulse itself is highly structured: quasi-periodic at ~105 bpm and multiphasic,
so its energy sits in the fundamental and a few harmonics while the vibration is
broadband. This package smooths "based on what it likely is" -- band-limiting to
the pulse fundamental + harmonics and beat-synchronous averaging -- and computes
the input-output pressure gradient.

CLI::

    mp-pressure-gradient -i recording.csv -o results/

Python::

    from mprls_pressure import analyze_pressure_csv
    a = analyze_pressure_csv("recording.csv")
    print(a.summary())
    print(a.gradient.mean_gradient, "mmHg mean;",
          a.gradient.pulsatile_amplitude, "mmHg pulsatile")

See :func:`mprls_pressure.pressure.smooth_pressure` and
:func:`mprls_pressure.pressure.ensemble_average` for the two smoothers, and
:func:`mprls_pressure.pressure.pressure_gradient` for the gradient.
"""

from .pressure import (  # noqa: F401
    CHANNEL_COLUMNS,
    ChannelResult,
    DEFAULT_BPM_RANGE,
    DEFAULT_N_HARMONICS,
    Ensemble,
    GradientResult,
    PressureAnalysis,
    analyze_channel,
    analyze_pressure_csv,
    analyze_pressure_file,
    cli,
    cross_coherence,
    detect_beats,
    ensemble_average,
    estimate_fundamental,
    fill_dropouts,
    hampel_filter,
    load_pressure_csv,
    plot_analysis,
    pressure_gradient,
    roll_to_foot,
    smooth_pressure,
    to_uniform_grid,
)

__all__ = [
    "CHANNEL_COLUMNS",
    "ChannelResult",
    "DEFAULT_BPM_RANGE",
    "DEFAULT_N_HARMONICS",
    "Ensemble",
    "GradientResult",
    "PressureAnalysis",
    "analyze_channel",
    "analyze_pressure_csv",
    "analyze_pressure_file",
    "cli",
    "cross_coherence",
    "detect_beats",
    "ensemble_average",
    "estimate_fundamental",
    "fill_dropouts",
    "hampel_filter",
    "load_pressure_csv",
    "plot_analysis",
    "pressure_gradient",
    "roll_to_foot",
    "smooth_pressure",
    "to_uniform_grid",
]
