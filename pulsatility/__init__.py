"""Pulsatility quantification for brightfield / neurosurgical microscopy video.

A self-contained tool that turns a video of perfused parenchyma (a gel block,
or -- in the intended follow-up -- exposed brain) into a **bulk pulsatility
waveform**: a continuous wave describing how the tissue moves with each pulse of
the perfusing flow, plus the pulse rate, a pulsatility index, and a spatial map
of where the parenchyma pulses most.

The physics
-----------
The parenchyma is perfused from above by a *pulsatile* flow through embedded
vessels. Every pulse briefly pressurises the vessels and shoves/relaxes the
surrounding tissue. In a brightfield video this shows up as tiny, spatially
coherent **motion** of the parenchyma. Measuring that motion frame-to-frame
gives a "tissue-speed" signal that rises and falls once per pulse -- the
continuous wave we want to quantify. Because motion is measured directly it is
robust to the oblique filming angle and to slow illumination drift (which
corrupts a naive brightness signal).

Usage
-----
From the command line::

    mp-pulsatility -i Pulsatility_video.mp4 -o results/

From Python::

    from pulsatility import analyze_pulsatility_video
    result = analyze_pulsatility_video("Pulsatility_video.mp4", "results/")
    print(result.summary())

See :func:`pulsatility.pulsatility.analyze_pulsatility` for the array-level core.
"""

from .pulsatility import (  # noqa: F401
    MAX_ROIS,
    PulsatilityResult,
    Region,
    RespirationResult,
    analyze_pulsatility,
    analyze_pulsatility_video,
    build_roi_masks,
    decompose_respiration,
    load_video_gray,
    motion_signal,
    motion_signals,
    plot_breathing_decomposition,
    plot_pulsatility,
    plot_pulsatility_comparison,
    plot_pulsatility_multi,
    stabilize_frames,
    cli,
)

__all__ = [
    "MAX_ROIS",
    "PulsatilityResult",
    "Region",
    "RespirationResult",
    "analyze_pulsatility",
    "analyze_pulsatility_video",
    "build_roi_masks",
    "decompose_respiration",
    "load_video_gray",
    "motion_signal",
    "motion_signals",
    "plot_breathing_decomposition",
    "plot_pulsatility",
    "plot_pulsatility_comparison",
    "plot_pulsatility_multi",
    "stabilize_frames",
    "cli",
]
