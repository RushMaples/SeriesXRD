"""Display labels for internal unit codes and channel names.

Internal codes (``q_A^-1``, ``2th_deg``) stay exactly as pyFAI and the HDF5
files spell them; everything a *person* reads gets proper notation. One
place, so an axis is never labeled ``q_A^-1`` in one tab and ``q (Å⁻¹)`` in
another.
"""
from __future__ import annotations

_UNIT_LABELS = {
    "q_a^-1": "q (Å⁻¹)",
    "q_nm^-1": "q (nm⁻¹)",
    "2th_deg": "2θ (°)",
    "2th_rad": "2θ (rad)",
    "r_mm": "r (mm)",
    "d*2_a^-2": "d*² (Å⁻²)",
}

AZIMUTH_LABEL = "Azimuth (°)"
INTENSITY_LABEL = "Intensity (counts)"
INTENSITY_ARB_LABEL = "Intensity (a.u.)"
D_SPACING_LABEL = "d-spacing (Å)"
PRESSURE_LABEL = "Pressure (GPa)"
FRAME_LABEL = "Frame index"
# The contamination score is Σ max(mean − robust, 0) per frame: the
# integrated positive diamond-spot residual. Unitless; compare within a
# series, not across experiments.
CONTAMINATION_LABEL = "Contamination score"


def unit_label(unit: str | None) -> str:
    """Human axis label for an internal radial-unit code.

    Unknown codes pass through unchanged (better a raw code than a wrong
    label); an empty unit falls back to a generic axis name.
    """
    u = str(unit or "").strip()
    if not u:
        return "Radial coordinate"
    return _UNIT_LABELS.get(u.lower(), u)
