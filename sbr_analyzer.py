"""Touchstone single-bit response and reference equalizer analyzer.

Edit only the USER CONTROL section for normal use.  NumPy, SciPy, and
Matplotlib are required; scikit-rf is not required.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.linalg import solve
from scipy.signal import fftconvolve

# File-only backend: works on servers and Python installs without Tk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Young-Min Lee"
__version__ = "1.0.0"


# ============================================================
# USER CONTROL -- edit this section only
# ============================================================

S_PARAMETER_FILE = str(
    Path(__file__).resolve().parent /
    "sample_diff_channel_200Mbaud_minus6dB_at_nyquist.s4p"
)

CHANNEL_MODE = "DIFF"           # "SE" / "DIFF"

# Single-ended mode (Touchstone port numbers are 1-based)
TX_PORT = 1
RX_PORT = 2

# Differential mode (used only when CHANNEL_MODE = "DIFF")
TX_POS_PORT = 1
TX_NEG_PORT = 3
RX_POS_PORT = 2
RX_NEG_PORT = 4

RX_TERMINATION_OHM = None     # None = matched; SE: ohm, DIFF: differential ohm

SYMBOL_RATE_BAUD = 200.0E+06   # [Baud = symbols/s]

TX_PULSE_VOLTAGE = 1.0        # [V] at the channel input
TX_PULSE_WIDTH_UI = 1.0       # pulse width [UI], normally 1.0
TX_RISE_TIME = 0.0            # [s], 0 = ideal rectangular pulse
TX_FALL_TIME = 0.0            # [s], 0 = ideal rectangular pulse

CTLE_DB = 0.0                 # [dB], 0 = OFF; target = Nyquist frequency

MODE = "DFE"                # "NONE" / "FFE" / "DFE" / "BOTH"

FFE_MAX_PRE_TAPS = 3
FFE_MAX_POST_TAPS = 5
DFE_MAX_TAPS = 8
FFE_MAX_ABS_TAP_SUM = 1.0     # TX constraint: sum(abs(FFE taps)) <= this value
TAP_STOP_IMPROVEMENT_PERCENT = 1.0

CURSOR_PRE_COUNT = 5
CURSOR_POST_COUNT = 10
SAMPLING_PHASE_MODE = "AUTO" # "AUTO" / "MANUAL"
MANUAL_PHASE_UI = 0.5         # used only for MANUAL mode

TIME_AXIS_UNIT = "UI"        # "UI" / "s"
SHOW_TX_PULSE = True
SHOW_FFE_OUTPUT = True
SHOW_CHANNEL = True
SHOW_AFTER_FFE = True
SHOW_AFTER_CTLE = True
SHOW_AFTER_DFE = True
SHOW_SAMPLES = True

PRINT_ABSOLUTE_CURSOR = True
PRINT_NORMALIZED_CURSOR = True
PRINT_EQ_TAPS = True
PRINT_RESIDUAL_ISI = True
DISPLAY_DECIMAL_PLACES = 6    # Console/GUI display only, valid range: 1..15
PULSE_PLOT_CACHE = None       # populated after analysis for fast GUI trace toggling
SAVE_RESULT_CSV = True
SAVE_PLOT_PNG = True
OUTPUT_DIRECTORY = r"C:\Users\roy97\OneDrive\바탕 화면\Codex\PR_Result"

# ============================================================
# END OF USER CONTROL
# ============================================================


@dataclass
class Touchstone:
    frequency_hz: np.ndarray
    s: np.ndarray                 # shape: (frequency, receive port, source port)
    reference_ohm: float


def _port_count(path: Path) -> int:
    match = re.search(r"\.s(\d+)p$", path.name, re.IGNORECASE)
    if not match:
        raise ValueError("Touchstone filename must end in .sNp (for example .s2p or .s4p).")
    return int(match.group(1))


def read_touchstone(filename: str) -> Touchstone:
    """Read a Touchstone 1.x RI/MA/DB file (including wrapped data lines)."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"S-parameter file not found: {path}")
    ports = _port_count(path)
    unit_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
    freq_scale, data_format, reference = 1e9, "MA", 50.0
    numbers: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            raise ValueError("Touchstone 2.0 bracketed syntax is not supported by this compact parser.")
        if line.startswith("#"):
            tokens = line[1:].upper().split()
            for token in tokens:
                if token in unit_scale:
                    freq_scale = unit_scale[token]
                elif token in {"RI", "MA", "DB"}:
                    data_format = token
            if "R" in tokens:
                reference = float(tokens[tokens.index("R") + 1])
            continue
        numbers.extend(float(x.replace("D", "E").replace("d", "e")) for x in line.split())

    row_size = 1 + 2 * ports * ports
    if len(numbers) % row_size:
        raise ValueError(f"Touchstone numeric data is incomplete (expected {row_size} values per point).")
    rows = np.asarray(numbers, dtype=float).reshape(-1, row_size)
    f = rows[:, 0] * freq_scale
    pairs = rows[:, 1:].reshape(-1, ports * ports, 2)
    if data_format == "RI":
        values = pairs[..., 0] + 1j * pairs[..., 1]
    else:
        magnitude = pairs[..., 0] if data_format == "MA" else 10.0 ** (pairs[..., 0] / 20.0)
        values = magnitude * np.exp(1j * np.deg2rad(pairs[..., 1]))
    # Touchstone 1.x order is S11,S21,...,SN1,S12,... (column-major).
    s = values.reshape(-1, ports, ports, order="F")
    order = np.argsort(f)
    return Touchstone(f[order], s[order], reference)


def selected_transfer(ts: Touchstone) -> tuple[np.ndarray, float, complex]:
    """Return loaded transfer, applicable reference impedance, and load gamma.

    TX_PULSE_VOLTAGE is interpreted as the forward/incident voltage at the
    channel input. The load correction includes the RX total-voltage factor
    and repeated reflection between the load and channel output S22.
    """
    n = ts.s.shape[1]

    def index(port: int) -> int:
        if not 1 <= port <= n:
            raise ValueError(f"Port {port} is invalid for a {n}-port file.")
        return port - 1

    mode = CHANNEL_MODE.upper()
    if mode == "SE":
        tx, rx = index(TX_PORT), index(RX_PORT)
        s21 = ts.s[:, rx, tx]
        s22 = ts.s[:, rx, rx]
        reference_ohm = ts.reference_ohm
    elif mode == "DIFF":
        tp, tn = index(TX_POS_PORT), index(TX_NEG_PORT)
        rp, rn = index(RX_POS_PORT), index(RX_NEG_PORT)

        def sdd(receive_pos, receive_neg, source_pos, source_neg):
            return 0.5 * (ts.s[:, receive_pos, source_pos]
                          - ts.s[:, receive_pos, source_neg]
                          - ts.s[:, receive_neg, source_pos]
                          + ts.s[:, receive_neg, source_neg])

        s21 = sdd(rp, rn, tp, tn)
        s22 = sdd(rp, rn, rp, rn)
        reference_ohm = 2.0 * ts.reference_ohm
    else:
        raise ValueError('CHANNEL_MODE must be "SE" or "DIFF".')

    load_ohm = reference_ohm if RX_TERMINATION_OHM is None else float(RX_TERMINATION_OHM)
    if load_ohm <= 0:
        raise ValueError("RX_TERMINATION_OHM must be positive or None.")
    gamma_load = (load_ohm - reference_ohm) / (load_ohm + reference_ohm)
    denominator = 1.0 - s22 * gamma_load
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("RX load/channel combination is singular: 1 - S22*GammaL is near zero.")
    loaded = (1.0 + gamma_load) * s21 / denominator
    return loaded, reference_ohm, gamma_load


def ctle_breakpoints(boost_db: float, nyquist_hz: float):
    """Return Z1/P1/P2 so gain and its maximum occur at Nyquist.

    P1 is anchored at Nyquist/3. Z1 and P2 are solved from the requested
    Nyquist gain and the zero-slope condition at Nyquist.
    """
    if abs(boost_db) < 1e-12:
        return None
    gain = 10.0 ** (boost_db / 20.0)
    if gain <= 1.0:
        raise ValueError("Automatic Z1/P1/P2 CTLE requires CTLE_DB >= 0; use 0 for OFF.")
    p1_hz = nyquist_hz / 3.0
    p2_ratio_squared = (1.0 - 1.0 / (gain * gain)) / 9.0
    p2_hz = nyquist_hz / math.sqrt(p2_ratio_squared)
    z1_ratio_squared = 10.0 * gain * gain * (1.0 + p2_ratio_squared) - 1.0
    z1_hz = nyquist_hz / math.sqrt(z1_ratio_squared)
    return z1_hz, p1_hz, p2_hz


def ctle_response(frequency_hz: np.ndarray, boost_db: float, nyquist_hz: float) -> np.ndarray:
    """Z1/P1/P2 CTLE: DC=0 dB, peak at Nyquist, -20 dB/dec after P2."""
    breakpoints = ctle_breakpoints(boost_db, nyquist_hz)
    if breakpoints is None:
        return np.ones_like(frequency_hz, dtype=complex)
    zero_hz, pole1_hz, pole2_hz = breakpoints
    s = 1j * 2.0 * np.pi * frequency_hz
    return ((1.0 + s / (2 * np.pi * zero_hz)) /
            ((1.0 + s / (2 * np.pi * pole1_hz)) *
             (1.0 + s / (2 * np.pi * pole2_hz))))


def interpolate_transfer(f_in: np.ndarray, h_in: np.ndarray, f_out: np.ndarray) -> np.ndarray:
    unique_f, unique_idx = np.unique(f_in, return_index=True)
    h = h_in[unique_idx]
    mag = np.maximum(np.abs(h), 1e-15)
    phase = np.unwrap(np.angle(h))
    log_mag = np.interp(f_out, unique_f, np.log(mag), left=np.log(mag[0]), right=np.log(mag[-1]))
    out_phase = np.interp(f_out, unique_f, phase, left=phase[0], right=phase[-1])
    out = np.exp(log_mag + 1j * out_phase)
    out[f_out > unique_f[-1]] = 0.0
    return out


def time_response(frequency_hz: np.ndarray, transfer: np.ndarray, ui: float):
    positive = np.diff(np.unique(frequency_hz))
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError("At least two distinct frequency points are required.")
    measured_df = float(np.median(positive))
    # FFT record length is 1/df. Limit df so the record always contains the
    # requested pre/main/post cursor window plus alignment headroom.
    required_record_ui = max(
        32.0,
        float(2 * CURSOR_PRE_COUNT + CURSOR_POST_COUNT + 4),
    )
    df = min(measured_df, 1.0 / (required_record_ui * ui))
    fmax = float(np.max(frequency_hz))
    required_nfft = max(
        2048,
        int(math.ceil(32.0 / (ui * df))),
        int(math.ceil(2.0 * fmax / df)) + 2,
    )
    nfft = 1 << int(math.ceil(math.log2(required_nfft)))
    f_bins = np.arange(nfft // 2 + 1) * df
    spectrum = interpolate_transfer(frequency_hz, transfer, f_bins)
    # Gentle taper over the final 5% of measured bandwidth to reduce ringing.
    start = 0.95 * fmax
    mask = (f_bins >= start) & (f_bins <= fmax)
    spectrum[mask] *= 0.5 * (1.0 + np.cos(np.pi * (f_bins[mask] - start) / (fmax - start)))
    impulse = np.fft.irfft(spectrum, n=nfft)
    dt = 1.0 / (nfft * df)
    t = np.arange(nfft) * dt
    return t, impulse, f_bins, spectrum


def make_tx_pulse(t: np.ndarray, ui: float) -> np.ndarray:
    width = TX_PULSE_WIDTH_UI * ui
    tx = np.zeros_like(t)
    tx[t < width] = TX_PULSE_VOLTAGE
    # Optional linear edges; zero retains the ideal rectangular pulse.
    if TX_RISE_TIME > 0:
        rise = (t >= 0) & (t < TX_RISE_TIME)
        tx[rise] *= t[rise] / TX_RISE_TIME
    if TX_FALL_TIME > 0:
        fall = (t >= max(0.0, width - TX_FALL_TIME)) & (t < width)
        tx[fall] *= (width - t[fall]) / TX_FALL_TIME
    return tx


def extract_cursors(wave: np.ndarray, dt: float, ui: float):
    samples_per_ui = ui / dt
    if SAMPLING_PHASE_MODE.upper() == "AUTO":
        search_end = max(1, len(wave) - int((CURSOR_POST_COUNT + 1) * samples_per_ui))
        main_index = int(np.argmax(np.abs(wave[:search_end])))
    elif SAMPLING_PHASE_MODE.upper() == "MANUAL":
        peak = int(np.argmax(np.abs(wave)))
        base = peak - int(round((peak * dt / ui) % 1.0 * samples_per_ui))
        main_index = base + int(round(MANUAL_PHASE_UI * samples_per_ui))
    else:
        raise ValueError('SAMPLING_PHASE_MODE must be "AUTO" or "MANUAL".')
    cursor_k = np.arange(-CURSOR_PRE_COUNT, CURSOR_POST_COUNT + 1)
    positions = main_index + np.rint(cursor_k * samples_per_ui).astype(int)
    if positions[0] < 0 or positions[-1] >= len(wave):
        raise ValueError("The time record is too short for the requested cursor range.")
    return cursor_k, wave[positions], positions, main_index


def isi_metric(cursors: np.ndarray, main_pos: int) -> float:
    main = abs(cursors[main_pos])
    if main < 1e-15:
        return float("inf")
    return float(np.sqrt(np.sum(np.delete(cursors, main_pos) ** 2)) / main)


def ffe_solution(h: np.ndarray, h_k: np.ndarray, pre: int, post: int):
    tap_k = np.arange(-pre, post + 1)
    out_k = np.arange(h_k[0] - pre, h_k[-1] + post + 1)
    lookup = {int(k): float(v) for k, v in zip(h_k, h)}
    a = np.array([[lookup.get(int(k - tap), 0.0) for tap in tap_k] for k in out_k])
    main_pos = int(np.where(out_k == 0)[0][0])

    # Choose the FFE *ratio* that maximizes the main cursor relative to all
    # other cursor energy.  This is the minimum-ISI solution subject to a
    # unit main cursor; its final amplitude is applied separately below.
    main_row = a[main_pos]
    other_rows = np.delete(a, main_pos, axis=0)
    isi_matrix = other_rows.T @ other_rows
    regularization = 1e-8 * max(float(np.trace(isi_matrix)) / len(tap_k), 1e-15)
    direction = solve(
        isi_matrix + regularization * np.eye(len(tap_k)),
        main_row,
        assume_a="pos",
    )
    main_gain = float(main_row @ direction)
    if abs(main_gain) < 1e-15:
        raise RuntimeError("FFE optimization could not produce a non-zero main cursor.")
    direction /= main_gain

    # The ratio objective is scale-invariant. Use the available TX tap budget
    # after the ratio has been chosen, while guaranteeing sum(abs(taps)) <= limit.
    abs_sum = float(np.sum(np.abs(direction)))
    if abs_sum < 1e-15:
        raise RuntimeError("FFE optimization produced zero tap weights.")
    w = direction * (FFE_MAX_ABS_TAP_SUM / abs_sum)
    effective = a @ w
    return tap_k, w, out_k, effective, isi_metric(effective, main_pos)


def choose_ffe(h: np.ndarray, h_k: np.ndarray):
    candidates = []
    for pre in range(FFE_MAX_PRE_TAPS + 1):
        for post in range(FFE_MAX_POST_TAPS + 1):
            result = ffe_solution(h, h_k, pre, post)
            candidates.append((pre + post + 1, pre, post, result))
    best_metric = min(item[3][4] for item in candidates)
    tolerance = 1.0 + TAP_STOP_IMPROVEMENT_PERCENT / 100.0
    eligible = [item for item in candidates if item[3][4] <= best_metric * tolerance]
    return min(eligible, key=lambda item: (item[0], item[3][4]))[3]


def choose_dfe(h: np.ndarray, h_k: np.ndarray):
    lookup = {int(k): float(v) for k, v in zip(h_k, h)}
    main = lookup[0]
    if abs(main) < 1e-15:
        raise ValueError("Main cursor is zero; DFE coefficients cannot be calculated.")
    main_weight = TX_PULSE_VOLTAGE / main
    max_taps = min(DFE_MAX_TAPS, max(0, int(h_k[-1])))
    original_energy = sum(lookup.get(k, 0.0) ** 2 for k in range(1, max_taps + 1))
    taps = max_taps
    remaining = original_energy
    for count in range(1, max_taps + 1):
        previous = remaining
        remaining -= lookup.get(count, 0.0) ** 2
        improvement = 100.0 * (previous - remaining) / max(previous, 1e-30)
        if count > 1 and improvement < TAP_STOP_IMPROVEMENT_PERCENT:
            taps = count - 1
            break
    tap_k = np.arange(1, taps + 1)
    weights = np.array([-lookup.get(int(k), 0.0) / main for k in tap_k])
    corrected = h * main_weight
    for k in tap_k:
        corrected[np.where(h_k == k)[0][0]] = 0.0
    return main_weight, tap_k, weights, corrected


def apply_symbol_spaced_ffe(wave: np.ndarray, tap_k: np.ndarray, weights: np.ndarray,
                            samples_per_ui: float) -> np.ndarray:
    result = np.zeros_like(wave)
    for tap, weight in zip(tap_k, weights):
        shift = int(round(tap * samples_per_ui))
        if shift >= 0:
            result[shift:] += weight * wave[:len(wave) - shift]
        else:
            result[:shift] += weight * wave[-shift:]
    return result


def apply_dfe_one_ui_hold(wave: np.ndarray, cursor_k: np.ndarray,
                          cursor_positions: np.ndarray,
                          dfe_main_weight: float, dfe_k: np.ndarray,
                          dfe_weights: np.ndarray,
                          samples_per_ui: float) -> np.ndarray:
    """Display-model DFE using a one-UI zero-order hold around each post sample."""
    result = wave * dfe_main_weight
    half_ui = 0.5 * samples_per_ui
    for tap, weight in zip(dfe_k, dfe_weights):
        cursor_slot = int(np.where(cursor_k == tap)[0][0])
        center = int(cursor_positions[cursor_slot])
        start = max(0, int(math.ceil(center - half_ui)))
        stop = min(len(result), int(math.ceil(center + half_ui)))
        result[start:stop] += TX_PULSE_VOLTAGE * weight
    return result


def save_cursor_csv(path: Path, k: np.ndarray, before: np.ndarray, after: np.ndarray):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cursor_index", "before_V", "before_normalized", "after_EQ_V", "after_EQ_normalized"])
        main = int(np.where(k == 0)[0][0])
        for idx, a, b in zip(k, before, after):
            writer.writerow([int(idx), a, a / before[main], b, b / after[main] if after[main] else np.nan])


def render_cached_pulse_plot(path: Path) -> None:
    """Redraw the pulse figure from the last analysis without recomputing the channel."""
    if PULSE_PLOT_CACHE is None:
        raise RuntimeError("No pulse data is cached. Run Analysis first.")
    cache = PULSE_PLOT_CACHE
    time = cache["time"]
    dt = cache["dt"]
    ui = cache["ui"]
    main_index = cache["main_index"]
    mode = cache["mode"]
    ctle_enabled = cache["ctle_enabled"]
    x_time = (time - main_index * dt) / ui if TIME_AXIS_UNIT.upper() == "UI" else time - main_index * dt

    fig, ax_wave = plt.subplots(figsize=(11, 6), constrained_layout=True)
    if SHOW_TX_PULSE:
        ax_wave.plot(x_time, np.roll(cache["tx"], main_index), label="TX pulse (1 UI)", lw=1.5)
    if mode in {"FFE", "BOTH"} and SHOW_FFE_OUTPUT:
        ax_wave.plot(
            x_time, cache["tx_after_ffe"], label="FFE output",
            color="tab:cyan", lw=1.5, linestyle="--",
        )

    sample_positions = cache["positions"]

    def plot_rx(wave, label, color, lw=1.6, alpha=1.0):
        marker_options = {
            "marker": "o",
            "markevery": sample_positions,
            "markersize": 4.5,
            "markeredgewidth": 0.8,
        } if SHOW_SAMPLES else {}
        ax_wave.plot(x_time, wave, label=label, color=color, lw=lw, alpha=alpha, **marker_options)

    if SHOW_CHANNEL:
        plot_rx(cache["rx_channel"], "RX: Channel", "tab:orange", lw=1.45, alpha=0.75)
    if mode in {"FFE", "BOTH"} and SHOW_AFTER_FFE:
        plot_rx(cache["rx_after_ffe"], "RX: FFE + Channel", "tab:purple", lw=1.45)
    if ctle_enabled and SHOW_AFTER_CTLE:
        ctle_label = "RX: FFE + Channel + CTLE" if mode in {"FFE", "BOTH"} else "RX: Channel + CTLE"
        plot_rx(cache["rx_after_ctle"], ctle_label, "tab:green", lw=1.6)
    if mode in {"DFE", "BOTH"} and SHOW_AFTER_DFE:
        if mode == "BOTH":
            dfe_label = "RX: FFE + Channel + CTLE + DFE" if ctle_enabled else "RX: FFE + Channel + DFE"
        else:
            dfe_label = "RX: Channel + CTLE + DFE" if ctle_enabled else "RX: Channel + DFE"
        plot_rx(cache["rx_after_dfe"], dfe_label, "tab:red", lw=1.8)
    if TIME_AXIS_UNIT.upper() == "UI":
        ax_wave.set_xlim(-max(CURSOR_PRE_COUNT + 1, 2), CURSOR_POST_COUNT + 1)
    ax_wave.set_xlabel("Time [UI]" if TIME_AXIS_UNIT.upper() == "UI" else "Time [s]")
    ax_wave.set_ylabel("Differential voltage [V]" if cache["channel_mode"] == "DIFF" else "Voltage [V]")
    ax_wave.grid(True, alpha=0.3)
    handles, _ = ax_wave.get_legend_handles_labels()
    if handles:
        ax_wave.legend(loc="best")
    ax_wave.set_title("TX/RX pulse overlay")
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    temporary.replace(path)


def generate_sample(path: Path):
    """Generate a simple passive 50-ohm matched single-ended channel."""
    lines = [
        "! Synthetic practice channel: ~1.7 dB loss at 16 GHz, 1.8 ns delay",
        "! Touchstone order: S11 S21 S12 S22",
        "# GHZ S RI R 50",
    ]
    for f_ghz in np.linspace(0.0, 40.0, 401):
        f_hz = f_ghz * 1e9
        # Skin/dielectric-like monotonic loss plus a small reflection ripple.
        loss_db = 0.35 * math.sqrt(max(f_ghz, 0.0)) + 0.018 * f_ghz
        transmission = 10 ** (-loss_db / 20) * np.exp(-1j * 2 * np.pi * f_hz * 1.8e-9)
        reflection = 0.035 * np.exp(-1j * 2 * np.pi * f_hz * 0.22e-9)
        lines.append(
            f"{f_ghz:.6f} {reflection.real:.9e} {reflection.imag:.9e} "
            f"{transmission.real:.9e} {transmission.imag:.9e} "
            f"{transmission.real:.9e} {transmission.imag:.9e} "
            f"{reflection.real:.9e} {reflection.imag:.9e}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Generated sample Touchstone file: {path}")


def main():
    mode = MODE.upper()
    if mode not in {"NONE", "FFE", "DFE", "BOTH"}:
        raise ValueError('MODE must be "NONE", "FFE", "DFE", or "BOTH".')
    if SYMBOL_RATE_BAUD <= 0:
        raise ValueError("SYMBOL_RATE_BAUD must be positive.")
    if not 1 <= int(DISPLAY_DECIMAL_PLACES) <= 15:
        raise ValueError("DISPLAY_DECIMAL_PLACES must be between 1 and 15.")
    display_digits = int(DISPLAY_DECIMAL_PLACES)
    ui = 1.0 / SYMBOL_RATE_BAUD
    nyquist = SYMBOL_RATE_BAUD / 2.0
    ts = read_touchstone(S_PARAMETER_FILE)
    transfer, channel_reference_ohm, load_gamma = selected_transfer(ts)
    if ts.frequency_hz[-1] < nyquist:
        print("WARNING: S-parameter maximum frequency is below Nyquist.")
    if ts.frequency_hz[0] > 0:
        print("WARNING: DC is absent; the lowest-frequency value is extended to DC.")

    time, impulse, f_bins, base_spectrum = time_response(ts.frequency_hz, transfer, ui)
    dt = time[1] - time[0]
    # A short/low-delay channel can peak before the requested pre-cursor
    # window. Add a non-circular time-origin delay without changing its shape.
    alignment_samples = int(math.ceil((CURSOR_PRE_COUNT + 1) * ui / dt))
    if alignment_samples >= len(impulse):
        raise ValueError("The requested pre-cursor range exceeds the generated time record.")
    impulse = np.pad(impulse, (alignment_samples, 0))[:len(impulse)]
    tx = make_tx_pulse(time, ui)
    rx_channel = fftconvolve(impulse, tx, mode="full")[:len(time)]

    ctle = ctle_response(f_bins, CTLE_DB, nyquist)
    impulse_ctle = np.fft.irfft(base_spectrum * ctle, n=len(time))
    impulse_ctle = np.pad(impulse_ctle, (alignment_samples, 0))[:len(impulse_ctle)]
    rx_channel_ctle = fftconvolve(impulse_ctle, tx, mode="full")[:len(time)]
    k, cursors, positions, main_index = extract_cursors(rx_channel_ctle, dt, ui)
    main_pos = int(np.where(k == 0)[0][0])
    before_metric = isi_metric(cursors, main_pos)

    ffe_k = np.array([], dtype=int)
    ffe_w = np.array([], dtype=float)
    dfe_k = np.array([], dtype=int)
    dfe_w = np.array([], dtype=float)
    dfe_main_weight = 1.0
    tx_after_ffe = np.roll(tx, main_index)
    rx_after_ffe = rx_channel.copy()
    rx_after_ctle = rx_channel_ctle.copy()
    rx_after_dfe = rx_channel_ctle.copy()
    after_k, after_cursor = k.copy(), cursors.copy()
    after_positions = positions.copy()
    after_main_index = main_index

    if mode in {"FFE", "BOTH"}:
        ffe_k, ffe_w, _, _, _ = choose_ffe(cursors, k)
        tx_after_ffe = apply_symbol_spaced_ffe(
            np.roll(tx, main_index), ffe_k, ffe_w, ui / dt
        )
        # FFE is physically before the channel. For display, expose its output
        # both before and after CTLE; the final linear result is identical to
        # applying the same symbol-spaced filter after Channel + CTLE.
        rx_after_ffe = apply_symbol_spaced_ffe(rx_channel, ffe_k, ffe_w, ui / dt)
        rx_after_ctle = apply_symbol_spaced_ffe(rx_channel_ctle, ffe_k, ffe_w, ui / dt)
        after_k, after_cursor, after_positions, after_main_index = extract_cursors(rx_after_ctle, dt, ui)
        rx_after_dfe = rx_after_ctle.copy()
    if mode in {"DFE", "BOTH"}:
        dfe_input_cursor = after_cursor.copy()
        dfe_main_weight, dfe_k, dfe_w, after_cursor = choose_dfe(dfe_input_cursor, after_k)
        rx_after_dfe = apply_dfe_one_ui_hold(
            rx_after_ctle,
            after_k,
            after_positions,
            dfe_main_weight,
            dfe_k,
            dfe_w,
            ui / dt,
        )

    after_main_pos = int(np.where(after_k == 0)[0][0])
    after_metric = isi_metric(after_cursor, after_main_pos)

    out_dir = Path(OUTPUT_DIRECTORY)
    if not out_dir.is_dir():
        raise FileNotFoundError(
            "Result output directory does not exist. Create it first or change "
            f"OUTPUT_DIRECTORY in USER CONTROL: {out_dir.resolve()}"
        )
    print(f"\nChannel mode : {CHANNEL_MODE.upper()} ({ts.s.shape[1]} ports, R={ts.reference_ohm:g} ohm)")
    load_ohm = channel_reference_ohm if RX_TERMINATION_OHM is None else float(RX_TERMINATION_OHM)
    load_mode = "matched" if RX_TERMINATION_OHM is None else "custom"
    print(f"RX termination: {load_ohm:g} ohm ({load_mode}, reference "
          f"{channel_reference_ohm:g} ohm, GammaL={load_gamma:+.6f})")
    print(f"Symbol rate  : {SYMBOL_RATE_BAUD:.6E} Baud")
    print(f"UI           : {ui:.6E} s")
    print(f"Nyquist      : {nyquist:.6E} Hz")
    print(f"CTLE         : {CTLE_DB:.3f} dB at Nyquist ({'OFF' if CTLE_DB == 0 else 'ON'})")
    ctle_points = ctle_breakpoints(CTLE_DB, nyquist)
    if ctle_points is not None:
        print(f"CTLE Z1/P1/P2: {ctle_points[0]:.6E} / {ctle_points[1]:.6E} / "
              f"{ctle_points[2]:.6E} Hz")
    print(f"EQ mode      : {'FFE + DFE' if mode == 'BOTH' else mode.title() if mode == 'NONE' else mode}")

    if PRINT_ABSOLUTE_CURSOR or PRINT_NORMALIZED_CURSOR:
        print("\nCursor values")
        print(" index       absolute [V]       normalized")
        for idx, value in zip(k, cursors):
            print(f" {idx:+4d}       {value:.{display_digits}f}    "
                  f"{value/cursors[main_pos]:.{display_digits}f}")
    ffe_display = {int(i): float(w) for i, w in zip(ffe_k, ffe_w)}
    dfe_display = {0: float(dfe_main_weight)} if mode in {"DFE", "BOTH"} else {}
    dfe_display.update({int(i): float(w) for i, w in zip(dfe_k, dfe_w)})
    if PRINT_EQ_TAPS:
        display_indices = sorted(set(ffe_display) | set(dfe_display))
        print("\nRecommended equalizer taps (reference)")
        print(" tap [UI]        FFE weight        DFE weight")
        print(" --------    ---------------   ---------------")
        for idx in display_indices:
            ffe_text = f"{ffe_display[idx]:.{display_digits}f}" if idx in ffe_display else "       -       "
            dfe_text = f"{dfe_display[idx]:.{display_digits}f}" if idx in dfe_display else "       -       "
            label = " 0 (Main)" if idx == 0 else f"{idx:+4d}      "
            print(f" {label}    {ffe_text:>15}   {dfe_text:>15}")
        if ffe_k.size:
            print("\nFFE TX constraint")
            print(f"  Sum(abs(taps)) : {np.sum(np.abs(ffe_w)):.{display_digits}f} "
                  f"(limit {FFE_MAX_ABS_TAP_SUM:.{display_digits}f})")
        if mode in {"DFE", "BOTH"}:
            print(f"  DFE tap 0 (Main gain) = TX pulse / EQ-input Main = "
                  f"{dfe_main_weight:.{display_digits}f}; target Main = "
                  f"{TX_PULSE_VOLTAGE:.{display_digits}f} V.")
    if PRINT_RESIDUAL_ISI:
        print(f"\nRMS residual ISI/main: {before_metric:.6f} -> {after_metric:.6f}")

    if SAVE_RESULT_CSV:
        save_cursor_csv(out_dir / "cursor_results.csv", k, cursors, after_cursor)
        with (out_dir / "equalizer_taps.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tap_UI", "tap_name", "FFE_weight", "DFE_weight"])
            for idx in sorted(set(ffe_display) | set(dfe_display)):
                writer.writerow([
                    idx,
                    "Main" if idx == 0 else (f"Pre-{abs(idx)}" if idx < 0 else f"Post-{idx}"),
                    ffe_display.get(idx, ""),
                    dfe_display.get(idx, ""),
                ])

    global PULSE_PLOT_CACHE
    PULSE_PLOT_CACHE = {
        "time": time,
        "dt": dt,
        "ui": ui,
        "main_index": main_index,
        "tx": tx,
        "tx_after_ffe": tx_after_ffe,
        "rx_channel": rx_channel,
        "rx_after_ffe": rx_after_ffe,
        "rx_after_ctle": rx_after_ctle,
        "rx_after_dfe": rx_after_dfe,
        "mode": mode,
        "ctle_enabled": abs(CTLE_DB) >= 1e-12,
        "k": k,
        "cursors": cursors,
        "positions": positions,
        "after_positions": after_positions,
        "after_cursor": after_cursor,
        "channel_mode": CHANNEL_MODE.upper(),
    }
    if SAVE_PLOT_PNG:
        render_cached_pulse_plot(out_dir / "pulse_and_cursor.png")

    # Separate channel/CTLE/composite Bode plot.
    bode_mask = ts.frequency_hz > 0.0
    bode_f = ts.frequency_hz[bode_mask]
    bode_channel = transfer[bode_mask]
    bode_ctle = ctle_response(bode_f, CTLE_DB, nyquist)
    bode_composite = bode_channel * bode_ctle
    magnitude_floor = 1e-15

    fig_bode, ax_mag = plt.subplots(figsize=(11, 6), constrained_layout=True)
    bode_items = [(bode_channel, "Loaded channel transfer")]
    if abs(CTLE_DB) >= 1e-12:
        bode_items.extend([
            (bode_ctle, f"CTLE ({CTLE_DB:g} dB target)"),
            (bode_composite, "Loaded channel + CTLE compensation"),
        ])
    for response, label in bode_items:
        ax_mag.semilogx(
            bode_f,
            20.0 * np.log10(np.maximum(np.abs(response), magnitude_floor)),
            label=label,
        )
    ax_mag.axvline(nyquist, color="black", linestyle="--", alpha=0.55, label="Nyquist")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.axhline(0.0, color="gray", linewidth=0.9)
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title(
        "Channel and CTLE magnitude response"
        if abs(CTLE_DB) >= 1e-12 else "Channel magnitude response"
    )
    ax_mag.legend(loc="best")
    if SAVE_PLOT_PNG:
        fig_bode.savefig(out_dir / "channel_ctle_bode.png", dpi=180)
    plt.close(fig_bode)
    print(f"\nResults saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-sample", metavar="PATH", help="generate a synthetic .s2p practice file")
    args = parser.parse_args()
    if args.make_sample:
        generate_sample(Path(args.make_sample))
    else:
        main()
