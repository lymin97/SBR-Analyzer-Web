"""Minimal online demo for the SBR Analyzer."""

from __future__ import annotations

import contextlib
import csv
import io
import re
import tempfile
import threading
import zipfile
from pathlib import Path

import streamlit as st

import sbr_analyzer as analyzer


__author__ = "Young-Min Lee"
__version__ = "1.0.0"

APP_DIRECTORY = Path(__file__).resolve().parent
SAMPLE_FILE = APP_DIRECTORY / "sample_diff_channel_200Mbaud_minus6dB_at_nyquist.s4p"


@st.cache_resource
def shared_analysis_lock() -> threading.Lock:
    """Serialize access to the analyzer module's configuration globals."""
    return threading.Lock()


ANALYSIS_LOCK = shared_analysis_lock()


def csv_rows(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def selected_results_zip(results: dict[str, object], selected: dict[str, bool]) -> bytes:
    """Package the selected result files into one in-memory ZIP archive."""
    filenames = {
        "pulse": "sbr_analysis_pulse.png",
        "magnitude": "sbr_analysis_magnitude.png",
        "cursors": "sbr_analysis_cursors.csv",
        "taps": "sbr_analysis_eq_taps.csv",
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for key, enabled in selected.items():
            if enabled:
                bundle.writestr(filenames[key], results[key])
    return archive.getvalue()


def configure_analyzer(settings: dict[str, object], input_path: Path, output_path: Path) -> None:
    analyzer.S_PARAMETER_FILE = str(input_path)
    analyzer.OUTPUT_DIRECTORY = str(output_path)
    analyzer.CHANNEL_MODE = str(settings["channel_mode"])
    analyzer.TX_PORT = int(settings["tx_port"])
    analyzer.RX_PORT = int(settings["rx_port"])
    analyzer.TX_POS_PORT = int(settings["tx_pos_port"])
    analyzer.TX_NEG_PORT = int(settings["tx_neg_port"])
    analyzer.RX_POS_PORT = int(settings["rx_pos_port"])
    analyzer.RX_NEG_PORT = int(settings["rx_neg_port"])
    analyzer.RX_TERMINATION_OHM = settings["rx_termination"]
    analyzer.SYMBOL_RATE_BAUD = float(settings["symbol_rate"])
    analyzer.TX_PULSE_VOLTAGE = float(settings["tx_voltage"])
    analyzer.TX_PULSE_WIDTH_UI = float(settings["pulse_width"])
    analyzer.TX_RISE_TIME = float(settings["rise_time"])
    analyzer.TX_FALL_TIME = float(settings["fall_time"])
    analyzer.CTLE_DB = float(settings["ctle_db"])
    analyzer.MODE = str(settings["eq_mode"])
    analyzer.FFE_MAX_PRE_TAPS = int(settings["ffe_pre"])
    analyzer.FFE_MAX_POST_TAPS = int(settings["ffe_post"])
    analyzer.DFE_MAX_TAPS = int(settings["dfe_taps"])
    analyzer.FFE_MAX_ABS_TAP_SUM = float(settings["ffe_limit"])
    analyzer.TAP_STOP_IMPROVEMENT_PERCENT = float(settings["stop_improvement"])
    analyzer.CURSOR_PRE_COUNT = int(settings["cursor_pre"])
    analyzer.CURSOR_POST_COUNT = int(settings["cursor_post"])
    analyzer.SAMPLING_PHASE_MODE = "AUTO"
    analyzer.TIME_AXIS_UNIT = str(settings["time_unit"])
    analyzer.DISPLAY_DECIMAL_PLACES = int(settings["decimal_places"])
    analyzer.SAVE_RESULT_CSV = True
    analyzer.SAVE_PLOT_PNG = True
    analyzer.SHOW_TX_PULSE = bool(settings["show_tx_pulse"])
    analyzer.SHOW_FFE_OUTPUT = bool(settings["show_ffe_output"])
    analyzer.SHOW_CHANNEL = bool(settings["show_channel"])
    analyzer.SHOW_AFTER_FFE = bool(settings["show_after_ffe"])
    analyzer.SHOW_AFTER_CTLE = bool(settings["show_after_ctle"])
    analyzer.SHOW_AFTER_DFE = bool(settings["show_after_dfe"])
    analyzer.SHOW_SAMPLES = bool(settings["show_samples"])


def run_analysis(uploaded_file, settings: dict[str, object]) -> dict[str, object]:
    with ANALYSIS_LOCK, tempfile.TemporaryDirectory(prefix="sbr_web_") as temporary:
        work = Path(temporary)
        if uploaded_file is None:
            input_path = SAMPLE_FILE
            input_name = SAMPLE_FILE.name
        else:
            input_name = Path(uploaded_file.name).name
            if not re.fullmatch(r"(?i).+\.s\d+p", input_name):
                raise ValueError("Upload a Touchstone file such as .s2p or .s4p.")
            input_path = work / input_name
            input_path.write_bytes(uploaded_file.getvalue())

        configure_analyzer(settings, input_path, work)
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            analyzer.main()

        filenames = {
            "pulse": "pulse_and_cursor.png",
            "magnitude": "channel_ctle_bode.png",
            "cursors": "cursor_results.csv",
            "taps": "equalizer_taps.csv",
        }
        payload: dict[str, object] = {"log": log.getvalue(), "input_name": input_name}
        for key, filename in filenames.items():
            path = work / filename
            if not path.is_file():
                raise RuntimeError(f"Expected result was not generated: {filename}")
            payload[key] = path.read_bytes()
        return payload


st.set_page_config(page_title="SBR Analyzer", page_icon="📈", layout="wide")
st.title("Single-Bit Response Analyzer")
st.caption("Designed and developed by Young-Min Lee · Online demo")
st.info(
    "Uploaded Touchstone files are processed temporarily by the hosted server. "
    "Do not upload confidential channel data to a public deployment."
)

uploaded = st.file_uploader(
    "Touchstone S-parameter file",
    help=("Click Browse files or drag and drop any .sNp Touchstone file. "
          "If omitted, the included 200 MBaud differential sample is used."),
)
if uploaded is None:
    st.caption(f"Using example: {SAMPLE_FILE.name}")

st.subheader("Channel / Ports")
channel_mode_label = st.selectbox("Channel mode", ["Single-ended", "Differential"], index=1)
channel_mode = "SE" if channel_mode_label == "Single-ended" else "DIFF"

with st.container(border=True):
    port_columns = st.columns(4)
    if channel_mode == "SE":
        tx_port = port_columns[0].number_input("TX port", min_value=1, value=1, step=1)
        rx_port = port_columns[1].number_input("RX port", min_value=1, value=2, step=1)
        tx_pos_port, tx_neg_port, rx_pos_port, rx_neg_port = 1, 3, 2, 4
    else:
        tx_pos_port = port_columns[0].number_input("TX+ port", min_value=1, value=1, step=1)
        tx_neg_port = port_columns[1].number_input("TX− port", min_value=1, value=3, step=1)
        rx_pos_port = port_columns[2].number_input("RX+ port", min_value=1, value=2, step=1)
        rx_neg_port = port_columns[3].number_input("RX− port", min_value=1, value=4, step=1)
        tx_port, rx_port = 1, 2

    matched = st.checkbox("Use matched RX termination", value=True)
    rx_termination_value = st.number_input(
        "RX termination [ohm]", min_value=0.001, value=100.0 if channel_mode == "DIFF" else 50.0,
        disabled=matched,
    )
    rx_termination = None if matched else float(rx_termination_value)

    st.subheader("Signal / Cursor")
    signal_columns = st.columns(3)
    symbol_rate = signal_columns[0].number_input("Symbol rate [Baud]", min_value=1.0, value=200.0e6, format="%.6e")
    tx_voltage = signal_columns[1].number_input("TX pulse [V]", min_value=0.001, value=1.0)
    pulse_width = signal_columns[2].number_input("Pulse width [UI]", min_value=0.001, value=1.0)
    edge_columns = st.columns(2)
    rise_time = edge_columns[0].number_input("TX rise time [s]", min_value=0.0, value=0.0, format="%.6e")
    fall_time = edge_columns[1].number_input("TX fall time [s]", min_value=0.0, value=0.0, format="%.6e")
    cursor_columns = st.columns(4)
    cursor_pre = cursor_columns[0].number_input("Pre cursors", min_value=0, value=5, step=1)
    cursor_post = cursor_columns[1].number_input("Post cursors", min_value=0, value=10, step=1)
    time_unit = cursor_columns[2].selectbox("Time unit", ["UI", "s"])
    decimal_places = cursor_columns[3].number_input("Display decimals", min_value=1, max_value=15, value=6, step=1)

    st.subheader("Equalizer")
    eq_columns = st.columns(3)
    ctle_db = eq_columns[0].number_input("CTLE boost [dB]", min_value=0.0, value=0.0)
    eq_mode_label = eq_columns[1].selectbox("EQ mode", ["None", "FFE", "DFE", "FFE + DFE"], index=2)
    eq_mode = {"None": "NONE", "FFE": "FFE", "DFE": "DFE", "FFE + DFE": "BOTH"}[eq_mode_label]
    stop_improvement = eq_columns[2].number_input("Tap stop improvement [%]", min_value=0.0, value=1.0)
    tap_columns = st.columns(4)
    ffe_pre = tap_columns[0].number_input("FFE max pre taps", min_value=0, value=3, step=1)
    ffe_post = tap_columns[1].number_input("FFE max post taps", min_value=0, value=5, step=1)
    ffe_limit = tap_columns[2].number_input("FFE sum(abs) limit", min_value=0.001, value=1.0)
    dfe_taps = tap_columns[3].number_input("DFE max taps", min_value=0, value=8, step=1)

    st.subheader("Graph Options")
    graph_columns = st.columns(6)
    show_tx_pulse = graph_columns[0].checkbox("TX pulse", value=True)
    show_ffe_output = graph_columns[1].checkbox("FFE output", value=True)
    show_channel = graph_columns[2].checkbox("Channel", value=True)
    show_after_ffe = graph_columns[3].checkbox("FFE + Channel", value=True)
    show_after_ctle = graph_columns[4].checkbox("FFE + Channel + CTLE", value=True)
    show_after_dfe = graph_columns[5].checkbox("FFE + Channel + CTLE + DFE", value=True)
    show_samples = st.checkbox("Sample markers (1 UI interval)", value=True)

    submitted = st.button("Run Analysis", type="primary", use_container_width=True)

settings = {
    "channel_mode": channel_mode,
    "tx_port": tx_port,
    "rx_port": rx_port,
    "tx_pos_port": tx_pos_port,
    "tx_neg_port": tx_neg_port,
    "rx_pos_port": rx_pos_port,
    "rx_neg_port": rx_neg_port,
    "rx_termination": rx_termination,
    "symbol_rate": symbol_rate,
    "tx_voltage": tx_voltage,
    "pulse_width": pulse_width,
    "rise_time": rise_time,
    "fall_time": fall_time,
    "cursor_pre": cursor_pre,
    "cursor_post": cursor_post,
    "time_unit": time_unit,
    "decimal_places": decimal_places,
    "ctle_db": ctle_db,
    "eq_mode": eq_mode,
    "stop_improvement": stop_improvement,
    "ffe_pre": ffe_pre,
    "ffe_post": ffe_post,
    "ffe_limit": ffe_limit,
    "dfe_taps": dfe_taps,
    "show_tx_pulse": show_tx_pulse,
    "show_ffe_output": show_ffe_output,
    "show_channel": show_channel,
    "show_after_ffe": show_after_ffe,
    "show_after_ctle": show_after_ctle,
    "show_after_dfe": show_after_dfe,
    "show_samples": show_samples,
}

if submitted:
    try:
        with st.spinner("Analyzing channel..."):
            st.session_state["analysis_results"] = run_analysis(uploaded, settings)
        st.success("Analysis completed.")
    except Exception as exc:
        st.session_state.pop("analysis_results", None)
        st.error(f"{type(exc).__name__}: {exc}")

results = st.session_state.get("analysis_results")
if results:
    st.divider()
    st.subheader(f"Results · {results['input_name']}")
    pulse_tab, magnitude_tab, cursor_tab, tap_tab = st.tabs(
        ["Pulse Response", "Magnitude", "Cursor Results", "EQ Taps"]
    )
    with pulse_tab:
        st.image(results["pulse"], use_container_width=True)
    with magnitude_tab:
        st.image(results["magnitude"], use_container_width=True)
    with cursor_tab:
        st.dataframe(csv_rows(results["cursors"]), use_container_width=True, hide_index=True)
    with tap_tab:
        st.dataframe(csv_rows(results["taps"]), use_container_width=True, hide_index=True)

    st.subheader("Save Options")
    save_columns = st.columns(4)
    save_selection = {
        "pulse": save_columns[0].checkbox("Pulse PNG", value=True),
        "magnitude": save_columns[1].checkbox("Magnitude PNG", value=True),
        "cursors": save_columns[2].checkbox("Cursor CSV", value=True),
        "taps": save_columns[3].checkbox("EQ Taps CSV", value=True),
    }
    if any(save_selection.values()):
        download_zip = selected_results_zip(results, save_selection)
        st.download_button(
            "Download Selected Results",
            data=download_zip,
            file_name="sbr_analysis.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
    else:
        st.warning("Select at least one result to download.")

    with st.expander("Run Log"):
        st.code(results["log"] or "Analysis completed.", language="text")

st.caption("SBR Analyzer · Version 1.0 · © 2026 Young-Min Lee")
