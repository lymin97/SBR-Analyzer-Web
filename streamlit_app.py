"""Minimal online demo for the SBR Analyzer."""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import re
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

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


def setting_row(label: str, description: str, widget: str, **kwargs):
    """Render one setting per row with a short muted description."""
    label_column, input_column, help_column = st.columns([1.15, 1.35, 2.5], vertical_alignment="center")
    label_column.markdown(f"**{label}**")
    value = getattr(input_column, widget)(
        label,
        label_visibility="collapsed",
        **kwargs,
    )
    help_column.caption(description)
    return value


def csv_rows(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def fixed_decimal_csv(data: bytes, decimal_places: int) -> bytes:
    """Format decimal CSV fields without scientific notation."""
    source = io.StringIO(data.decode("utf-8-sig"))
    reader = csv.reader(source)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    threshold = 0.5 * 10.0 ** (-decimal_places)
    for row in reader:
        formatted = []
        for value in row:
            stripped = value.strip()
            if "." not in stripped and "e" not in stripped.lower():
                formatted.append(value)
                continue
            try:
                number = float(stripped)
            except ValueError:
                formatted.append(value)
                continue
            if abs(number) < threshold:
                number = 0.0
            formatted.append(f"{number:.{decimal_places}f}")
        writer.writerow(formatted)
    return output.getvalue().encode("utf-8-sig")


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
    analyzer.SHOW_TX_PULSE = True
    analyzer.SHOW_FFE_OUTPUT = True
    analyzer.SHOW_CHANNEL = True
    analyzer.SHOW_AFTER_FFE = True
    analyzer.SHOW_AFTER_CTLE = True
    analyzer.SHOW_AFTER_DFE = True
    analyzer.SHOW_SAMPLES = True


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
        payload: dict[str, object] = {
            "log": log.getvalue(),
            "input_name": input_name,
            "plot_cache": copy.deepcopy(analyzer.PULSE_PLOT_CACHE),
            "plot_context": {
                "eq_mode": str(settings["eq_mode"]),
                "ctle_db": float(settings["ctle_db"]),
                "cursor_pre": int(settings["cursor_pre"]),
                "cursor_post": int(settings["cursor_post"]),
                "time_unit": str(settings["time_unit"]),
                "decimal_places": int(settings["decimal_places"]),
            },
        }
        for key, filename in filenames.items():
            path = work / filename
            if not path.is_file():
                raise RuntimeError(f"Expected result was not generated: {filename}")
            payload[key] = path.read_bytes()
        return payload


def render_session_pulse(results: dict[str, object], options: dict[str, bool]) -> bytes:
    """Redraw one session's cached pulse data without repeating channel analysis."""
    with ANALYSIS_LOCK, tempfile.TemporaryDirectory(prefix="sbr_plot_") as temporary:
        path = Path(temporary) / "pulse_and_cursor.png"
        saved = {
            "cache": analyzer.PULSE_PLOT_CACHE,
            "pre": analyzer.CURSOR_PRE_COUNT,
            "post": analyzer.CURSOR_POST_COUNT,
            "unit": analyzer.TIME_AXIS_UNIT,
            "tx": analyzer.SHOW_TX_PULSE,
            "ffe_output": analyzer.SHOW_FFE_OUTPUT,
            "channel": analyzer.SHOW_CHANNEL,
            "after_ffe": analyzer.SHOW_AFTER_FFE,
            "after_ctle": analyzer.SHOW_AFTER_CTLE,
            "after_dfe": analyzer.SHOW_AFTER_DFE,
            "samples": analyzer.SHOW_SAMPLES,
        }
        context = results["plot_context"]
        try:
            analyzer.PULSE_PLOT_CACHE = copy.deepcopy(results["plot_cache"])
            analyzer.CURSOR_PRE_COUNT = int(context["cursor_pre"])
            analyzer.CURSOR_POST_COUNT = int(context["cursor_post"])
            analyzer.TIME_AXIS_UNIT = str(context["time_unit"])
            analyzer.SHOW_TX_PULSE = options["tx"]
            analyzer.SHOW_FFE_OUTPUT = options["ffe_output"]
            analyzer.SHOW_CHANNEL = options["channel"]
            analyzer.SHOW_AFTER_FFE = options["after_ffe"]
            analyzer.SHOW_AFTER_CTLE = options["after_ctle"]
            analyzer.SHOW_AFTER_DFE = options["after_dfe"]
            analyzer.SHOW_SAMPLES = options["samples"]
            analyzer.render_cached_pulse_plot(path)
            return path.read_bytes()
        finally:
            analyzer.PULSE_PLOT_CACHE = saved["cache"]
            analyzer.CURSOR_PRE_COUNT = saved["pre"]
            analyzer.CURSOR_POST_COUNT = saved["post"]
            analyzer.TIME_AXIS_UNIT = saved["unit"]
            analyzer.SHOW_TX_PULSE = saved["tx"]
            analyzer.SHOW_FFE_OUTPUT = saved["ffe_output"]
            analyzer.SHOW_CHANNEL = saved["channel"]
            analyzer.SHOW_AFTER_FFE = saved["after_ffe"]
            analyzer.SHOW_AFTER_CTLE = saved["after_ctle"]
            analyzer.SHOW_AFTER_DFE = saved["after_dfe"]
            analyzer.SHOW_SAMPLES = saved["samples"]


def interactive_pulse_figure(
    results: dict[str, object], options: dict[str, bool], mobile_layout: bool = False
) -> go.Figure:
    """Build a browser-rendered pulse plot for fast trace toggling."""
    cache = results["plot_cache"]
    context = results["plot_context"]
    time = np.asarray(cache["time"])
    main_index = int(cache["main_index"])
    if str(context["time_unit"]).upper() == "UI":
        x_values = (time - main_index * float(cache["dt"])) / float(cache["ui"])
        x_title = "Time [UI]"
        x_range = [-int(context["cursor_pre"]), int(context["cursor_post"])]
    else:
        x_values = time - main_index * float(cache["dt"])
        x_title = "Time [s]"
        x_range = [
            -int(context["cursor_pre"]) * float(cache["ui"]),
            int(context["cursor_post"]) * float(cache["ui"]),
        ]
    if x_range[0] == x_range[1]:
        half_ui = 0.5 if x_title == "Time [UI]" else 0.5 * float(cache["ui"])
        x_range = [x_range[0] - half_ui, x_range[1] + half_ui]
    visible_indices = np.flatnonzero((x_values >= x_range[0]) & (x_values <= x_range[1]))
    plot_x = x_values[visible_indices]

    figure = go.Figure()
    sample_positions = np.asarray(cache["positions"], dtype=int)

    def add_trace(key: str, y_values, label: str, color: str, dash: str = "solid") -> None:
        if not options[key]:
            return
        y_array = np.asarray(y_values)
        figure.add_trace(go.Scatter(
            x=plot_x, y=y_array[visible_indices], mode="lines", name=label,
            line={"color": color, "width": 2, "dash": dash},
        ))
        if options["samples"] and key not in {"tx", "ffe_output"}:
            valid = sample_positions[(sample_positions >= 0) & (sample_positions < y_array.size)]
            valid = valid[(x_values[valid] >= x_range[0]) & (x_values[valid] <= x_range[1])]
            figure.add_trace(go.Scatter(
                x=x_values[valid], y=y_array[valid], mode="markers",
                name=f"{label} samples", showlegend=False,
                marker={"color": color, "size": 7, "line": {"width": 1, "color": "white"}},
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y}}<extra></extra>",
            ))

    add_trace("tx", np.roll(cache["tx"], main_index), "TX pulse (1 UI)", "#1f77b4")
    add_trace("ffe_output", cache["tx_after_ffe"], "FFE output", "#17becf", "dash")
    add_trace("channel", cache["rx_channel"], "RX: Channel", "#ff7f0e")
    add_trace("after_ffe", cache["rx_after_ffe"], "RX: FFE + Channel", "#9467bd")

    ffe_enabled = str(context["eq_mode"]) in {"FFE", "BOTH"}
    ctle_label = "RX: FFE + Channel + CTLE" if ffe_enabled else "RX: Channel + CTLE"
    add_trace("after_ctle", cache["rx_after_ctle"], ctle_label, "#2ca02c")
    dfe_prefix = "FFE + Channel" if ffe_enabled else "Channel"
    if float(context["ctle_db"]) > 0.0:
        dfe_prefix += " + CTLE"
    add_trace("after_dfe", cache["rx_after_dfe"], f"RX: {dfe_prefix} + DFE", "#d62728")

    legend_layout = (
        {
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
        }
        if mobile_layout
        else {"orientation": "v"}
    )
    figure.update_layout(
        title="" if mobile_layout else "TX/RX pulse overlay",
        xaxis_title=x_title,
        yaxis_title="Differential voltage [V]" if cache["channel_mode"] == "DIFF" else "Voltage [V]",
        hovermode="x unified",
        legend=legend_layout,
        margin={"l": 55, "r": 25, "t": 150 if mobile_layout else 55, "b": 50},
        height=400 if mobile_layout else 520,
    )
    figure.update_xaxes(range=x_range, autorange=False, showgrid=True)
    figure.update_yaxes(showgrid=True)
    return figure


st.set_page_config(page_title="SBR Analyzer", page_icon="📈", layout="wide")

stored_browser_settings = streamlit_js_eval(
    js_expressions="window.sessionStorage.getItem('sbrAnalyzerSettings') || '{}'",
    want_output=True,
    key="load_browser_settings",
)
if not st.session_state.get("browser_settings_loaded", False):
    if stored_browser_settings is None:
        st.stop()
    try:
        restored_settings = json.loads(stored_browser_settings)
    except (TypeError, json.JSONDecodeError):
        restored_settings = {}
    if isinstance(restored_settings, dict):
        for restored_key, restored_value in restored_settings.items():
            st.session_state[restored_key] = restored_value
    st.session_state["browser_settings_loaded"] = True
    st.rerun()

st.markdown(
    """
    <style>
    .st-key-mobile_pulse_plot {
        display: none;
    }
    div[data-testid="stMainBlockContainer"] {
        max-width: 80vw;
        margin-left: auto;
        margin-right: auto;
    }
    @media (max-width: 768px) {
        .st-key-desktop_pulse_plot {
            display: none;
        }
        .st-key-mobile_pulse_plot {
            display: block;
            width: 100%;
        }
        div[data-testid="stMainBlockContainer"] {
            max-width: 96vw;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
    div[data-testid="stNumberInput"] button {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
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

if uploaded is not None:
    input_signature = (uploaded.name, uploaded.size)
if uploaded is not None and st.session_state.get("detected_input_signature") != input_signature:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".s2p":
        st.session_state["channel_mode"] = "Single-ended"
    elif suffix == ".s4p":
        st.session_state["channel_mode"] = "Differential"
    st.session_state["detected_input_signature"] = input_signature

st.subheader("Channel / Ports")
channel_mode_label = setting_row(
    "Channel mode", "Select a single-ended or differential channel.",
    "selectbox", options=["Single-ended", "Differential"], index=1, key="channel_mode",
)
channel_mode = "SE" if channel_mode_label == "Single-ended" else "DIFF"

with st.container(border=True):
    if channel_mode == "SE":
        tx_port = setting_row(
            "TX port", "Single-ended transmitter port number.",
            "number_input", min_value=1, value=1, step=1, key="tx_port",
        )
        rx_port = setting_row(
            "RX port", "Single-ended receiver port number.",
            "number_input", min_value=1, value=2, step=1, key="rx_port",
        )
        tx_pos_port, tx_neg_port, rx_pos_port, rx_neg_port = 1, 3, 2, 4
    else:
        tx_pos_port = setting_row(
            "TX+ port", "Differential transmitter positive port.",
            "number_input", min_value=1, value=1, step=1, key="tx_pos_port",
        )
        tx_neg_port = setting_row(
            "TX− port", "Differential transmitter negative port.",
            "number_input", min_value=1, value=3, step=1, key="tx_neg_port",
        )
        rx_pos_port = setting_row(
            "RX+ port", "Differential receiver positive port.",
            "number_input", min_value=1, value=2, step=1, key="rx_pos_port",
        )
        rx_neg_port = setting_row(
            "RX− port", "Differential receiver negative port.",
            "number_input", min_value=1, value=4, step=1, key="rx_neg_port",
        )
        tx_port, rx_port = 1, 2

    matched = setting_row(
        "Matched RX termination", "Use the Touchstone reference impedance.",
        "checkbox", value=True, key="matched_rx",
    )
    rx_termination_value = setting_row(
        "RX termination [ohm]", "Optional implemented receiver termination impedance.",
        "number_input", min_value=0.001, value=100.0 if channel_mode == "DIFF" else 50.0,
        disabled=matched, key=f"rx_termination_{channel_mode.lower()}",
    )
    rx_termination = None if matched else float(rx_termination_value)

    st.subheader("Signal / Cursor")
    symbol_rate = setting_row("Symbol rate [Baud]", "Data symbol rate used to define 1 UI.", "number_input", min_value=1.0, value=200.0e6, format="%.6e", key="symbol_rate")
    tx_voltage = setting_row("TX pulse [V]", "Transmitted single-bit pulse amplitude.", "number_input", min_value=0.001, value=1.0, key="tx_voltage")
    pulse_width = setting_row("Pulse width [UI]", "Duration of the transmitted pulse in UI.", "number_input", min_value=0.001, value=1.0, key="pulse_width")
    rise_time = setting_row("TX rise time [s]", "Optional transmitter rise time; zero is ideal.", "number_input", min_value=0.0, value=0.0, format="%.6e", key="rise_time")
    fall_time = setting_row("TX fall time [s]", "Optional transmitter fall time; zero is ideal.", "number_input", min_value=0.0, value=0.0, format="%.6e", key="fall_time")
    cursor_pre = setting_row("Pre cursors", "Number of pre-cursor UI samples to report.", "number_input", min_value=0, value=5, step=1, key="cursor_pre")
    cursor_post = setting_row("Post cursors", "Number of post-cursor UI samples to report.", "number_input", min_value=0, value=10, step=1, key="cursor_post")
    time_unit = setting_row("Time unit", "Horizontal unit used in the pulse plot.", "selectbox", options=["UI", "s"], key="time_unit")
    decimal_places = setting_row("Display decimals", "Decimal places used in result tables.", "number_input", min_value=1, max_value=15, value=6, step=1, key="decimal_places")

    st.subheader("Equalizer")
    ctle_db = setting_row("CTLE boost [dB]", "High-frequency boost; zero disables CTLE.", "number_input", min_value=0.0, value=0.0, key="ctle_db")
    eq_mode_label = setting_row("EQ mode", "Select the equalizer stages to recommend and apply.", "selectbox", options=["None", "FFE", "DFE", "FFE + DFE"], index=2, key="eq_mode")
    eq_mode = {"None": "NONE", "FFE": "FFE", "DFE": "DFE", "FFE + DFE": "BOTH"}[eq_mode_label]
    if eq_mode in {"FFE", "BOTH"}:
        ffe_pre = setting_row("FFE max pre taps", "Maximum number of FFE pre-cursor taps.", "number_input", min_value=0, value=3, step=1, key="ffe_pre")
        ffe_post = setting_row("FFE max post taps", "Maximum number of FFE post-cursor taps.", "number_input", min_value=0, value=5, step=1, key="ffe_post")
        ffe_limit = setting_row("FFE sum(abs) limit", "Maximum absolute sum of all TX FFE taps.", "number_input", min_value=0.001, value=1.0, key="ffe_limit")
    else:
        ffe_pre = int(st.session_state.get("ffe_pre", 3))
        ffe_post = int(st.session_state.get("ffe_post", 5))
        ffe_limit = float(st.session_state.get("ffe_limit", 1.0))
    if eq_mode in {"DFE", "BOTH"}:
        dfe_taps = setting_row("DFE max taps", "Maximum number of DFE post-cursor taps.", "number_input", min_value=0, value=8, step=1, key="dfe_taps")
    else:
        dfe_taps = int(st.session_state.get("dfe_taps", 8))
    if eq_mode != "NONE":
        stop_improvement = setting_row("Tap stop improvement [%]", "Stop adding taps below this incremental improvement.", "number_input", min_value=0.0, value=1.0, key="stop_improvement")
    else:
        stop_improvement = float(st.session_state.get("stop_improvement", 1.0))

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
}

browser_settings = {
    "channel_mode": channel_mode_label,
    "tx_port": int(tx_port),
    "rx_port": int(rx_port),
    "tx_pos_port": int(tx_pos_port),
    "tx_neg_port": int(tx_neg_port),
    "rx_pos_port": int(rx_pos_port),
    "rx_neg_port": int(rx_neg_port),
    "matched_rx": bool(matched),
    "rx_termination_se": float(st.session_state.get("rx_termination_se", 50.0)),
    "rx_termination_diff": float(st.session_state.get("rx_termination_diff", 100.0)),
    "symbol_rate": float(symbol_rate),
    "tx_voltage": float(tx_voltage),
    "pulse_width": float(pulse_width),
    "rise_time": float(rise_time),
    "fall_time": float(fall_time),
    "cursor_pre": int(cursor_pre),
    "cursor_post": int(cursor_post),
    "time_unit": time_unit,
    "decimal_places": int(decimal_places),
    "ctle_db": float(ctle_db),
    "eq_mode": eq_mode_label,
    "ffe_pre": int(ffe_pre),
    "ffe_post": int(ffe_post),
    "ffe_limit": float(ffe_limit),
    "dfe_taps": int(dfe_taps),
    "stop_improvement": float(stop_improvement),
}
browser_settings_json = json.dumps(browser_settings, ensure_ascii=False, separators=(",", ":"))
streamlit_js_eval(
    js_expressions=(
        "window.sessionStorage.setItem('sbrAnalyzerSettings', "
        f"{json.dumps(browser_settings_json)}); true"
    ),
    want_output=False,
    key="save_browser_settings",
)

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
        plot_context = results["plot_context"]
        result_mode = str(plot_context["eq_mode"])
        ctle_enabled = float(plot_context["ctle_db"]) > 0.0
        ffe_enabled = result_mode in {"FFE", "BOTH"}
        dfe_enabled = result_mode in {"DFE", "BOTH"}

        st.markdown("**Graph Options**")
        graph_items = [
            ("tx", "TX pulse"),
            ("channel", "Channel"),
        ]
        if ffe_enabled:
            graph_items.insert(1, ("ffe_output", "FFE output"))
            graph_items.append(("after_ffe", "FFE + Channel"))
        if ctle_enabled:
            ctle_label = "FFE + Channel + CTLE" if ffe_enabled else "Channel + CTLE"
            graph_items.append(("after_ctle", ctle_label))
        if dfe_enabled:
            prefix = "FFE + Channel" if ffe_enabled else "Channel"
            if ctle_enabled:
                prefix += " + CTLE"
            graph_items.append(("after_dfe", f"{prefix} + DFE"))

        graph_values = {
            "tx": False,
            "ffe_output": False,
            "channel": False,
            "after_ffe": False,
            "after_ctle": False,
            "after_dfe": False,
        }
        graph_columns = st.columns(len(graph_items))
        for column, (key, label) in zip(graph_columns, graph_items):
            graph_values[key] = column.checkbox(label, value=True, key=f"result_graph_{key}")
        graph_values["samples"] = st.checkbox(
            "Sample markers (1 UI interval)", value=True, key="result_graph_samples"
        )

        chart_config = {
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": "sbr_analysis_pulse"},
        }
        with st.container(key="desktop_pulse_plot"):
            st.plotly_chart(
                interactive_pulse_figure(results, graph_values, mobile_layout=False),
                use_container_width=True,
                config=chart_config,
                key="desktop_pulse_chart",
            )
        with st.container(key="mobile_pulse_plot"):
            st.plotly_chart(
                interactive_pulse_figure(results, graph_values, mobile_layout=True),
                use_container_width=True,
                config=chart_config,
                key="mobile_pulse_chart",
            )
        st.caption("The camera icon in the graph toolbar saves the currently visible traces as a PNG.")
    with magnitude_tab:
        st.image(results["magnitude"], use_container_width=True)
    with cursor_tab:
        cursor_display = fixed_decimal_csv(
            results["cursors"], int(results["plot_context"]["decimal_places"])
        )
        st.dataframe(csv_rows(cursor_display), use_container_width=True, hide_index=True)
    with tap_tab:
        tap_display = fixed_decimal_csv(
            results["taps"], int(results["plot_context"]["decimal_places"])
        )
        st.dataframe(csv_rows(tap_display), use_container_width=True, hide_index=True)

    st.subheader("Save Options")
    save_columns = st.columns(4)
    save_selection = {
        "pulse": save_columns[0].checkbox("Pulse PNG", value=True),
        "magnitude": save_columns[1].checkbox("Magnitude PNG", value=True),
        "cursors": save_columns[2].checkbox("Cursor CSV", value=True),
        "taps": save_columns[3].checkbox("EQ Taps CSV", value=True),
    }
    if any(save_selection.values()):
        def make_selected_download() -> bytes:
            download_results = dict(results)
            if save_selection["pulse"]:
                download_results["pulse"] = render_session_pulse(results, graph_values)
            return selected_results_zip(download_results, save_selection)

        st.download_button(
            "Download Selected Results",
            data=make_selected_download,
            file_name="sbr_analysis.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
            on_click="ignore",
        )
    else:
        st.warning("Select at least one result to download.")

    with st.expander("Run Log"):
        st.code(results["log"] or "Analysis completed.", language="text")

st.caption("SBR Analyzer · Version 1.0 · © 2026 Young-Min Lee")
