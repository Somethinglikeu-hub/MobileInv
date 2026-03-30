"""Pipeline — Run CLI pipeline stages from the dashboard.

Displays:
- 5 stage buttons: Veri Cek, Temizle, Skorla, Sec, Rapor + "Tumunu Calistir"
- Options: ticker filter, prices-only, dry-run
- Visual progress bar, elapsed timer, live log during execution
- Cache clear on success
"""

import re
import time

import streamlit as st

from bist_picker.dashboard.pipeline_runner import PipelineResult, stream_command

# Pattern to extract progress like "45/606" or "123 / 606" from output
_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
# Pattern to extract percentage like "45%" or "100%"
_PERCENT_RE = re.compile(r"(\d+)%")


def render():
    """Render the Pipeline page."""
    st.header("Pipeline Yonetimi")

    # -- Options --
    with st.expander("Secenekler", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_filter = st.text_input(
                "Ticker Filtre",
                placeholder="orn. THYAO",
                help="Sadece belirtilen hisseyi cek (fetch icin)",
            )
        with col2:
            prices_only = st.checkbox("Sadece Fiyat", value=False)
        with col3:
            dry_run = st.checkbox("Dry Run", value=False, help="DB'ye yazmadan calistir")

    st.divider()

    # -- Stage Buttons --
    stages = [
        ("fetch", "Veri Cek", "Veri indiriliyor..."),
        ("clean", "Temizle", "Veri temizleniyor..."),
        ("score", "Skorla", "Skorlar hesaplaniyor..."),
        ("pick", "Sec", "Portfolyo seciliyor..."),
        ("report", "Rapor", "Rapor olusturuluyor..."),
    ]

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for i, (stage, label, spinner_msg) in enumerate(stages):
        with columns[i]:
            if st.button(label, key=f"btn_{stage}", use_container_width=True):
                _run_stage_live(stage, spinner_msg, ticker_filter, prices_only, dry_run)

    st.divider()

    # -- Run All Button --
    if st.button("Tumunu Calistir", type="primary", use_container_width=True):
        _run_all_live(stages, ticker_filter, prices_only, dry_run)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _parse_progress(line: str) -> float | None:
    """Try to extract a 0.0-1.0 progress value from a line.

    Looks for patterns like '45/606' or '100%'.
    Returns None if no progress pattern found.
    """
    # Try fraction pattern first (more precise): "45/606"
    m = _FRACTION_RE.search(line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return min(current / total, 1.0)

    # Try percentage pattern: "45%"
    m = _PERCENT_RE.search(line)
    if m:
        pct = int(m.group(1))
        return min(pct / 100.0, 1.0)

    return None


def _is_progress_line(line: str) -> bool:
    """Check if line is a Rich progress bar update (not a real log line)."""
    return any(c in line for c in ("━", "─", "█", "▓")) or _FRACTION_RE.search(line) is not None


def _run_streaming(
    stage: str,
    ticker: str,
    prices_only: bool,
    dry_run: bool,
    progress_bar,
    status_placeholder,
    log_placeholder,
) -> PipelineResult:
    """Execute a single stage with streaming output, progress bar, and timer.

    Updates the provided Streamlit placeholders in real-time.
    Returns the PipelineResult when done.
    """
    start = time.time()
    lines: list[str] = []
    log_lines: list[str] = []
    result: PipelineResult | None = None
    current_progress: float = 0.0

    gen = stream_command(
        stage=stage,
        ticker=ticker.strip() or None,
        prices_only=prices_only,
        dry_run=dry_run,
    )

    try:
        while True:
            line = next(gen)
            lines.append(line)
            elapsed = time.time() - start

            # Try to extract progress
            prog = _parse_progress(line)
            if prog is not None:
                current_progress = prog

            is_prog = _is_progress_line(line)

            if not is_prog:
                log_lines.append(line)

            # Update progress bar
            progress_bar.progress(
                current_progress,
                text=f"{current_progress * 100:.0f}%  —  {_fmt_elapsed(elapsed)}  —  {line[:70]}",
            )

            # Update status
            status_placeholder.markdown(
                f"**Sure:** `{_fmt_elapsed(elapsed)}`  &nbsp;|&nbsp;  "
                f"**Ilerleme:** `{current_progress * 100:.0f}%`  &nbsp;|&nbsp;  "
                f"**Satir:** `{len(lines)}`"
            )

            # Live log (last 25 real lines)
            visible = log_lines[-25:]
            log_placeholder.code("\n".join(visible), language="log")

    except StopIteration as e:
        result = e.value

    elapsed = time.time() - start

    if result is None:
        result = PipelineResult(
            success=False,
            command="?",
            stdout="\n".join(lines),
            stderr="Unexpected generator exit.",
            return_code=-1,
        )

    # Final state
    status_icon = "✅" if result.success else "❌"
    progress_bar.progress(
        1.0 if result.success else current_progress,
        text=f"{'Tamamlandi' if result.success else 'Basarisiz'}  —  {_fmt_elapsed(elapsed)}",
    )
    status_placeholder.markdown(
        f"{status_icon} **{'Tamamlandi' if result.success else 'Basarisiz'}** &nbsp;|&nbsp; "
        f"**Sure:** `{_fmt_elapsed(elapsed)}`  &nbsp;|&nbsp;  "
        f"**Toplam satir:** `{len(lines)}`"
    )

    # Final log
    if log_lines:
        log_placeholder.code("\n".join(log_lines)[-5000:], language="log")

    return result


def _run_stage_live(
    stage: str,
    spinner_msg: str,
    ticker: str,
    prices_only: bool,
    dry_run: bool,
):
    """Execute a single pipeline stage with live output."""
    st.subheader(spinner_msg)

    progress_bar = st.progress(0, text="Baslatiliyor...")
    status_placeholder = st.empty()
    log_placeholder = st.empty()

    result = _run_streaming(
        stage, ticker, prices_only, dry_run,
        progress_bar, status_placeholder, log_placeholder,
    )

    if result.success:
        st.success(f"{stage.upper()} basariyla tamamlandi.")
        st.cache_data.clear()
    else:
        st.error(f"{stage.upper()} basarisiz (kod: {result.return_code})")
        if result.stderr:
            with st.expander("Hata Detayi", expanded=True):
                st.code(result.stderr[-3000:], language="log")


def _run_all_live(
    stages: list[tuple[str, str, str]],
    ticker: str,
    prices_only: bool,
    dry_run: bool,
):
    """Execute all pipeline stages sequentially with live output per stage."""
    total = len(stages)
    overall_progress = st.progress(0, text="Pipeline baslatiliyor...")
    all_success = True

    for i, (stage, _label, spinner_msg) in enumerate(stages):
        overall_progress.progress(
            i / total,
            text=f"[{i + 1}/{total}] {spinner_msg}",
        )

        st.subheader(f"[{i + 1}/{total}] {spinner_msg}")
        stage_progress = st.progress(0, text="Baslatiliyor...")
        status_placeholder = st.empty()
        log_placeholder = st.empty()

        result = _run_streaming(
            stage, ticker,
            prices_only if stage == "fetch" else False,
            dry_run,
            stage_progress, status_placeholder, log_placeholder,
        )

        if result.success:
            st.success(f"{stage.upper()} tamamlandi.")
        else:
            st.error(f"{stage.upper()} basarisiz! Pipeline durduruluyor.")
            if result.stderr:
                with st.expander("Hata Detayi", expanded=True):
                    st.code(result.stderr[-3000:], language="log")
            all_success = False
            break

    overall_progress.progress(
        1.0,
        text="Pipeline tamamlandi." if all_success else "Pipeline hata ile durdu.",
    )

    if all_success:
        st.cache_data.clear()
        st.balloons()
