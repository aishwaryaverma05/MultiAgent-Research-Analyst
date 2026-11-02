import html
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st

from pipeline import run_research_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    ("Search", "Collects recent, reliable sources on the topic."),
    ("Read", "Scrapes the most relevant page for depth."),
    ("Write", "Drafts a structured research report."),
    ("Critique", "Reviews the report for gaps and quality."),
]

EXAMPLES = [
    "Impact of generative AI on healthcare diagnostics",
    "Solid-state batteries: current state of the technology",
    "Quantum error correction: recent developments",
]

# ---------------------------------------------------------------------------
# Theme (light / dark) + styling
# ---------------------------------------------------------------------------
LIGHT_VARS = """
  --bg: #FFFFFF; --card: #FFFFFF; --surface: #F8FAFC;
  --ink: #0F172A; --muted: #64748B; --line: #E2E8F0;
  --accent: #1E40AF; --accent-solid: #1E40AF; --accent-soft: rgba(30,64,175,.08);
  --ok: #047857; --err: #B91C1C; --warn: #B45309;
  --chip: #F1F5F9; --dot: #CBD5E1; color-scheme: light;
"""

DARK_VARS = """
  --bg: #0B1220; --card: #0F172A; --surface: #111A2B;
  --ink: #E2E8F0; --muted: #94A3B8; --line: #223047;
  --accent: #60A5FA; --accent-solid: #2563EB; --accent-soft: rgba(96,165,250,.14);
  --ok: #34D399; --err: #F87171; --warn: #FBBF24;
  --chip: #1E293B; --dot: #475569; color-scheme: dark;
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}

/* ---------- App surfaces ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; color: var(--ink); }
header[data-testid="stHeader"] { background: var(--bg) !important; }
section[data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: var(--surface) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebarContent"] { padding-top: 1rem; }

/* Hide only the menu / deploy button. The sidebar toggle stays available. */
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stDecoration"], footer { display: none !important; }

/* Sidebar open / close controls */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {
  display: flex !important; visibility: visible !important; opacity: 1 !important;
}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="stSidebarCollapseButton"] * {
  color: var(--ink) !important; fill: var(--ink) !important;
}

.block-container { padding-top: 5rem !important; padding-bottom: 4rem; max-width: 1120px; }

/* ---------- Typography ---------- */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th,
[data-testid="stWidgetLabel"] p { color: var(--ink) !important; }
[data-testid="stCaptionContainer"] [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
[data-testid="stMarkdownContainer"] a { color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] code { background: var(--chip); color: var(--ink); padding: 2px 6px; border-radius: 4px; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td { border: 1px solid var(--line); padding: 6px 10px; }

/* ---------- Top bar ---------- */
.topbar-brand { display: flex; align-items: center; gap: 12px; }
.logo { width: 38px; height: 38px; border-radius: 10px; background: var(--accent-solid); color: #fff; font-size: 14px; font-weight: 700; letter-spacing: .04em; display: inline-flex; align-items: center; justify-content: center; }
.brand-name { font-size: 19px; font-weight: 700; color: var(--ink); letter-spacing: -.015em; }
.brand-tag { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; }
.topbar-rule { height: 1px; background: var(--line); margin: 10px 0 0; }
.st-key-dark_mode { display: flex; justify-content: flex-end; }

/* ---------- Hero ---------- */
.hero { padding: 48px 0 12px; }
.eyebrow { font-size: 13px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: 14px; }
.title { font-size: 56px; font-weight: 700; letter-spacing: -.03em; color: var(--ink); line-height: 1.08; max-width: 980px; }
.subtitle { font-size: 18px; color: var(--muted); margin-top: 20px; max-width: 720px; line-height: 1.65; }

/* ---------- Section headers ---------- */
.sec { margin: 56px 0 20px; padding-bottom: 16px; border-bottom: 1px solid var(--line); display: flex; align-items: baseline; gap: 16px; }
.sec-num { font-size: 15px; font-weight: 600; letter-spacing: .08em; color: var(--accent); }
.sec-title { font-size: 32px; font-weight: 700; letter-spacing: -.025em; color: var(--ink); line-height: 1.15; }
.sec-desc { font-size: 15px; color: var(--muted); margin-left: auto; text-align: right; }

/* ---------- Form ---------- */
[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 12px; padding: 1.25rem 1.25rem 1rem; background: var(--surface); }
.stTextInput input { border-radius: 8px; padding: .75rem .9rem; font-size: 15px; background: var(--card) !important; color: var(--ink) !important; }
.stTextInput input::placeholder { color: var(--muted) !important; opacity: .85; }
div[data-baseweb="input"], div[data-baseweb="base-input"] { background: var(--card) !important; border-color: var(--line) !important; border-radius: 8px !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 8px; font-weight: 500; font-size: 14px; padding: .55rem 1.1rem; transition: all .15s ease;
}
button[kind="secondary"], button[kind="secondaryFormSubmit"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--card) !important; border: 1px solid var(--line) !important; color: var(--ink) !important;
}
button[kind="secondary"]:hover, button[kind="secondaryFormSubmit"]:hover, [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}
button[kind="primary"], button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--accent-solid) !important; border: 1px solid var(--accent-solid) !important; color: #fff !important;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover, [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { filter: brightness(1.12); }
.stApp button p, .stApp button span { color: inherit !important; }

/* ---------- Stepper ---------- */
.stepper { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.step { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; background: var(--card); transition: all .2s ease; }
.step-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.idx { width: 22px; height: 22px; border-radius: 50%; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; background: var(--chip); color: var(--muted); }
.step-name { font-size: 14px; font-weight: 600; color: var(--ink); }
.step-desc { font-size: 12.5px; color: var(--muted); line-height: 1.45; min-height: 36px; }
.step-status { margin-top: 12px; font-size: 12px; font-weight: 500; color: var(--muted); display: flex; align-items: center; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dot); display: inline-block; margin-right: 8px; }
.step.running { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.step.running .idx { background: var(--accent-solid); color: #fff; }
.step.running .step-status { color: var(--accent); }
.step.running .dot { background: var(--accent); animation: pulse 1.2s ease-in-out infinite; }
.step.done .idx { background: var(--ok); color: #fff; }
.step.done .dot { background: var(--ok); }
.step.done .step-status { color: var(--ok); }
.step.error { border-color: var(--err); }
.step.error .idx { background: var(--err); color: #fff; }
.step.error .dot { background: var(--err); }
.step.error .step-status { color: var(--err); }
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .35; transform: scale(.8); } }

/* ---------- Results header + metrics ---------- */
.result-head { margin: 4px 0 18px; }
.result-title { font-size: 38px; font-weight: 700; letter-spacing: -.025em; color: var(--ink); line-height: 1.2; max-width: 940px; }
.result-meta { font-size: 14px; color: var(--muted); margin-top: 10px; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 8px; }
.metric { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; background: var(--surface); }
.m-label { font-size: 12px; color: var(--muted); font-weight: 500; }
.m-value { font-size: 22px; font-weight: 600; color: var(--ink); margin-top: 4px; letter-spacing: -.01em; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { font-weight: 500; font-size: 14px; padding: .6rem 1rem; color: var(--muted); }
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent-solid) !important; }
.stTabs [data-baseweb="tab-border"] { background: var(--line) !important; }

/* ---------- Report / feedback reading panels ---------- */
.st-key-report_box, .st-key-feedback_box { border: 1px solid var(--line); border-radius: 12px; padding: 1.75rem 2.25rem; background: var(--card); margin-top: 8px; }
.st-key-report_box [data-testid="stMarkdownContainer"] h1, .st-key-feedback_box [data-testid="stMarkdownContainer"] h1 {
  font-size: 32px; font-weight: 700; letter-spacing: -.025em; margin: 0 0 .7em; padding-bottom: .45em; border-bottom: 1px solid var(--line);
}
.st-key-report_box [data-testid="stMarkdownContainer"] h2, .st-key-feedback_box [data-testid="stMarkdownContainer"] h2 {
  font-size: 24px; font-weight: 600; letter-spacing: -.015em; margin: 1.7em 0 .5em;
}
.st-key-report_box [data-testid="stMarkdownContainer"] h3, .st-key-feedback_box [data-testid="stMarkdownContainer"] h3 {
  font-size: 18px; font-weight: 600; margin: 1.4em 0 .4em;
}
.st-key-report_box [data-testid="stMarkdownContainer"] p, .st-key-report_box [data-testid="stMarkdownContainer"] li,
.st-key-feedback_box [data-testid="stMarkdownContainer"] p, .st-key-feedback_box [data-testid="stMarkdownContainer"] li {
  font-size: 15px; line-height: 1.75;
}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 10px; background: var(--card); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }

/* ---------- Callouts ---------- */
.callout { border: 1px solid var(--line); border-left: 3px solid var(--accent); background: var(--surface); border-radius: 8px; padding: 12px 14px; font-size: 14px; color: var(--ink); margin: 8px 0; }
.callout.error { border-left-color: var(--err); }
.callout.warn { border-left-color: var(--warn); }

/* ---------- Sidebar ---------- */
.brand { font-size: 22px; font-weight: 700; color: var(--ink); letter-spacing: -.02em; }
.brand-sub { font-size: 12.5px; color: var(--muted); margin-top: 4px; line-height: 1.5; }
.side-label { font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: 24px 0 8px; }
.side-desc { color: var(--muted); font-size: 12.5px; }

/* ---------- Sources ---------- */
.source-row { padding: 10px 0; border-bottom: 1px solid var(--line); font-size: 14px; }
.source-row a { color: var(--accent); text-decoration: none; font-weight: 500; }
.source-row a:hover { text-decoration: underline; }
.source-url { color: var(--muted); font-size: 12px; word-break: break-all; margin-top: 2px; }

@media (max-width: 800px) {
  .stepper, .metrics { grid-template-columns: repeat(2, 1fr); }
  .title { font-size: 36px; }
  .sec-title { font-size: 26px; }
  .result-title { font-size: 28px; }
  .sec { flex-direction: column; gap: 4px; }
  .sec-desc { margin-left: 0; text-align: left; }
}
"""


def build_css(dark: bool) -> str:
    return ":root {" + (DARK_VARS if dark else LIGHT_VARS) + "}" + BASE_CSS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fresh_states():
    return [{"status": "pending", "started": None, "elapsed": 0.0} for _ in STEPS]


def render_stepper(states) -> str:
    cards = []
    for i, (name, desc) in enumerate(STEPS):
        s = states[i]
        status = s["status"]
        if status == "done":
            label = f"Completed in {s['elapsed']:.1f}s"
        elif status == "running":
            label = "In progress"
        elif status == "error":
            label = "Failed"
        else:
            label = "Pending"
        cards.append(
            f'<div class="step {status}">'
            f'<div class="step-head"><span class="idx">{i + 1}</span><span class="step-name">{name}</span></div>'
            f'<div class="step-desc">{desc}</div>'
            f'<div class="step-status"><span class="dot"></span>{label}</div>'
            f"</div>"
        )
    return '<div class="stepper">' + "".join(cards) + "</div>"


def render_metrics(items) -> str:
    cards = "".join(
        f'<div class="metric"><div class="m-label">{label}</div><div class="m-value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="metrics">{cards}</div>'


def section(num: str, title: str, desc: str = ""):
    desc_html = f'<span class="sec-desc">{html.escape(desc)}</span>' if desc else ""
    st.markdown(
        f'<div class="sec"><span class="sec-num">{num}</span><span class="sec-title">{html.escape(title)}</span>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def callout(kind: str, text: str):
    st.markdown(f'<div class="callout {kind}">{html.escape(text)}</div>', unsafe_allow_html=True)


def extract_urls(*texts):
    seen, urls = set(), []
    for text in texts:
        for url in re.findall(r"https?://[^\s)\]>\"']+", text or ""):
            url = url.rstrip(".,;:")
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def set_example(text: str):
    st.session_state.topic_input = text


def load_from_history(idx: int):
    item = st.session_state.history[idx]
    st.session_state.result = item["result"]
    st.session_state.topic = item["topic"]
    st.session_state.step_states = item["states"]
    st.session_state.duration = item["duration"]
    st.session_state.generated_at = item["time"]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "result": None,
    "topic": "",
    "history": [],
    "topic_input": "",
    "step_states": None,
    "duration": 0.0,
    "generated_at": "",
    "dark_mode": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Inject theme CSS (reads the toggle value from the previous interaction)
st.markdown(f"<style>{build_css(st.session_state.dark_mode)}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">Research Analyst</div>'
        '<div class="brand-sub">A multi-agent system that searches, reads, writes and reviews, end to end.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Workflow</div>', unsafe_allow_html=True)
    for i, (name, desc) in enumerate(STEPS):
        st.markdown(f'**{i + 1}. {name}**  \n<span class="side-desc">{desc}</span>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="side-label">Recent runs</div>', unsafe_allow_html=True)
        for i, item in reversed(list(enumerate(st.session_state.history))):
            title = item["topic"] if len(item["topic"]) <= 34 else item["topic"][:33] + "..."
            st.button(
                f"{title}  ·  {item['time']}",
                key=f"hist_{i}",
                on_click=load_from_history,
                args=(i,),
                use_container_width=True,
            )
        if st.button("Clear history", key="clear_hist", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.session_state.step_states = None
            st.rerun()

# ---------------------------------------------------------------------------
# Top bar + hero
# ---------------------------------------------------------------------------
bar_left, bar_right = st.columns([6, 1.3], vertical_alignment="center")
with bar_left:
    st.markdown(
        '<div class="topbar-brand"><span class="logo">RA</span>'
        '<span class="brand-name">Research Analyst</span>'
        '<span class="brand-tag">Multi-Agent</span></div>',
        unsafe_allow_html=True,
    )
with bar_right:
    st.toggle("Dark mode", key="dark_mode")
st.markdown('<div class="topbar-rule"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">Multi-Agent Research System</div>'
    '<div class="title">Research any topic, with a built-in reviewer.</div>'
    '<div class="subtitle">Enter a topic. Specialised agents gather sources, read the most relevant page, '
    "write a structured report, and critique it before you see it.</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 01 Research brief
# ---------------------------------------------------------------------------
section("01", "Research brief", "Describe what you want researched")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Impact of generative AI on healthcare diagnostics",
    )
    run_clicked = st.form_submit_button("Run research", type="primary")

st.caption("Suggested topics")
ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,), use_container_width=True)

# ---------------------------------------------------------------------------
# 02 Pipeline execution
# ---------------------------------------------------------------------------
if run_clicked and not topic.strip():
    callout("warn", "Please enter a research topic to continue.")

if run_clicked and topic.strip():
    topic = topic.strip()
    states = fresh_states()
    section("02", "Pipeline progress", "Each agent reports status live")
    stepper_slot = st.empty()
    stepper_slot.markdown(render_stepper(states), unsafe_allow_html=True)
    start = time.time()

    def on_progress(step_idx: int, label: str, output):
        if output is None:
            states[step_idx] = {"status": "running", "started": time.time(), "elapsed": 0.0}
        else:
            states[step_idx]["status"] = "done"
            states[step_idx]["elapsed"] = time.time() - states[step_idx]["started"]
        stepper_slot.markdown(render_stepper(states), unsafe_allow_html=True)

    try:
        result = run_research_pipeline(topic, progress_callback=on_progress)
    except Exception as e:
        for s in states:
            if s["status"] == "running":
                s["status"] = "error"
        stepper_slot.markdown(render_stepper(states), unsafe_allow_html=True)
        callout("error", f"The pipeline stopped with an error: {e}")
        st.stop()

    duration = time.time() - start
    stamp = datetime.now().strftime("%H:%M")
    st.session_state.result = result
    st.session_state.topic = topic
    st.session_state.step_states = states
    st.session_state.duration = duration
    st.session_state.generated_at = stamp
    st.session_state.history.append(
        {"topic": topic, "result": result, "states": states, "duration": duration, "time": stamp}
    )

elif st.session_state.step_states and st.session_state.result:
    section("02", "Pipeline progress", "Each agent reports status live")
    st.markdown(render_stepper(st.session_state.step_states), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 03 Results
# ---------------------------------------------------------------------------
result = st.session_state.result
if result:
    report = str(result.get("report", ""))
    feedback = str(result.get("Feedback", ""))
    search_results = str(result.get("search_results", ""))
    scraped = str(result.get("scraped_content", ""))
    urls = extract_urls(search_results, scraped)

    section("03", "Results", "Report, critique and sources")

    st.markdown(
        '<div class="result-head">'
        '<div class="eyebrow">Research report</div>'
        f'<div class="result-title">{html.escape(st.session_state.topic)}</div>'
        f'<div class="result-meta">Generated at {html.escape(st.session_state.generated_at)} '
        f"&nbsp;·&nbsp; {len(STEPS)} agents &nbsp;·&nbsp; {st.session_state.duration:.1f}s total</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        render_metrics(
            [
                ("Total time", f"{st.session_state.duration:.1f}s"),
                ("Report length", f"{len(report.split()):,} words"),
                ("Sources found", str(len(urls))),
                ("Agents run", str(len(STEPS))),
            ]
        ),
        unsafe_allow_html=True,
    )

    tab_report, tab_feedback, tab_sources, tab_raw = st.tabs(
        ["Report", "Critic feedback", "Sources", "Raw research"]
    )

    with tab_report:
        with st.container(key="report_box"):
            st.markdown(report)
        st.download_button(
            "Download report (.md)",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
            key="dl_report",
        )

    with tab_feedback:
        with st.container(key="feedback_box"):
            st.markdown(feedback)
        st.download_button(
            "Download feedback (.md)",
            data=feedback,
            file_name="critic_feedback.md",
            mime="text/markdown",
            key="dl_feedback",
        )

    with tab_sources:
        if urls:
            rows = "".join(
                f'<div class="source-row"><a href="{html.escape(u, quote=True)}" target="_blank" rel="noopener">'
                f'{html.escape(urlparse(u).netloc.replace("www.", ""))}</a>'
                f'<div class="source-url">{html.escape(u)}</div></div>'
                for u in urls
            )
            st.markdown(rows, unsafe_allow_html=True)
        else:
            callout("info", "No URLs were detected in the agent output.")

    with tab_raw:
        with st.expander("Search agent output", expanded=False):
            st.markdown(search_results)
        with st.expander("Reader agent output", expanded=False):
            st.markdown(scraped)