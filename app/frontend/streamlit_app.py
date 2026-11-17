
"""Professional research dashboard for Multi-Agent Research Analyst."""

import html

import requests
import streamlit as st

from app.frontend.components.research_view import render_task_status
from app.frontend.utils.api_client import ResearchAPIClient

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Research Analyst",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================

AGENTS = [
    {
        "number": "01",
        "name": "Supervisor",
        "description": "Plans and coordinates the research workflow.",
    },
    {
        "number": "02",
        "name": "Research Workers",
        "description": "Gather information, sources and relevant evidence.",
    },
    {
        "number": "03",
        "name": "Writer",
        "description": "Creates a structured and comprehensive report.",
    },
    {
        "number": "04",
        "name": "Critic",
        "description": "Reviews quality, clarity and research completeness.",
    },
    {
        "number": "05",
        "name": "Fact Checker",
        "description": "Validates claims and checks supporting evidence.",
    },
]

EXAMPLES = [
    "Impact of generative AI on healthcare diagnostics",
    "Latest developments in multi-agent AI systems",
    "Future of renewable energy technologies",
]

STATUS_LABELS = {
    "queued": "Queued",
    "running": "Running",
    "awaiting_approval": "Awaiting approval",
    "approval_queued": "Approval queued",
    "completed": "Completed",
    "partial": "Partially completed",
    "rejected": "Rejected",
    "failed": "Failed",
}


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "task_id": "",
    "task_data": None,
    "dark_mode": True,
    "topic_input": "",
    "task_history": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# THEME
# =========================================================

LIGHT_THEME = """
--bg: #F5F7FC;
--sidebar: #FFFFFF;
--card: #FFFFFF;
--surface: #F0F3FA;
--ink: #172033;
--muted: #68758B;
--line: #E3E8F2;
--accent: #635BDB;
--accent-hover: #5148C8;
--accent-soft: #EFEEFF;
--success: #16845B;
--warning: #B7791F;
--danger: #D14343;
--placeholder: #64748B;
--shadow: 0 5px 20px rgba(31, 41, 70, .045);
color-scheme: light;
"""

DARK_THEME = """
--bg: #0C1220;
--sidebar: #111A2B;
--card: #141F32;
--surface: #1A2940;
--ink: #EDF2FC;
--muted: #9AA9BF;
--line: #293951;
--accent: #948BFF;
--accent-hover: #A9A2FF;
--accent-soft: #292849;
--success: #4DD4A0;
--warning: #F2C46D;
--danger: #FF8585;
--placeholder: #94A3B8;
--shadow: 0 5px 25px rgba(0, 0, 0, .15);
color-scheme: dark;
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], button, input, textarea {
    font-family: 'Inter', sans-serif !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: var(--sidebar) !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--line);
}

[data-testid="stSidebarContent"] {
    padding: 1.4rem 1.1rem;
}

[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
footer {
    display: none !important;
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* General typography */

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stWidgetLabel"] p {
    color: var(--ink);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--accent) !important;
}

/* Brand */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 43px;
    height: 43px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #7770EF, #4F46C8);
    color: white;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: -.5px;
}

.brand-title {
    font-size: 18px;
    font-weight: 750;
    color: var(--ink);
}

.brand-subtitle {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
}

/* Topbar */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--line);
}

.topbar-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
}

/* Hero */

.hero {
    padding: 48px 0 35px;
}

.hero-label {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 30px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .09em;
    margin-bottom: 20px;
}

.hero-title {
    color: var(--ink);
    font-size: clamp(36px, 5vw, 55px);
    line-height: 1.13;
    font-weight: 800;
    letter-spacing: -2.5px;
    max-width: 850px;
}

.hero-highlight {
    color: var(--accent);
}

.hero-description {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.8;
    max-width: 720px;
    margin-top: 17px;
}

/* Sections */

.section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 37px;
    margin-bottom: 19px;
}

.section-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 9px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 12px;
    font-weight: 800;
}

.section-title {
    color: var(--ink);
    font-size: 21px;
    font-weight: 700;
    letter-spacing: -.5px;
}

.section-description {
    color: var(--muted);
    font-size: 12px;
    margin-left: auto;
}

/* Workflow cards */

.workflow-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 17px 15px;
    min-height: 150px;
    box-shadow: var(--shadow);
    transition: transform .2s ease, border-color .2s ease;
}

.workflow-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent);
}

.workflow-index {
    color: var(--accent);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    margin-bottom: 17px;
}

.workflow-name {
    color: var(--ink);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
}

.workflow-description {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.65;
}

/* Forms */

[data-testid="stForm"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 22px;
    box-shadow: var(--shadow);
}


.stTextInput input,
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 9px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
input::placeholder,
textarea::placeholder {
    color: var(--placeholder) !important;
    opacity: 1 !important;
}

[data-testid="stAppViewContainer"] input::placeholder,
[data-testid="stAppViewContainer"] textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

.stApp[style*="dark"] input::placeholder,
.stApp[style*="dark"] textarea::placeholder {
    color: #94A3B8 !important;
    opacity: 1 !important;
}


.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

[data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 42px;
    transition: all .18s ease;
}

button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #FFFFFF !important;
}

button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}

.stApp button p,
.stApp button span {
    color: inherit;
}

/* Cards and containers */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--line) !important;
    border-radius: 13px;
    box-shadow: var(--shadow);
}

/* Metrics */

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 17px;
    box-shadow: var(--shadow);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--ink) !important;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
}

/* Expander and alerts */

[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--line) !important;
    border-radius: 10px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Sidebar */

.sidebar-heading {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 27px 0 13px;
}

.sidebar-agent {
    padding: 9px 0;
    border-bottom: 1px solid var(--line);
}

.sidebar-agent-name {
    color: var(--ink);
    font-size: 12px;
    font-weight: 650;
}

.sidebar-agent-description {
    color: var(--muted);
    font-size: 10px;
    line-height: 1.5;
    margin-top: 4px;
}

/* Status badges */

.status-badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 700;
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--ink);
}

.status-running {
    color: var(--accent);
    background: var(--accent-soft);
    border-color: var(--accent);
}

.status-success {
    color: var(--success);
    border-color: var(--success);
}

.status-warning {
    color: var(--warning);
    border-color: var(--warning);
}

.status-error {
    color: var(--danger);
    border-color: var(--danger);
}

/* Footer */

.footer {
    text-align: center;
    padding: 27px 0 8px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 11px;
    margin-top: 50px;
}

@media (max-width: 850px) {
    .hero-title {
        font-size: 38px;
    }

    .section-description {
        display: none;
    }
}
"""


def build_css() -> str:
    variables = DARK_THEME if st.session_state.dark_mode else LIGHT_THEME
    return f"<style>:root {{{variables}}}{BASE_CSS}</style>"


st.markdown(build_css(), unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================

api_client = ResearchAPIClient()


def section(number: str, title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-number">{html.escape(number)}</span>
            <span class="section-title">{html.escape(title)}</span>
            <span class="section-description">{html.escape(description)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_status_class(status: str) -> str:
    if status == "completed":
        return "status-success"

    if status in {"running", "approval_queued"}:
        return "status-running"

    if status in {"queued", "awaiting_approval", "partial"}:
        return "status-warning"

    if status in {"failed", "rejected"}:
        return "status-error"

    return ""


def render_status(status: str) -> None:
    label = STATUS_LABELS.get(
        status,
        status.replace("_", " ").title(),
    )
    css_class = get_status_class(status)

    st.markdown(
        f'<span class="status-badge {css_class}">'
        f'{html.escape(label)}</span>',
        unsafe_allow_html=True,
    )


def set_example(topic: str) -> None:
    st.session_state.topic_input = topic


def show_api_error(error: requests.RequestException) -> None:
    if error.response is not None:
        try:
            detail = error.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None

        if detail:
            st.error(str(detail))
            return

    st.error(
        "Unable to connect to the backend. "
        "Please make sure the FastAPI server is running."
    )


def fetch_task(task_id: str) -> None:
    try:
        with st.spinner("Retrieving research task..."):
            data = api_client.get_research_status(task_id)

        st.session_state.task_id = task_id
        st.session_state.task_data = data

        query = data.get("query", "Research task")

        if not any(
            item["task_id"] == task_id
            for item in st.session_state.task_history
        ):
            st.session_state.task_history.append(
                {"task_id": task_id, "query": query}
            )

    except requests.RequestException as error:
        show_api_error(error)


def get_report_text(result) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ("content", "report", "final_report"):
            value = result.get(key)
            if isinstance(value, str):
                return value

    return ""


def get_word_count(result) -> int:
    report = get_report_text(result)
    return len(report.split()) if report else 0


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">RA</div>
            <div>
                <div class="brand-title">Research Analyst</div>
                <div class="brand-subtitle">Intelligent research workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">Research team</div>',
        unsafe_allow_html=True,
    )

    for agent in AGENTS:
        st.markdown(
            f"""
            <div class="sidebar-agent">
                <div class="sidebar-agent-name">
                    {agent["number"]}. {html.escape(agent["name"])}
                </div>
                <div class="sidebar-agent-description">
                    {html.escape(agent["description"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-heading">Appearance</div>',
        unsafe_allow_html=True,
    )

    st.toggle(
        "Dark appearance",
        key="dark_mode",
    )

    st.markdown(
        '<div class="sidebar-heading">Recent research</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.task_history:
        for item in reversed(st.session_state.task_history[-5:]):
            title = item["query"]

            if len(title) > 32:
                title = title[:29] + "..."

            st.button(
                title,
                key=f'history_{item["task_id"]}',
                on_click=fetch_task,
                args=(item["task_id"],),
                use_container_width=True,
            )
    else:
        st.caption("Your recent tasks will appear here.")

    st.divider()
    st.caption("Multi-Agent Research Analyst | Version 1.0")


# =========================================================
# TOPBAR
# =========================================================

top_left, top_right = st.columns([5, 1.5], vertical_alignment="center")

with top_left:
    st.markdown(
        '<div class="topbar-label">RESEARCH INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        '<div style="text-align:right;color:var(--muted);'
        'font-size:12px;">Workspace v1.0</div>',
        unsafe_allow_html=True,
    )



# =========================================================
# HERO
# =========================================================


st.html(
    """
    <div class="hero">
        <div class="hero-label">MULTI-AGENT RESEARCH SYSTEM</div>

        <div class="hero-title">
            Research with intelligence.<br>
            <span class="hero-highlight">Reports with evidence.</span>
        </div>

        <div class="hero-description">
            Explore complex topics with a coordinated team of AI agents.
            Generate structured research reports, evaluate findings,
            and maintain control through human review.
        </div>
    </div>
    """
)


# =========================================================
# WORKFLOW
# =========================================================

section(
    "01",
    "Your research team",
    "Five coordinated components",
)

workflow_columns = st.columns(5, gap="small")

for column, agent in zip(workflow_columns, AGENTS, strict=False):
    with column:
        st.markdown(
            f"""
            <div class="workflow-card">
                <div class="workflow-index">
                    COMPONENT {agent["number"]}
                </div>
                <div class="workflow-name">
                    {html.escape(agent["name"])}
                </div>
                <div class="workflow-description">
                    {html.escape(agent["description"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    "Workflow overview. Individual agent execution telemetry "
    "is not currently exposed by the backend."
)


# =========================================================
# RESEARCH INPUT
# =========================================================

section(
    "02",
    "Create a research brief",
    "Define your research objective",
)

with st.form("research_form", clear_on_submit=False):

    topic = st.text_area(
        "Research objective",
        key="topic_input",
        height=130,
        placeholder=(
            "Describe the topic, scope, key questions and "
            "the outcomes you want to understand..."
        ),
    )

    st.caption(
        "Provide a clear research objective with at least 8 characters."
    )

    run_clicked = st.form_submit_button(
        "Start research",
        type="primary",
        use_container_width=True,
    )


st.markdown("**Need inspiration? Try a suggested topic.**")

example_columns = st.columns(3)

for index, (column, example) in enumerate(
    zip(example_columns, EXAMPLES, strict=False)
):
    with column:
        st.button(
            example,
            key=f"example_{index}",
            on_click=set_example,
            args=(example,),
            use_container_width=True,
        )


# =========================================================
# SUBMIT RESEARCH
# =========================================================

if run_clicked:

    clean_topic = topic.strip()

    if len(clean_topic) < 8:
        st.warning(
            "Please enter a research objective with at least 8 characters."
        )

    else:
        try:
            with st.spinner("Submitting your research request..."):
                response = api_client.submit_research(clean_topic)

            task_id = response["task_id"]

            st.session_state.task_id = task_id
            st.session_state.task_data = None

            if not any(
                item["task_id"] == task_id
                for item in st.session_state.task_history
            ):
                st.session_state.task_history.append(
                    {
                        "task_id": task_id,
                        "query": clean_topic,
                    }
                )

            st.success(response.get("message", "Research request submitted."))

            st.markdown("**Your research task ID**")
            st.code(task_id, language=None)

        except requests.RequestException as error:
            show_api_error(error)


# =========================================================
# TASK TRACKING
# =========================================================

section(
    "03",
    "Research activity",
    "Track and retrieve your research",
)

with st.container(border=True):

    task_id_input = st.text_input(
        "Research task ID",
        value=st.session_state.task_id,
        placeholder="Enter your research task ID",
    )

    if st.button(
        "Retrieve research status",
        type="primary",
        use_container_width=True,
    ):
        if not task_id_input.strip():
            st.warning("Please enter a research task ID.")
        else:
            fetch_task(task_id_input.strip())


# =========================================================
# RESULTS
# =========================================================

task_data = st.session_state.task_data

if task_data:

    section(
        "04",
        "Research results",
        "Your report and execution details",
    )

    status = task_data.get("status", "unknown")
    result = task_data.get("result")
    query_text = task_data.get("query", "Research task")

    st.markdown("**Current execution status**")
    render_status(status)

    metric_columns = st.columns(3)

    with metric_columns[0]:
        st.metric(
            "Task status",
            STATUS_LABELS.get(status, status.title()),
        )

    with metric_columns[1]:
        st.metric(
            "Report words",
            f"{get_word_count(result):,}",
        )

    with metric_columns[2]:
        st.metric(
            "Workflow components",
            len(AGENTS),
        )

    st.markdown("**Research objective**")
    st.write(query_text)

    report_tab, details_tab, raw_tab = st.tabs(
        ["Research report", "Task details", "Raw result"]
    )

    with report_tab:

        if result:

            render_task_status(task_data)

            report_text = get_report_text(result)

            if report_text:
                st.download_button(
                    "Download research report",
                    data=report_text,
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        else:
            st.info(
                "Your research report is not available yet. "
                "Retrieve the task status again as the workflow progresses."
            )

    with details_tab:

        st.markdown("**Task ID**")
        st.code(task_data.get("task_id", ""), language=None)

        st.markdown("**Created at**")
        st.write(task_data.get("created_at", "Not available"))

        st.markdown("**Execution status**")
        render_status(status)

    with raw_tab:
        st.json(task_data)


# =========================================================
# HUMAN APPROVAL
# =========================================================

if task_data and task_data.get("status") == "awaiting_approval":

    section(
        "05",
        "Human review",
        "Review and approve the generated report",
    )

    with st.container(border=True):

        st.markdown("**Review feedback**")

        feedback = st.text_area(
            "Feedback for the research team",
            placeholder=(
                "Mention missing information, incorrect claims "
                "or improvements required..."
            ),
            label_visibility="collapsed",
        )

        approve_column, revision_column = st.columns(2)

        with approve_column:

            if st.button(
                "Approve report",
                type="primary",
                use_container_width=True,
            ):
                try:
                    response = api_client.submit_approval(
                        task_data["task_id"],
                        approved=True,
                    )

                    st.success(response["message"])
                    st.session_state.task_data = None
                    st.rerun()

                except requests.RequestException as error:
                    show_api_error(error)

        with revision_column:

            if st.button(
                "Request revision",
                use_container_width=True,
            ):
                try:
                    response = api_client.submit_approval(
                        task_data["task_id"],
                        approved=False,
                        feedback=feedback,
                    )

                    st.success(response["message"])
                    st.session_state.task_data = None
                    st.rerun()

                except requests.RequestException as error:
                    show_api_error(error)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Multi-Agent Research Analyst
        &nbsp;|&nbsp; Intelligent Research Workspace
        &nbsp;|&nbsp; Version 1.0
    </div>
    """,
    unsafe_allow_html=True,
)
