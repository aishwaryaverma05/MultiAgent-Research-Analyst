
"""Streamlit components for displaying research tasks."""

import json
from typing import Any

import streamlit as st


def render_research_result(result: dict[str, Any] | None) -> None:
    """Display the generated research report."""

    if not result:
        st.info("Research report is not available yet.")
        return

    st.subheader("Research Report")

    if isinstance(result, dict):
        report_text = result.get("content") or result.get("report")

        if report_text:
            st.markdown(str(report_text))
        else:
            st.json(result)
    else:
        st.write(result)


def render_task_status(task: dict[str, Any]) -> None:
    """Display task status and report details."""

    status_value = task.get("status", "unknown")
    result = task.get("result")

    st.subheader("Research Status")

    status_messages = {
        "queued": "Research request is waiting in the queue.",
        "running": "Research is currently in progress.",
        "awaiting_approval": "Your report is waiting for human review.",
        "approval_queued": "Your approval decision is queued.",
        "completed": "Research completed successfully.",
        "partial": "Research completed with partial results.",
        "rejected": "Research report was rejected.",
        "failed": "Research execution failed.",
    }

    st.write(f"**Status:** {status_value.replace('_', ' ').title()}")
    st.info(status_messages.get(status_value, "Current task status."))

    if result:
        render_research_result(result)

        with st.expander("View raw report data"):
            st.code(
                json.dumps(result, indent=2, ensure_ascii=False, default=str),
                language="json",
            )
