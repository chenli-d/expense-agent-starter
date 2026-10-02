"""Streamlit expense claim pre-screen (SPEC section 8, owner B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hmac
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from app.llm import GeminiLLM, RuleBasedFakeLLM
from app.models import Claim
from app.pipeline import run_pipeline
from app.trace import run_trace


ATTACHMENTS = [
    "itemized_hotel_receipt", "card_slip", "flight_itinerary", "meal_receipt",
    "attendee_list", "pre_approval", "mileage_log", "manager_approval",
]
MAX_RUNS = 20


def setting(name: str) -> str:
    """Prefer Streamlit secrets, with an environment fallback for local runs."""
    try:
        value = st.secrets.get(name)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        value = None
    return str(value) if value is not None else os.getenv(name, "")


def main() -> None:
    st.set_page_config(page_title="Expense Claim Pre-screen")
    st.title("Expense Claim Pre-screen")
    st.caption("A human reviewer makes the final decision.")

    required_code = setting("APP_ACCESS_CODE")
    if not required_code:
        st.error("Set APP_ACCESS_CODE in Streamlit secrets or the environment to enable pre-screening.")
        st.stop()
    entered_code = st.text_input("Access code", type="password")
    if not hmac.compare_digest(entered_code.encode(), required_code.encode()):
        if entered_code:
            st.error("Incorrect access code.")
        st.stop()

    st.session_state.setdefault("pre_screen_runs", 0)
    runs = st.session_state.pre_screen_runs
    st.caption(f"Runs used this session: {runs}/{MAX_RUNS}")
    if runs >= MAX_RUNS:
        st.warning("This session has reached the limit of 20 runs.")

    with st.form("expense_claim"):
        description = st.text_area("Description")
        trip_type = st.selectbox("Trip type", ["domestic", "international"])
        total = st.number_input("Total", min_value=0.0, value=0.0, step=1.0)
        attachments = st.multiselect("Attachments", ATTACHMENTS)
        submitted = st.form_submit_button("Pre-screen", disabled=runs >= MAX_RUNS)

    if submitted and st.session_state.pre_screen_runs < MAX_RUNS:
        st.session_state.pre_screen_runs += 1
        st.session_state.pop("pre_screen_outcome", None)
        st.session_state.pop("pre_screen_error", None)
        with run_trace() as run:
            try:
                llm = (
                    RuleBasedFakeLLM() if setting("USE_FAKE") == "1"
                    else GeminiLLM(api_key=setting("GEMINI_API_KEY"), model=setting("MODEL"))
                )
                claim = Claim(description=description, trip_type=trip_type,
                              total=total, attachments=attachments)
                with st.spinner("Pre-screening claim…"):
                    st.session_state.pre_screen_outcome = run_pipeline(llm, claim)
            except Exception:
                st.session_state.pre_screen_error = (
                    "Pre-screening failed. Check the model configuration and try again."
                )
            finally:
                st.session_state.pre_screen_steps = list(run.steps)

    if st.session_state.get("pre_screen_error"):
        st.error(st.session_state.pre_screen_error)
    outcome = st.session_state.get("pre_screen_outcome")
    if outcome is not None:
        if outcome.error:
            st.error(f"Pre-screening error: {outcome.error}")
        if outcome.report is not None:
            st.write(f"Status: {outcome.report.status}")
            st.write(outcome.report.message)
        for finding in outcome.findings:
            st.write(f"{finding.severity.upper()} · {finding.rule_id}")
            st.write(finding.message)
            st.caption(f"Source: {finding.source}")
        if outcome.estimate is not None:
            st.write("Processing estimate")
            st.json(outcome.estimate)

    with st.expander("Agent steps"):
        steps = st.session_state.get("pre_screen_steps", [])
        for step in steps:
            st.json(step)
        if not steps:
            st.caption("No trace steps recorded.")


if __name__ == "__main__":
    main()
