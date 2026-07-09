"""Streamlit front-end for the BAF agentic fraud-triage service.

Fills an application form, calls the FastAPI `/predict` endpoint, and shows the
fraud probability, deterministic risk route, and the per-instance SHAP risk
factors returned by the model. Run with:

    streamlit run frontend/streamlit_app.py
"""
import pandas as pd
import requests
import streamlit as st

# Public URL of the deployed FastAPI service (change if you redeploy elsewhere).
DEFAULT_API = "https://baf-fraud-api.onrender.com"

# A realistic sample application, embedded so the app is self-contained when
# deployed to Streamlit Community Cloud (no dependency on the repo's data files).
SAMPLE = {
    "income": 0.9, "name_email_similarity": 0.1090901760520179, "prev_address_months_count": -1,
    "current_address_months_count": 101, "customer_age": 60, "days_since_request": 0.008267058448347,
    "intended_balcon_amount": -0.6050493597585469, "payment_type": "AC", "zip_count_4w": 641,
    "velocity_6h": 4672.487784842243, "velocity_24h": 4107.3269985284005, "velocity_4w": 5046.035049730841,
    "bank_branch_count_8w": 0, "date_of_birth_distinct_emails_4w": 1, "employment_status": "CA",
    "credit_risk_score": 240, "email_is_free": 1, "housing_status": "BC", "phone_home_valid": 0,
    "phone_mobile_valid": 1, "bank_months_count": -1, "has_other_cards": 0, "proposed_credit_limit": 1900.0,
    "foreign_request": 0, "source": "INTERNET", "session_length_in_minutes": 3.869309047491965,
    "device_os": "windows", "keep_alive_session": 0, "device_distinct_emails_8w": 1, "device_fraud_count": 0,
}

# Allowed categories in the BAF Base dataset (for the dropdowns).
CATEGORIES = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["windows", "macintosh", "linux", "x11", "other"],
}
BINARY = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
          "has_other_cards", "foreign_request", "keep_alive_session"]

st.set_page_config(page_title="BAF Fraud Triage", page_icon="🛡️", layout="wide")
st.title("🛡️ Bank Account-Opening Fraud Triage")
st.caption("Scores an application with the deployed model and shows the SHAP risk factors "
           "the AI agent reasons over. Prediction + explanation only — the allow/review/block "
           "decision stays on deterministic thresholds.")

# --- Sidebar: where the API lives ---
st.sidebar.header("Settings")
api_url = st.sidebar.text_input(
    "API base URL",
    value=DEFAULT_API,
    help="The deployed Render service by default; change to http://127.0.0.1:8000 for local testing.",
).rstrip("/")
if st.sidebar.button("Check /health"):
    try:
        h = requests.get(f"{api_url}/health", timeout=60).json()
        st.sidebar.success(f"OK · {h.get('model')} · threshold {h.get('operating_threshold'):.3f}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Not reachable: {exc}")

# --- Application input form (pre-filled with a real sample) ---
st.subheader("Application details")
values = {}
with st.form("application"):
    tabs = st.tabs(["Applicant / financial", "Identity & contact", "Behaviour / velocity", "Device / channel"])
    groups = {
        "Applicant / financial": ["income", "customer_age", "employment_status", "housing_status",
                                   "proposed_credit_limit", "credit_risk_score", "has_other_cards",
                                   "intended_balcon_amount"],
        "Identity & contact": ["name_email_similarity", "email_is_free", "phone_home_valid",
                                "phone_mobile_valid", "date_of_birth_distinct_emails_4w",
                                "prev_address_months_count", "current_address_months_count",
                                "bank_months_count"],
        "Behaviour / velocity": ["velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w",
                                  "bank_branch_count_8w", "days_since_request",
                                  "session_length_in_minutes", "keep_alive_session"],
        "Device / channel": ["device_os", "source", "foreign_request", "device_distinct_emails_8w",
                              "payment_type", "device_fraud_count"],
    }
    for tab, (label, fields) in zip(tabs, groups.items()):
        with tab:
            cols = st.columns(2)
            for i, field in enumerate(fields):
                col = cols[i % 2]
                default = SAMPLE[field]
                if field in CATEGORIES:
                    opts = CATEGORIES[field]
                    values[field] = col.selectbox(field, opts,
                                                  index=opts.index(default) if default in opts else 0)
                elif field in BINARY:
                    values[field] = col.selectbox(field, [0, 1], index=int(default))
                elif isinstance(default, int):
                    values[field] = int(col.number_input(field, value=int(default), step=1))
                else:
                    values[field] = float(col.number_input(field, value=float(default), format="%.6f"))
    submitted = st.form_submit_button("🔎 Score application", use_container_width=True)

# --- Call the API and render the result ---
if submitted:
    try:
        with st.spinner("Scoring… (first call to a sleeping cloud service can take ~50s)"):
            resp = requests.post(f"{api_url}/predict", json=values, timeout=90)
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        st.stop()

    prob = result["fraud_probability"]
    level = result["risk_level"]
    action = result["deterministic_action"]
    colour = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(level, "⚪")

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud probability", f"{prob:.1%}")
    m2.metric("Risk level", f"{colour} {level.upper()}")
    m3.metric("Deterministic action", action.replace("_", " "))
    st.caption(f"Operating threshold: {result['operating_threshold']:.3f} · "
               f"explanation: {result['explanation_method']}")

    st.markdown("#### Top risk factors (per-instance SHAP)")
    rf = pd.DataFrame(result["top_risk_factors"])
    chart = rf.assign(feature=rf["feature"]).set_index("feature")["shap_value"]
    st.bar_chart(chart)
    st.dataframe(rf, use_container_width=True, hide_index=True)

    with st.expander("Raw API response"):
        st.json(result)
