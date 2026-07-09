"""Streamlit UI that calls the /predict API (single + batch scoring)."""
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import altair as alt
    HAS_ALT = True
except Exception:  # pragma: no cover
    HAS_ALT = False

HERE = Path(__file__).parent
DEFAULT_API = "https://baf-fraud-api.onrender.com"

# fallback if sample_pool.csv is missing
HIGH_RISK = {
    "income": 0.9, "name_email_similarity": 0.109, "prev_address_months_count": -1,
    "current_address_months_count": 101, "customer_age": 60, "days_since_request": 0.0083,
    "intended_balcon_amount": -0.605, "payment_type": "AC", "zip_count_4w": 641,
    "velocity_6h": 4672.49, "velocity_24h": 4107.33, "velocity_4w": 5046.04,
    "bank_branch_count_8w": 0, "date_of_birth_distinct_emails_4w": 1, "employment_status": "CA",
    "credit_risk_score": 240, "email_is_free": 1, "housing_status": "BC", "phone_home_valid": 0,
    "phone_mobile_valid": 1, "bank_months_count": -1, "has_other_cards": 0, "proposed_credit_limit": 1900.0,
    "foreign_request": 0, "source": "INTERNET", "session_length_in_minutes": 3.87,
    "device_os": "windows", "keep_alive_session": 0, "device_distinct_emails_8w": 1, "device_fraud_count": 0,
}
LOW_RISK = {**HIGH_RISK, "income": 0.3, "name_email_similarity": 0.95, "prev_address_months_count": 48,
            "customer_age": 35, "phone_home_valid": 1, "proposed_credit_limit": 200.0,
            "bank_months_count": 24, "has_other_cards": 1, "device_os": "macintosh", "email_is_free": 0}
FIELDS = list(HIGH_RISK)

CATEGORIES = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["windows", "macintosh", "linux", "x11", "other"],
}
BINARY = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
          "has_other_cards", "foreign_request", "keep_alive_session"]
NUMERIC = [f for f in FIELDS if f not in CATEGORIES]
GROUPS = {
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


def prettify(feat: str) -> str:
    for cat in CATEGORIES:
        if feat.startswith(cat + "_"):
            return cat.replace("_", " ").capitalize() + ": " + feat[len(cat) + 1:]
    if feat.startswith("missingindicator_"):
        return feat[len("missingindicator_"):].replace("_", " ") + " (missing / unknown)"
    return feat.replace("_", " ")


@st.cache_data(show_spinner=False)
def load_pool():
    for path in (HERE / "sample_pool.csv", Path("frontend/sample_pool.csv"), Path("sample_pool.csv")):
        try:
            if Path(path).exists():
                return pd.read_csv(path)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_reference():
    try:
        return json.loads((HERE / "reference.json").read_text())
    except Exception:
        return {}


REFERENCE = load_reference()


def applicant_vs_population(name, payload):
    # applicant value vs population baseline for one feature
    ref = REFERENCE.get(name)
    if name in NUMERIC:
        av = payload.get(name)
        appl = f"{av:.2f}" if isinstance(av, (int, float)) else str(av)
        if ref is None or not isinstance(av, (int, float)):
            return appl, ("-" if ref is None else f"{ref:.2f}"), ""
        vs = "higher" if av > ref else ("lower" if av < ref else "about average")
        return appl, f"{ref:.2f}", vs
    if name.startswith("missingindicator_"):
        av = payload.get(name[len("missingindicator_"):]) == -1
    else:
        av = any(payload.get(c) == name[len(c) + 1:] for c in CATEGORIES if name.startswith(c + "_"))
    return ("yes" if av else "no"), ("-" if ref is None else f"{ref:.0%}"), ""


st.set_page_config(page_title="BAF Fraud Triage", layout="wide")


@st.cache_data(ttl=120, show_spinner=False)
def get_health(url: str):
    try:
        return requests.get(f"{url}/health", timeout=8).json()
    except Exception:
        return None


def score_one(url: str, payload: dict) -> dict:
    r = requests.post(f"{url}/predict", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


st.title("Bank Account-Opening Fraud Triage")
st.caption("Scores an application with the deployed model and shows the SHAP risk factors the AI "
           "agent reasons over. Prediction and explanation only; the allow / review / block decision "
           "stays on deterministic thresholds.")

st.sidebar.header("Settings")
api_url = st.sidebar.text_input(
    "API base URL", value=DEFAULT_API,
    help="The deployed Render service by default; use http://127.0.0.1:8000 for local testing.",
).rstrip("/")
health = get_health(api_url)
if health:
    st.sidebar.success(f"Online — {health.get('model')}, threshold "
                       f"{health.get('operating_threshold', 0):.3f}")
else:
    st.sidebar.warning("Service is waking up (free tier). The first score may take about 50s; "
                       "sending the request now wakes it.")
if st.sidebar.button("Re-check status"):
    get_health.clear()
    st.rerun()

for _k, _v in HIGH_RISK.items():
    st.session_state.setdefault(f"f_{_k}", _v)


def _set_fields(row: dict):
    for f in FIELDS:
        v = row[f]
        if f in CATEGORIES:
            v = str(v)
        elif isinstance(HIGH_RISK[f], int):
            v = int(v)
        else:
            v = float(v)
        st.session_state[f"f_{f}"] = v


def load_random(kind: str):
    pool = load_pool()
    if pool is None:
        _set_fields(HIGH_RISK if kind == "fraud" else LOW_RISK)
        st.session_state.pop("loaded_payload", None)
        st.session_state.pop("loaded_label", None)
        return
    sub = pool
    if kind == "fraud":
        sub = pool[pool["fraud_bool"] == 1]
    elif kind == "nonfraud":
        sub = pool[pool["fraud_bool"] == 0]
    row = sub.sample(1).iloc[0].to_dict()
    _set_fields(row)
    # keep the true label for the hit/miss check (only valid if not edited)
    st.session_state["loaded_payload"] = {f: st.session_state[f"f_{f}"] for f in FIELDS}
    st.session_state["loaded_label"] = int(row["fraud_bool"])


BANNER = {
    "block_and_escalate": ("#b3261e", "#fdeceb", "BLOCK & ESCALATE"),
    "manual_review": ("#8a6d00", "#fff6e0", "MANUAL REVIEW"),
    "allow": ("#1e7d32", "#e8f4ea", "ALLOW"),
}


def render_result(result: dict, payload: dict, true_label=None):
    action = result["deterministic_action"]
    prob = float(result["fraud_probability"])
    level = result["risk_level"]
    thr = float(result["operating_threshold"])

    fg, bg, txt = BANNER.get(action, ("#333", "#eee", action))
    st.markdown(
        f'<div style="background:{bg};border-left:6px solid {fg};padding:12px 16px;'
        f'border-radius:6px;margin:6px 0;">'
        f'<span style="color:{fg};font-weight:700;font-size:1.05rem;">{txt}</span>'
        f'<span style="color:#444;"> &nbsp;&middot;&nbsp; fraud probability {prob:.1%} '
        f'&nbsp;&middot;&nbsp; risk {level.upper()} &nbsp;&middot;&nbsp; threshold {thr:.3f}</span></div>',
        unsafe_allow_html=True)

    if true_label is not None:
        flagged = action != "allow"
        actual = "FRAUD" if true_label == 1 else "LEGITIMATE"
        if true_label == 1 and flagged:
            verdict, bar = "the model flagged it — a correct catch (true positive).", "#1e7d32"
        elif true_label == 1 and not flagged:
            verdict, bar = "the model allowed it — a MISS (false negative).", "#b3261e"
        elif true_label == 0 and flagged:
            verdict, bar = "the model flagged it — a false alarm (false positive).", "#8a6d00"
        else:
            verdict, bar = "the model allowed it — correctly cleared (true negative).", "#1e7d32"
        st.markdown(
            f'<div style="background:#eef2f7;border-left:6px solid {bar};padding:10px 14px;'
            f'border-radius:6px;margin:6px 0;color:#333;"><b>Ground-truth outcome</b> for this real '
            f'application: <b>{actual}</b>. Model decision: <b>{action.replace("_", " ")}</b> — '
            f'{verdict}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{prob:.1%}")
    c2.metric("Risk level", level.upper())
    c3.metric("Deterministic action", action.replace("_", " "))
    st.progress(min(max(prob, 0.0), 1.0),
                text=f"Fraud probability {prob:.1%} (operating threshold {thr:.1%})")

    st.markdown("#### Why — top risk factors (per-instance SHAP)")
    rf = pd.DataFrame(result["top_risk_factors"])
    rf["Feature"] = rf["feature"].map(prettify)
    rf["Effect"] = rf["direction"].map({"increases_risk": "increases risk",
                                        "decreases_risk": "decreases risk"})
    triples = [applicant_vs_population(n, payload) for n in rf["feature"]]
    rf["Applicant"] = [a for a, _, _ in triples]
    rf["Population"] = [p for _, p, _ in triples]
    rf["vs avg"] = [v for _, _, v in triples]

    if HAS_ALT:
        chart = (alt.Chart(rf).mark_bar().encode(
            x=alt.X("shap_value:Q", title="SHAP contribution to the score (right = increases risk)"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.Color("direction:N", legend=None, scale=alt.Scale(
                domain=["increases_risk", "decreases_risk"], range=["#b3261e", "#3b6ea5"])),
            tooltip=["Feature", "Applicant", "Population", "vs avg", "Effect"]).properties(height=240))
        st.altair_chart(chart, use_container_width=True)

    rf_show = rf[["Feature", "Applicant", "Population", "vs avg", "Effect"]].rename(
        columns={"Population": "Population avg / base rate"})

    def _colour(v):
        return ("color:#b3261e;font-weight:600" if v == "increases risk"
                else "color:#1e7d32;font-weight:600" if v == "decreases risk" else "")
    st.dataframe(rf_show.style.map(_colour, subset=["Effect"]),
                 hide_index=True, use_container_width=True)
    st.caption("**How to read this:** whether a *high* or a *low* value is risky is not the same across "
               "features — for some, a high value raises the fraud score; for others, a low value does. "
               "You do not have to judge that yourself: the **Effect** column is the model's per-application "
               "answer (via SHAP) of whether this exact value is pushing the fraud score up (red) or down "
               "(green). **Applicant / Population / vs avg** only show how unusual the value is.")
    with st.expander("Raw API response"):
        st.json(result)


tab_single, tab_batch = st.tabs(["Single application", "Batch scoring (CSV)"])

with tab_single:
    st.write("Load a real applicant at random (or fill the form manually), then score it.")
    b1, b2, b3, _ = st.columns([1, 1, 1, 1])
    b1.button("Random applicant", on_click=load_random, args=("any",), use_container_width=True)
    b2.button("Random fraud case", on_click=load_random, args=("fraud",), use_container_width=True)
    b3.button("Random legitimate case", on_click=load_random, args=("nonfraud",),
              use_container_width=True)
    # colour the two case buttons by label text
    components.html(
        """<script>
        function colourButtons() {
          const doc = window.parent.document;
          doc.querySelectorAll('button').forEach(function (b) {
            const t = (b.innerText || '').trim();
            if (t === 'Random fraud case') {
              b.style.setProperty('background-color', '#fbeceb', 'important');
              b.style.setProperty('border-color', '#e6b5b1', 'important');
              b.style.setProperty('color', '#b0322a', 'important');
            } else if (t === 'Random legitimate case') {
              b.style.setProperty('background-color', '#e9f4ec', 'important');
              b.style.setProperty('border-color', '#b3d8bf', 'important');
              b.style.setProperty('color', '#2f7d43', 'important');
            }
          });
        }
        colourButtons();
        new MutationObserver(colourButtons)
          .observe(window.parent.document.body, {childList: true, subtree: true});
        </script>""", height=0)
    _pool = load_pool()
    if _pool is None:
        st.warning("Sample pool not found — the case buttons are using fixed fallback examples. "
                   "(If you just deployed, reboot the app so it picks up sample_pool.csv.)")
    else:
        st.caption(f"Sample pool loaded: {len(_pool)} real applications "
                   f"({int((_pool['fraud_bool'] == 1).sum())} fraud, "
                   f"{int((_pool['fraud_bool'] == 0).sum())} legitimate). "
                   "The two case buttons draw a random real application with its true fraud outcome, "
                   "so after scoring you can compare the model decision with the ground truth.")

    with st.form("application"):
        subtabs = st.tabs(list(GROUPS))
        for stab, (_label, fields) in zip(subtabs, GROUPS.items()):
            with stab:
                cols = st.columns(2)
                for i, field in enumerate(fields):
                    col = cols[i % 2]
                    key = f"f_{field}"
                    if field in CATEGORIES:
                        col.selectbox(field, CATEGORIES[field], key=key)
                    elif field in BINARY:
                        col.selectbox(field, [0, 1], key=key)
                    elif isinstance(HIGH_RISK[field], int):
                        col.number_input(field, step=1, key=key)
                    else:
                        col.number_input(field, format="%.4f", key=key)
        submitted = st.form_submit_button("Score application", type="primary",
                                          use_container_width=True)

    if submitted:
        payload = {f: st.session_state[f"f_{f}"] for f in FIELDS}
        # only use the true label if the example wasn't edited
        true_label = (st.session_state.get("loaded_label")
                      if payload == st.session_state.get("loaded_payload") else None)
        try:
            with st.spinner("Scoring… (first call to a sleeping service can take about 50s)"):
                result = score_one(api_url, payload)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            st.stop()
        st.subheader("Result")
        render_result(result, payload, true_label)

with tab_batch:
    st.write("Upload a CSV with one application per row (columns = the 30 model features). "
             "Each row is scored through the deployed API.")
    MAX_ROWS = 50
    template = pd.DataFrame([HIGH_RISK, LOW_RISK])[FIELDS]
    st.download_button("Download a sample CSV template",
                       template.to_csv(index=False).encode(),
                       "sample_applications.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            st.stop()
        missing = [c for c in FIELDS if c not in df.columns]
        if missing:
            st.error("CSV is missing required columns: " + ", ".join(missing[:8])
                     + (" …" if len(missing) > 8 else ""))
        else:
            st.write(f"Loaded {len(df)} applications.")
            if len(df) > MAX_ROWS:
                st.warning(f"Only the first {MAX_ROWS} rows will be scored (free-tier API limit).")
                df = df.head(MAX_ROWS)
            if st.button(f"Score {len(df)} applications", type="primary"):
                rows, errors = [], 0
                prog = st.progress(0.0, text="Scoring…")
                for i, rec in enumerate(df[FIELDS].to_dict("records")):
                    try:
                        r = score_one(api_url, rec)
                        rows.append({"fraud_probability": round(r["fraud_probability"], 4),
                                     "risk_level": r["risk_level"],
                                     "action": r["deterministic_action"]})
                    except Exception:  # noqa: BLE001
                        rows.append({"fraud_probability": None, "risk_level": "error",
                                     "action": "error"})
                        errors += 1
                    prog.progress((i + 1) / len(df), text=f"Scoring… {i + 1}/{len(df)}")
                prog.empty()
                res = pd.concat([pd.DataFrame(rows), df.reset_index(drop=True)], axis=1)
                st.success(f"Scored {len(res)} applications"
                           + (f" ({errors} failed)" if errors else ""))
                s1, s2, s3 = st.columns(3)
                s1.metric("Block & escalate", int((res["action"] == "block_and_escalate").sum()))
                s2.metric("Manual review", int((res["action"] == "manual_review").sum()))
                s3.metric("Allow", int((res["action"] == "allow").sum()))
                st.dataframe(res, use_container_width=True, hide_index=True)
                st.download_button("Download results CSV", res.to_csv(index=False).encode(),
                                   "scored_applications.csv", "text/csv")
