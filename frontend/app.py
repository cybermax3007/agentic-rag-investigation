import requests
import streamlit as st


st.set_page_config(
    page_title="CaseFile AI",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Visual system: editorial / case-file aesthetic, not "AI dashboard".
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --paper: #f4f1ea;
        --paper-2: #ebe6dc;
        --ink: #171717;
        --muted: #6f6a61;
        --line: #c8c0b3;
        --accent: #7a2e2e;
        --verified: #355c4a;
        --unverified: #8b6b2c;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #e7e1d6;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: -0.02em;
        color: var(--ink);
    }

    h1 {
        font-size: 2.7rem !important;
        margin-bottom: 0.25rem !important;
    }

    h2 {
        font-size: 1.65rem !important;
        margin-top: 0.7rem !important;
    }

    p, li, label, .stMarkdown, .stTextInput, .stTextArea, .stSelectbox {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .masthead {
        border-top: 3px solid var(--ink);
        border-bottom: 1px solid var(--line);
        padding: 0.85rem 0 1.15rem 0;
        margin-bottom: 1.4rem;
    }

    .eyebrow {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.76rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .subtitle {
        color: var(--muted);
        max-width: 760px;
        font-size: 0.98rem;
        margin-top: 0.2rem;
    }

    .case-strip {
        display: flex;
        gap: 1.6rem;
        flex-wrap: wrap;
        padding: 0.8rem 0;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid var(--line);
    }

    .case-stat {
        min-width: 120px;
    }

    .case-stat .value {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.45rem;
        line-height: 1.1;
    }

    .case-stat .label {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.68rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    .evidence-card {
        background: rgba(255,255,255,0.28);
        border: 1px solid var(--line);
        border-left: 4px solid var(--ink);
        padding: 1rem 1.05rem;
        margin: 0.75rem 0;
    }

    .evidence-meta {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0.04em;
        margin-bottom: 0.55rem;
    }

    .status-verified {
        color: var(--verified);
        font-weight: 700;
    }

    .status-unverified {
        color: var(--unverified);
        font-weight: 700;
    }

    .status-other {
        color: var(--accent);
        font-weight: 700;
    }

    .note {
        background: #eee8de;
        border-left: 3px solid var(--accent);
        padding: 0.75rem 0.9rem;
        margin: 0.7rem 0;
        color: #3d3934;
    }

    .trace-step {
        border-top: 1px solid var(--line);
        padding-top: 0.9rem;
        margin-top: 1rem;
    }

    .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.78rem;
        color: var(--muted);
    }

    div[data-testid="stTabs"] button {
        font-size: 0.9rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.2rem;
        border-bottom: 1px solid var(--line);
    }

    .stButton > button {
        border-radius: 2px;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper);
        box-shadow: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        background: var(--accent);
        color: white;
    }

    .stTextInput input,
    .stTextArea textarea,
    div[data-baseweb="select"] > div {
        background: #fbf9f4 !important;
        border-radius: 2px !important;
    }

    details {
        border-radius: 0 !important;
    }

    hr {
        border-color: var(--line);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "initial_prediction" not in st.session_state:
    st.session_state.initial_prediction = None

if "investigation" not in st.session_state:
    st.session_state.investigation = None

if "fact_check" not in st.session_state:
    st.session_state.fact_check = None

if "final_verdict" not in st.session_state:
    st.session_state.final_verdict = ""


# ---------------------------------------------------------------------
# Sidebar / backend settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="eyebrow">CaseFile AI / Control Room</div>',
        unsafe_allow_html=True,
    )

    API_URL = st.text_input(
        "Backend URL",
        value="http://127.0.0.1:8000",
        help="FastAPI service used by the interface.",
    ).rstrip("/")

    st.markdown("---")


def api_get(path: str):
    response = requests.get(
        f"{API_URL}{path}",
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def api_post(
    path: str,
    payload: dict,
    timeout: int = 180,
):
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


try:
    summary = api_get("/case/summary")
except Exception as exc:
    st.error(
        "The backend is not reachable. Start it with "
        "`python -m uvicorn backend.main:app --reload`."
    )
    st.exception(exc)
    st.stop()


with st.sidebar:
    st.markdown("### Case index")
    st.caption("The Adventure of the Norwood Builder")
    st.markdown(
        f"""
        <div class="case-stat">
            <div class="value">{summary["evidence"]}</div>
            <div class="label">Evidence claims</div>
        </div>
        <br>
        <div class="case-stat">
            <div class="value">{summary["entities"]}</div>
            <div class="label">Canonical entities</div>
        </div>
        <br>
        <div class="case-stat">
            <div class="value">{summary["relationships"]}</div>
            <div class="label">Relationships</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="eyebrow">Investigation File / NORWOOD-01</div>
        <h1>CaseFile AI</h1>
        <div class="subtitle">
            Evidence-first investigation workspace for
            <em>The Adventure of the Norwood Builder</em>.
            Search the record, inspect relationships, test a theory,
            and challenge it before reaching a verdict.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="case-strip">
        <div class="case-stat">
            <div class="value">{summary["evidence"]}</div>
            <div class="label">Claims</div>
        </div>
        <div class="case-stat">
            <div class="value">{summary["entities"]}</div>
            <div class="label">Entities</div>
        </div>
        <div class="case-stat">
            <div class="value">{summary["relationships"]}</div>
            <div class="label">Relations</div>
        </div>
        <div class="case-stat">
            <div class="value">RRF</div>
            <div class="label">Hybrid retrieval</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tabs = st.tabs(
    [
        "Evidence",
        "Interrogation",
        "Graph",
        "Investigator",
        "Verdict",
    ]
)


def status_class(status: str) -> str:
    if status == "verified":
        return "status-verified"
    if status == "unverified":
        return "status-unverified"
    return "status-other"


# ---------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------
with tabs[0]:
    st.header("Evidence search")
    st.caption(
        "Hybrid lexical + semantic retrieval over grounded case evidence."
    )

    query = st.text_input(
        "Query the record",
        value="blood thumbprint McFarlane",
        key="evidence_search",
    )

    if st.button("Search record", key="search_evidence"):
        with st.spinner("Searching evidence..."):
            result = api_post(
                "/search",
                {
                    "query": query,
                    "top_k": 8,
                },
            )

        for item in result["results"]:
            css_class = status_class(item["status"])
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-meta">
                        {item["evidence_id"]} · {item["document_id"]} ·
                        <span class="{css_class}">
                            {item["status"].upper()}
                        </span>
                        · {item["evidence_type"].upper()}
                    </div>
                    <div>{item["claim"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Source excerpt"):
                st.write(item["source_excerpt"])
                st.caption(
                    f"Hybrid RRF score: {item['score']:.5f}"
                )


# ---------------------------------------------------------------------
# Grounded interrogation
# ---------------------------------------------------------------------
with tabs[1]:
    st.header("Interrogation")
    st.caption(
        "Ask about a person using retrieved evidence only. "
        "This is not free-form character roleplay."
    )

    candidate_names = [
        candidate["name"]
        for candidate in summary["candidates"]
    ]

    col1, col2 = st.columns([1, 2])

    with col1:
        candidate = st.selectbox(
            "Person",
            candidate_names,
        )

    with col2:
        interrogation_question = st.text_input(
            "Question",
            value="What evidence connects this person to the case?",
        )

    if st.button("Question the record", key="interrogate"):
        with st.spinner("Retrieving relevant testimony and evidence..."):
            result = api_post(
                "/interrogate",
                {
                    "candidate": candidate,
                    "question": interrogation_question,
                    "top_k": 6,
                },
            )

        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown(
            f"""
            <div class="note">{result["caveat"]}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="mono">Evidence: {", ".join(result["evidence_ids"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="mono">Documents: {", ".join(result["document_ids"])}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# Graph inspection
# ---------------------------------------------------------------------
with tabs[2]:
    st.header("Evidence graph")
    st.caption(
        "Inspect the evidence and relationship neighborhood around a person."
    )

    candidate_map = {
        candidate["name"]: candidate["entity_id"]
        for candidate in summary["candidates"]
    }

    graph_name = st.selectbox(
        "Person",
        list(candidate_map.keys()),
        key="graph_person",
    )

    if st.button("Inspect neighborhood", key="load_graph"):
        result = api_get(
            f"/graph/entity/{candidate_map[graph_name]}"
        )

        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Evidence")
            for item in result["evidence"]:
                css_class = status_class(item["status"])
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-meta">
                            {item["evidence_id"]} · {item["document_id"]} ·
                            <span class="{css_class}">
                                {item["status"].upper()}
                            </span>
                        </div>
                        <div>{item["claim"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:
            st.subheader("Relationships")
            for relation in result["relationships"]:
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-meta">
                            {relation["relationship_id"]} ·
                            {relation["status"].upper()}
                        </div>
                        <div>
                            <strong>{relation["source_name"]}</strong>
                            → {relation["relation"].replace("_", " ")} →
                            <strong>{relation["target_name"]}</strong>
                        </div>
                        <div class="mono">
                            {relation["document_id"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("Graph data"):
            st.json(result["subgraph"])


# ---------------------------------------------------------------------
# Investigator
# ---------------------------------------------------------------------
with tabs[3]:
    st.header("Investigator")
    st.caption(
        "Lock a theory first. The agent then retrieves evidence, "
        "tests sufficiency, and reformulates its query when necessary."
    )

    if st.session_state.initial_prediction is None:
        prediction = st.text_area(
            "Initial theory",
            placeholder=(
                "Record your theory before seeing the agent's conclusion."
            ),
            height=120,
        )

        if st.button(
            "Lock theory",
            key="lock_prediction",
        ):
            if prediction.strip():
                st.session_state.initial_prediction = (
                    prediction.strip()
                )
                st.rerun()
            else:
                st.warning(
                    "Enter an initial theory first."
                )
    else:
        st.markdown("**Initial theory — locked**")
        st.markdown(
            f"""
            <div class="note">
                {st.session_state.initial_prediction}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    investigation_question = st.text_input(
        "Investigation question",
        value=(
            "Does the available evidence establish that "
            "John Hector McFarlane murdered Jonas Oldacre?"
        ),
    )

    if st.button(
        "Run investigator",
        disabled=(
            st.session_state.initial_prediction is None
        ),
        key="run_investigator",
    ):
        with st.spinner(
            "Reviewing evidence and testing the working theory..."
        ):
            investigation = api_post(
                "/investigate",
                {
                    "question": investigation_question,
                    "initial_theory": (
                        st.session_state.initial_prediction
                    ),
                    "max_iterations": 3,
                    "top_k": 8,
                },
                timeout=240,
            )

        st.session_state.investigation = investigation
        st.session_state.fact_check = None

    investigation = st.session_state.investigation

    if investigation:
        st.markdown("---")
        st.subheader("Investigator finding")

        a, b = st.columns([2, 1])
        with a:
            st.markdown(
                f"### {investigation['verdict']}"
            )
            st.write(
                investigation["reasoning_summary"]
            )
        with b:
            st.markdown(
                f"""
                <div class="case-stat">
                    <div class="value">
                        {investigation["confidence"]:.0%}
                    </div>
                    <div class="label">Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Final working theory**")
        st.info(investigation["final_theory"])

        st.markdown(
            f'<div class="mono">Evidence: {", ".join(investigation["supporting_evidence_ids"] + investigation["contradicting_evidence_ids"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="mono">Documents: {", ".join(investigation["cited_document_ids"])}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Investigator trace"):
            for step in investigation["trace"]:
                st.markdown(
                    f"""
                    <div class="trace-step">
                        <div class="eyebrow">Iteration {step["iteration"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"**Query:** {step['query']}")
                st.write(f"**Theory:** {step['theory']}")
                st.write(
                    f"**Sufficient:** {step['sufficient']}"
                )
                st.write(f"**Assessment:** {step['reason']}")

                if step["next_query"]:
                    st.write(
                        f"**Reformulated query:** "
                        f"{step['next_query']}"
                    )

                st.caption(
                    "Evidence: "
                    + ", ".join(step["evidence_ids"])
                )

        if st.button(
            "Run adversarial review",
            key="run_fact_checker",
        ):
            with st.spinner(
                "Searching for counter-evidence..."
            ):
                fact_check = api_post(
                    "/fact-check",
                    {
                        "investigation": investigation,
                        "top_k": 10,
                    },
                    timeout=180,
                )

            st.session_state.fact_check = fact_check

    if st.session_state.fact_check:
        fact_check = st.session_state.fact_check

        st.markdown("---")
        st.subheader("Adversarial review")
        st.write(
            fact_check["overall_assessment"]
        )

        c1, c2 = st.columns(2)

        with c1:
            if fact_check["weaknesses"]:
                st.markdown("**Weaknesses**")
                for item in fact_check["weaknesses"]:
                    st.write(f"— {item}")

        with c2:
            if fact_check["alternative_explanations"]:
                st.markdown(
                    "**Alternative explanations**"
                )
                for item in fact_check[
                    "alternative_explanations"
                ]:
                    st.write(f"— {item}")

        if fact_check["conflicting_timeline_points"]:
            st.markdown("**Timeline conflicts**")
            for item in fact_check[
                "conflicting_timeline_points"
            ]:
                st.write(f"— {item}")


# ---------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------
with tabs[4]:
    st.header("Final verdict")
    st.caption(
        "Compare your first impression with the investigator "
        "and adversarial review before recording a conclusion."
    )

    if st.session_state.initial_prediction:
        st.markdown("**Initial theory**")
        st.write(st.session_state.initial_prediction)

    if st.session_state.investigation:
        st.markdown("**Investigator finding**")
        st.write(
            st.session_state.investigation[
                "reasoning_summary"
            ]
        )

    if st.session_state.fact_check:
        st.markdown("**Adversarial review**")
        st.write(
            st.session_state.fact_check[
                "overall_assessment"
            ]
        )

    st.markdown("---")

    final_verdict = st.text_area(
        "Your verdict",
        value=st.session_state.final_verdict,
        placeholder=(
            "State your conclusion and cite EV_### / DOC_###."
        ),
        height=180,
    )

    if st.button(
        "Record verdict",
        key="save_verdict",
    ):
        st.session_state.final_verdict = (
            final_verdict.strip()
        )
        st.success(
            "Verdict recorded for this session."
        )
