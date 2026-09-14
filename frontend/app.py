import html
import os

import requests
import streamlit as st


st.set_page_config(
    page_title="CaseFile AI",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Visual system: editorial / case-file aesthetic.
# High-contrast text is explicitly set because browser/Streamlit theme
# inheritance can otherwise make inactive tabs and labels too pale.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --paper: #f3efe6;
        --paper-2: #e7e0d3;
        --ink: #171613;
        --body: #302d28;
        --muted: #5d574f;
        --line: #bdb4a5;
        --accent: #71332e;
        --verified: #315847;
        --unverified: #7a5b20;
    }

    .stApp {
        background: var(--paper);
        color: var(--body);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: var(--paper-2);
        border-right: 1px solid var(--line);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: -0.02em;
        color: var(--ink) !important;
    }

    h1 {
        font-size: 2.7rem !important;
        margin-bottom: 0.25rem !important;
    }

    h2 {
        font-size: 1.65rem !important;
        margin-top: 0.7rem !important;
    }

    p, li, label, .stMarkdown {
        color: var(--body);
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
        max-width: 790px;
        font-size: 0.99rem;
        line-height: 1.55;
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
        color: var(--ink);
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
        background: rgba(255,255,255,0.35);
        border: 1px solid var(--line);
        border-left: 4px solid var(--ink);
        padding: 1rem 1.05rem;
        margin: 0.75rem 0;
        color: var(--body);
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
        background: #e9e1d4;
        border-left: 3px solid var(--accent);
        padding: 0.75rem 0.9rem;
        margin: 0.7rem 0;
        color: #332f2a;
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

    /* Fix low-contrast tab text */
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button p,
    div[data-baseweb="tab"] p {
        color: #443f38 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--accent) !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.2rem;
        border-bottom: 1px solid var(--line);
    }

    /* Fix form labels and input text */
    [data-testid="stTextInput"] label p,
    [data-testid="stTextArea"] label p,
    [data-testid="stSelectbox"] label p,
    [data-testid="stMultiSelect"] label p {
        color: #332f2a !important;
        font-weight: 600 !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] input {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        caret-color: var(--ink) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #7a7369 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] > div {
        background: #fbf8f1 !important;
        border-radius: 2px !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="select"] span {
        color: var(--ink) !important;
    }

    .stButton > button {
        border-radius: 2px;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: #f8f4ec !important;
        box-shadow: none;
        font-weight: 600;
    }

    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #f8f4ec !important;
        -webkit-text-fill-color: #f8f4ec !important;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        background: var(--accent);
    }

    [data-testid="stAlert"] {
        border-radius: 2px;
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


if "initial_prediction" not in st.session_state:
    st.session_state.initial_prediction = None

if "investigation" not in st.session_state:
    st.session_state.investigation = None

if "fact_check" not in st.session_state:
    st.session_state.fact_check = None

if "faithfulness" not in st.session_state:
    st.session_state.faithfulness = None

if "verdict_evaluation" not in st.session_state:
    st.session_state.verdict_evaluation = None


default_backend = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


with st.sidebar:
    st.markdown(
        '<div class="eyebrow">CaseFile AI / Control Room</div>',
        unsafe_allow_html=True,
    )

    API_URL = st.text_input(
        "Backend URL",
        value=default_backend,
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


def escape(value) -> str:
    return html.escape(str(value))


def graphviz_dot(subgraph: dict) -> str:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", subgraph.get("links", []))

    lines = [
        "digraph EvidenceGraph {",
        'graph [rankdir="LR", bgcolor="#f3efe6", pad="0.25", nodesep="0.35", ranksep="0.55"];',
        'node [fontname="Helvetica", fontsize="10", style="filled", color="#6f675c", fontcolor="#171613"];',
        'edge [fontname="Helvetica", fontsize="8", color="#8f877b", fontcolor="#5d574f"];',
    ]

    for node in nodes:
        node_id = str(node.get("id", ""))
        label = str(node.get("label") or node_id)
        node_type = node.get("node_type", "other")

        if len(label) > 42:
            label = label[:39] + "..."

        attrs = {
            "entity": ('ellipse', '#ded6c8'),
            "evidence": ('box', '#f8f4ec'),
            "document": ('folder', '#e4ddd1'),
            "hypothesis": ('diamond', '#ead7d3'),
        }
        shape, fill = attrs.get(
            node_type,
            ('box', '#f8f4ec'),
        )

        safe_id = node_id.replace('"', '\\"')
        safe_label = label.replace('"', '\\"')

        lines.append(
            f'"{safe_id}" [label="{safe_label}", shape="{shape}", fillcolor="{fill}"];'
        )

    for edge in edges:
        source = str(edge.get("source", "")).replace('"', '\\"')
        target = str(edge.get("target", "")).replace('"', '\\"')

        label = (
            edge.get("relation")
            or edge.get("edge_type")
            or ""
        )
        label = str(label).replace("_", " ")
        label = label.replace('"', '\\"')

        lines.append(
            f'"{source}" -> "{target}" [label="{label}"];'
        )

    lines.append("}")
    return "\n".join(lines)


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
                        {escape(item["evidence_id"])} ·
                        {escape(item["document_id"])} ·
                        <span class="{css_class}">
                            {escape(item["status"].upper())}
                        </span>
                        · {escape(item["evidence_type"].upper())}
                    </div>
                    <div>{escape(item["claim"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Source excerpt"):
                st.write(item["source_excerpt"])
                st.caption(
                    f"Retriever: {item['source']} · score: {item['score']:.5f}"
                )


with tabs[1]:
    st.header("Interrogation")
    st.caption(
        "Ask about any person in the case using retrieved evidence only. "
        "This is grounded questioning, not invented character roleplay."
    )

    people = summary.get(
        "people",
        summary["candidates"],
    )

    person_names = [
        person["name"]
        for person in people
    ]

    col1, col2 = st.columns([1, 2])

    with col1:
        candidate = st.selectbox(
            "Person",
            person_names,
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

        st.markdown("### Grounded answer")
        st.write(result["answer"])

        st.markdown(
            f"""
            <div class="note">{escape(result["caveat"])}</div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Evidence: "
            + ", ".join(result["evidence_ids"])
        )
        st.caption(
            "Documents: "
            + ", ".join(result["document_ids"])
        )


with tabs[2]:
    st.header("Evidence graph")
    st.caption(
        "Inspect the local evidence network around any person in the case."
    )

    candidate_map = {
        person["name"]: person["entity_id"]
        for person in summary.get(
            "people",
            summary["candidates"],
        )
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

        st.subheader("Graph view")
        st.graphviz_chart(
            graphviz_dot(result["subgraph"]),
            use_container_width=True,
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
                            {escape(item["evidence_id"])} ·
                            {escape(item["document_id"])} ·
                            <span class="{css_class}">
                                {escape(item["status"].upper())}
                            </span>
                        </div>
                        <div>{escape(item["claim"])}</div>
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
                            {escape(relation["relationship_id"])} ·
                            {escape(relation["status"].upper())}
                        </div>
                        <div>
                            <strong>{escape(relation["source_name"])}</strong>
                            → {escape(relation["relation"].replace("_", " "))} →
                            <strong>{escape(relation["target_name"])}</strong>
                        </div>
                        <div class="mono">
                            {escape(relation["document_id"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


with tabs[3]:
    st.header("Investigator")
    st.caption(
        "Lock your own prediction first. The Investigator then retrieves "
        "evidence, tests sufficiency, and reformulates its query when needed."
    )

    if st.session_state.initial_prediction is None:
        prediction = st.text_area(
            "Your initial prediction",
            placeholder=(
                "Record your theory before seeing the Investigator result."
            ),
            height=120,
        )

        if st.button(
            "Lock prediction",
            key="lock_prediction",
        ):
            if prediction.strip():
                st.session_state.initial_prediction = (
                    prediction.strip()
                )
                st.rerun()
            else:
                st.warning("Enter an initial prediction first.")
    else:
        st.markdown("**Initial prediction — locked**")
        st.markdown(
            f"""
            <div class="note">
                {escape(st.session_state.initial_prediction)}
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
        st.session_state.faithfulness = None

    investigation = st.session_state.investigation

    if investigation:
        st.markdown("---")
        st.subheader("Investigator finding")

        a, b = st.columns([2, 1])

        with a:
            st.markdown(
                f"### {escape(investigation['verdict'])}"
            )
            st.write(
                investigation["reasoning_summary"]
            )

            if investigation.get(
                "needs_more_evidence"
            ):
                st.warning(
                    "Retry limit reached before the Investigator "
                    "considered the evidence fully sufficient."
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
            st.caption(
                "Termination: "
                + investigation.get(
                    "termination_reason",
                    "unknown",
                )
            )

        st.markdown("**Final working theory**")
        st.info(investigation["final_theory"])

        st.caption(
            "Evidence: "
            + ", ".join(
                investigation["supporting_evidence_ids"]
                + investigation["contradicting_evidence_ids"]
            )
        )
        st.caption(
            "Documents: "
            + ", ".join(
                investigation["cited_document_ids"]
            )
        )

        with st.expander("Investigator execution trace"):
            for step in investigation["trace"]:
                phase_label = (
                    "Coverage sweep + verified evidence safety pass"
                    if step.get("phase") == "coverage_sweep"
                    else f"Iteration {step['iteration']}"
                )
                st.markdown(
                    f"#### {phase_label}"
                )
                st.write(f"**Query:** {step['query']}")
                st.write(f"**Theory:** {step['theory']}")
                st.write(
                    f"**Sufficient:** {step['sufficient']}"
                )
                st.write(
                    f"**Assessment:** {step['reason']}"
                )

                if step["next_query"]:
                    st.write(
                        "**Reformulated query:** "
                        + step["next_query"]
                    )

                st.caption(
                    "Evidence: "
                    + ", ".join(step["evidence_ids"])
                )

        action_left, action_right = st.columns(2)

        with action_left:
            if st.button(
                "Audit answer quality",
                key="run_faithfulness",
            ):
                with st.spinner(
                    "Checking citation grounding and evidence coverage..."
                ):
                    faithfulness = api_post(
                        "/faithfulness",
                        investigation,
                        timeout=180,
                    )

                st.session_state.faithfulness = faithfulness

        with action_right:
            if st.button(
                "Run adversarial review",
                key="run_fact_checker",
            ):
                with st.spinner(
                    "Searching independently for counter-evidence..."
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

    if st.session_state.faithfulness:
        faithfulness = st.session_state.faithfulness

        st.markdown("---")
        st.subheader("Answer quality audit")
        st.caption(
            "Grounding asks whether the answer is supported by what it cited. "
            "Coverage separately asks whether decisive evidence was missed."
        )

        left, middle, right = st.columns([1, 1, 2])

        left.metric(
            "Citation grounding",
            f"{faithfulness['support_ratio']:.0%}",
        )

        coverage_label = faithfulness["coverage_status"].upper()
        middle.metric(
            "Coverage",
            coverage_label,
        )

        right.write(faithfulness["overall_assessment"])
        right.caption(faithfulness["coverage_assessment"])

        if faithfulness["critical_omissions"]:
            st.error(
                "Critical omitted evidence was found. "
                "The answer may be grounded but incomplete."
            )

            for item in faithfulness["critical_omissions"]:
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-meta">
                            {escape(item["evidence_id"])} ·
                            {escape(item["document_id"])} ·
                            {escape(item["status"].upper())}
                        </div>
                        <div>{escape(item["claim"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        elif faithfulness["coverage_status"] == "warning":
            st.warning(
                "The audit found useful omitted evidence, "
                "but not a validated conclusion-changing omission."
            )
        else:
            st.success(
                "No critical omission was found in the independent coverage sweep."
            )

        with st.expander("Sentence-level grounding audit"):
            for item in faithfulness["assessments"]:
                status = "SUPPORTED" if item["supported"] else "NOT SUPPORTED"
                st.markdown(
                    f"**Sentence {item['sentence_index'] + 1} — {status}**"
                )
                st.write(item["reason"])
                if item["evidence_ids"]:
                    st.caption(
                        "Evidence: " + ", ".join(item["evidence_ids"])
                    )

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


with tabs[4]:
    st.header("Final verdict")
    st.caption(
        "Submit a candidate, your conclusion, and the evidence IDs you rely on. "
        "The backend checks citation validity and evaluates whether the cited "
        "record actually supports your conclusion."
    )

    if st.session_state.initial_prediction:
        with st.expander("Initial prediction"):
            st.write(st.session_state.initial_prediction)

    if st.session_state.investigation:
        with st.expander("Investigator finding"):
            st.write(
                st.session_state.investigation[
                    "reasoning_summary"
                ]
            )

    if st.session_state.fact_check:
        with st.expander("Adversarial review"):
            st.write(
                st.session_state.fact_check[
                    "overall_assessment"
                ]
            )

    candidate_names = [
        item["name"]
        for item in summary["candidates"]
    ]

    final_candidate = st.selectbox(
        "Final candidate / subject",
        candidate_names,
        key="final_candidate",
    )

    final_verdict = st.text_area(
        "Your conclusion",
        placeholder=(
            "State your final conclusion and explain why the cited evidence "
            "supports it."
        ),
        height=180,
    )

    default_ids = ""

    if st.session_state.investigation:
        default_ids = ", ".join(
            st.session_state.investigation[
                "supporting_evidence_ids"
            ][:6]
        )

    evidence_id_text = st.text_input(
        "Supporting evidence IDs",
        value=default_ids,
        placeholder="EV_012, EV_019, EV_083",
        help="Comma-separated canonical evidence IDs.",
    )

    if st.button(
        "Submit verdict",
        key="submit_verdict",
    ):
        evidence_ids = [
            item.strip().upper()
            for item in evidence_id_text.split(",")
            if item.strip()
        ]

        if not final_verdict.strip():
            st.warning("Write your conclusion first.")
        elif not evidence_ids:
            st.warning("Cite at least one EV_### evidence ID.")
        else:
            with st.spinner("Checking grounding and support..."):
                evaluation = api_post(
                    "/submit-verdict",
                    {
                        "candidate": final_candidate,
                        "conclusion": final_verdict.strip(),
                        "evidence_ids": evidence_ids,
                    },
                    timeout=180,
                )

            st.session_state.verdict_evaluation = evaluation

    if st.session_state.verdict_evaluation:
        evaluation = st.session_state.verdict_evaluation

        st.markdown("---")
        st.subheader("Verdict evaluation")

        c1, c2 = st.columns(2)

        c1.metric(
            "Support",
            evaluation["support_level"].replace("_", " ").title(),
        )
        c2.metric(
            "Citation validity",
            f"{evaluation['citation_validity']:.0%}",
        )

        st.write(evaluation["assessment"])

        st.caption(
            f"Verified cited evidence: "
            f"{evaluation['verified_evidence_count']} · "
            f"Unverified: {evaluation['unverified_evidence_count']} · "
            f"Other: {evaluation['other_status_count']}"
        )

        if evaluation["unsupported_evidence_ids"]:
            st.warning(
                "Unknown evidence IDs: "
                + ", ".join(
                    evaluation["unsupported_evidence_ids"]
                )
            )

        if evaluation["missing_or_conflicting_points"]:
            st.markdown("**Remaining gaps / conflicts**")
            for item in evaluation[
                "missing_or_conflicting_points"
            ]:
                st.write(f"— {item}")
