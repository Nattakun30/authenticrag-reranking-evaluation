import json

import pandas as pd

import plotly.express as px

import streamlit as st
 
st.set_page_config(

    page_title="RAG Retrieval Evaluation",

    page_icon="📊",

    layout="wide"

)
 
st.title("RAG Retrieval Evaluation Dashboard")

st.caption("BM25 + BGE-M3 + RRF vs BGE Reranker v2 M3")
 
# Load evaluation results

with open("evaluation_results.json", "r", encoding="utf-8") as f:

    results = json.load(f)
 
df = pd.DataFrame(results)
 
metric_cols = [

    "recall_at_5",

    "mrr",

    "ndcg_at_5",

    "latency_sec"

]
 
summary = df.groupby("method")[metric_cols].mean()
 
baseline = summary.loc["Baseline"]

improved = summary.loc["Improved"]
 
# Summary cards

st.subheader("Overall Performance")
 
col1, col2, col3, col4 = st.columns(4)
 
col1.metric(

    "Recall@5",

    f"{improved['recall_at_5']:.2%}",

    f"{(improved['recall_at_5'] - baseline['recall_at_5']):+.2%}"

)
 
col2.metric(

    "MRR",

    f"{improved['mrr']:.4f}",

    f"{(improved['mrr'] - baseline['mrr']):+.4f}"

)
 
col3.metric(

    "nDCG@5",

    f"{improved['ndcg_at_5']:.4f}",

    f"{(improved['ndcg_at_5'] - baseline['ndcg_at_5']):+.4f}"

)
 
col4.metric(

    "Average Latency",

    f"{improved['latency_sec']:.2f}s",

    f"{(improved['latency_sec'] - baseline['latency_sec']):+.2f}s",

    delta_color="inverse"

)
 
# Retrieval quality comparison

st.subheader("Retrieval Quality Comparison")
 
quality = (

    summary[["recall_at_5", "mrr", "ndcg_at_5"]]

    .reset_index()

    .melt(

        id_vars="method",

        var_name="Metric",

        value_name="Score"

    )

)
 
fig_quality = px.bar(

    quality,

    x="Metric",

    y="Score",

    color="method",

    barmode="group",

    text_auto=".3f",

    title="Baseline vs Improved"

)
 
fig_quality.update_yaxes(range=[0, 1.1])

st.plotly_chart(fig_quality, use_container_width=True)
 
# Latency comparison

st.subheader("Latency Comparison")
 
latency = summary.reset_index()
 
fig_latency = px.bar(

    latency,

    x="method",

    y="latency_sec",

    color="method",

    text_auto=".2f",

    title="Average Retrieval Latency (seconds)"

)
 
st.plotly_chart(fig_latency, use_container_width=True)
 
# Question-level analysis

st.subheader("Question-Level Analysis")
 
with open("evaluation_dataset.json", "r", encoding="utf-8") as f:

    dataset = json.load(f)
 
question_map = {

    item["question_id"]: item["question"]

    for item in dataset

}
 
question_ids = df["question_id"].unique().tolist()
 
selected_question = st.selectbox(

    "Select Question",

    question_ids,

    format_func=lambda q: f"{q}: {question_map.get(q, q)}"

)
 
question_df = df[df["question_id"] == selected_question]
 
left, right = st.columns(2)
 
for column, method in [

    (left, "Baseline"),

    (right, "Improved")

]:

    row = question_df[question_df["method"] == method].iloc[0]
 
    with column:

        st.markdown(f"### {method}")

        st.write(f"**Recall@5:** {row['recall_at_5']:.4f}")

        st.write(f"**MRR:** {row['mrr']:.4f}")

        st.write(f"**nDCG@5:** {row['ndcg_at_5']:.4f}")

        st.write(f"**Latency:** {row['latency_sec']:.4f} seconds")
 
        relevant = set(

            next(

                item["relevant_docs"]

                for item in dataset

                if item["question_id"] == selected_question

            )

        )
 
        for rank, doc_id in enumerate(row["retrieved_docs"], 1):

            status = "Relevant" if doc_id in relevant else "Other"

            st.write(f"{rank}. {doc_id} — {status}")
 
st.divider()
 
st.info(

    "Evaluation uses 4 questions and manually labeled relevant documents. "

    "Results measure retrieval quality, not final LLM answer quality. "

    "The experiment uses a local non-contextual hybrid retrieval baseline."

)
# ==========================================
# Download PDF Evaluation Report
# ==========================================
 
import io
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
 
 
def generate_pdf_report(df, dataset):
    buffer = io.BytesIO()
 
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=40,
        bottomMargin=40,
    )
 
    styles = getSampleStyleSheet()
    styles["Title"].alignment = TA_CENTER
 
    story = []
 
    story.append(
        Paragraph("RAG Retrieval Evaluation Report", styles["Title"])
    )
    story.append(Spacer(1, 15))
 
    story.append(
        Paragraph(
            "Baseline: BM25 + BGE-M3 + RRF<br/>"
            "Improved: Baseline + BGE Reranker v2 M3",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))
 
    # Overall results
    story.append(Paragraph("1. Overall Performance", styles["Heading2"]))
 
    metrics = ["recall_at_5", "mrr", "ndcg_at_5", "latency_sec"]
 
    averages = df.groupby("method")[metrics].mean()
 
    table_data = [
        ["Metric", "Baseline", "Improved"]
    ]
 
    for metric in metrics:
        table_data.append([
            metric,
            f"{averages.loc['Baseline', metric]:.4f}",
            f"{averages.loc['Improved', metric]:.4f}",
        ])
 
    table = Table(table_data, colWidths=[180, 140, 140])
 
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
    ]))
 
    story.append(table)
    story.append(Spacer(1, 20))
 
    # Question-level results
    story.append(Paragraph("2. Question-Level Results", styles["Heading2"]))
 
    for item in dataset:
        qid = item["question_id"]
        relevant = set(item["relevant_docs"])
 
        story.append(Paragraph(qid, styles["Heading3"]))
 
        rows = df[df["question_id"] == qid]
 
        for _, row in rows.iterrows():
            retrieved = row["retrieved_docs"]
 
            doc_text = ", ".join(
                f"{doc_id}{'*' if doc_id in relevant else ''}"
                for doc_id in retrieved
            )
 
            story.append(Paragraph(
                f"<b>{row['method']}</b>: {doc_text}",
                styles["Normal"]
            ))
 
            story.append(Paragraph(
                f"MRR: {row['mrr']:.4f} | "
                f"nDCG@5: {row['ndcg_at_5']:.4f} | "
                f"Latency: {row['latency_sec']:.4f}s",
                styles["Normal"]
            ))
 
            story.append(Spacer(1, 7))
 
    story.append(
        Paragraph(
            "* indicates a manually labeled relevant document.",
            styles["Italic"],
        )
    )
 
    story.append(Spacer(1, 15))
 
    # Findings
    story.append(Paragraph("3. Key Findings", styles["Heading2"]))
 
    story.append(Paragraph(
        "Cross-Encoder reranking improved the ranking of relevant "
        "documents in this four-question evaluation. "
        "Recall@5 remained unchanged, while MRR and nDCG@5 improved. "
        "However, the improved method required additional processing time.",
        styles["Normal"],
    ))
 
    story.append(Spacer(1, 15))
 
    # Limitations
    story.append(Paragraph("4. Limitations", styles["Heading2"]))
 
    story.append(Paragraph(
        "This experiment used four questions with manually labeled "
        "relevant documents and a local non-contextual hybrid retrieval "
        "baseline. The results do not establish improvement in final "
        "LLM-generated answer quality or performance of the original "
        "contextual RAG pipeline.",
        styles["Normal"],
    ))
 
    document.build(story)
 
    buffer.seek(0)
    return buffer.getvalue()
 
 
st.divider()
st.subheader("Download Evaluation Report")
 
pdf_bytes = generate_pdf_report(df, dataset)
 
st.download_button(
    label="Download PDF Report",
    data=pdf_bytes,
    file_name="RAG_Retrieval_Evaluation_Report.pdf",
    mime="application/pdf",
    use_container_width=True,
)