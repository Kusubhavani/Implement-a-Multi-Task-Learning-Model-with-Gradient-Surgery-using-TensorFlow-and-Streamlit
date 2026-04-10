import streamlit as st
import pandas as pd

st.title("MTL Gradient Surgery Dashboard")

# Gradient conflict
df = pd.read_csv("results/gradient_conflict.csv")

st.subheader("Gradient Conflict Monitor")

st.markdown('<div data-testid="gradient-conflict-monitor">', unsafe_allow_html=True)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(df["cosine_similarity"], label="Cosine Similarity")

# Highlight conflicts (negative values)
conflicts = df["cosine_similarity"] < 0
ax.scatter(
    df.index[conflicts],
    df["cosine_similarity"][conflicts],
    color='red',
    s=10,
    label="Conflicts"
)

ax.legend()
ax.set_title("Gradient Cosine Similarity Over Time")
ax.set_xlabel("Training Step")
ax.set_ylabel("Cosine Similarity")

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Metrics
st.subheader("Model Performance")

baseline = pd.read_csv("results/baseline_metrics.csv")
pcgrad = pd.read_csv("results/pcgrad_metrics.csv")

st.line_chart({
    "Baseline Loss A": baseline["loss_a"],
    "PCGrad Loss A": pcgrad["loss_a"],
    "Baseline Loss B": baseline["loss_b"],
    "PCGrad Loss B": pcgrad["loss_b"]
})