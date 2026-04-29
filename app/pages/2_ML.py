import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("ML")
st.write("This pages provides an overview on event- and session-level clustering experiments, results, and performance indicators. Along with some complementary analyses.")
from pathlib import Path
path = Path(__file__).parent / 'figures'

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Frequential analysis",
    "Event-Level clustering",
    "Session-Level clustering",
    "Cluster Analysis K-Means",
    "Cluster Analysis Word2Vec"
])


with tab1:
    st.markdown("## Frequential Analysis")

    st.info(
        "The click distribution is highly skewed: a small number of products receive most clicks, "
        "while most products receive very few."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Zipf R²", "0.923")
    col2.metric("Alpha", "0.95")
    col3.metric("50% of click volume", "48 items")
    col4.metric("90% sessions", "< 30 clicks")

    st.markdown("### Main conclusions")

    st.markdown("""
    - Product clicks approximately follow a **Zipf-like distribution**.
    - The fitted Zipf model explains the ranking trend well, with **R² ≈ 0.92**.
    - The tail of the distribution deviates from the Zipf estimate, meaning rare products are less stable/noisier.
    - Around **48 products account for 50% of all clicks**, showing strong concentration.
    - Session lengths are short: **90% of sessions contain fewer than 30 clicks**.
    """)

    st.markdown("### Visual analysis")

    images = {
        "Zipf fit — original scale": "zipf's curve.png",
        "Zipf fit — log-log scale": "zipf's fit in log space.png",
        "Residuals vs rank": "residuals_vs_rank.png",
        "Cumulative click probability": "cumulative_clicks.png",
        "Session length distribution": "probability + cumulative distribution of session lengths.png",
    }

    selected = st.radio(
        "Select figure",
        list(images.keys()),
        horizontal=True
    )

    st.image(path.joinpath(images[selected]), use_container_width=True)

    with st.expander("Summary"):
        st.markdown("""
        The frequency distribution confirms that user attention is concentrated on a limited set of products.
        The Zipf model fits the overall trend well, especially in the middle ranks.

        However, the residual plot shows stronger deviations at the tail, where very low-frequency products
        are harder to estimate accurately. This suggests that rare products contribute noise and may need
        filtering, grouping, or separate treatment.

        The session length distribution shows that most sessions are short. Since **90% of sessions are below
        30 clicks**, session-level models should account for short interaction histories.
        """)

with tab2:
    st.markdown("## Event-Level Clustering")

    st.success(
        "Best configuration: **Experiment 2 — no temporal features**, "
        "**K-Means**, **K = 8**, **15 PCA components**."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best model", "K-Means")
    with col2:
        st.metric("Best K", "= 8")
    with col3:
        st.metric("Best PCA", "= 15")

    st.markdown("### Main Result")

    st.markdown("""
    K-Means consistently produced the best clustering quality.  
    Removing temporal features gave the cleanest structure, with fewer and better-separated clusters.

    Temporal features made the clustering worse:
    - **Experiment 0:** day encoded using `sin` + `cos` + OHE month
    - **Experiment 1:** day encoded linearly, month exluded
    - **Experiment 2:** temporal features excluded completly
    """)

    st.markdown("### Visual comparison")

    images = {
        "Experiment 2 (best) — No temporal features": "output2.png",
        "Experiment 1 — Linear day encoding": "output1.png",
        "Experiment 0 — Sin/Cos day encoding + month": "output.png",
    }

    selected = st.radio(
        "Select experiment",
        list(images.keys()),
        horizontal=True
    )

    st.image(path.joinpath(images[selected]), use_container_width=True)

    with st.expander("Model-level conclusions"):
        st.markdown("""
        ### K-Means
        - Best overall model.
        - Highest silhouette scores, around **0.30–0.36**.
        - Best performance around **K = 8** in Experiment 2.
        - Lower Davies–Bouldin values indicate more compact clusters.
        - High Calinski–Harabasz scores support stronger cluster structure.

        ### Bisecting K-Means
        - Reasonable for smaller values of K.
        - Generally worse and less stable than standard K-Means.
        - Becomes less competitive when K increases.

        ### GMM
        - Poor performance across experiments.
        - Low or negative silhouette scores.
        - High Davies–Bouldin values suggest overlapping clusters.
        - Soft assignment likely does not fit this feature space well.
        """)

    with st.expander("Summary"):
        st.markdown("""
        The event-level clustering results suggest that temporal variables do not add useful structure.
        Both cyclic encoding and linear scaling of the day variable reduce clustering quality.
        """)

with tab3:
    st.markdown("## Session-Level Clustering")

    st.success("Strongest clustering metrics were observed for **Word2Vec / normalized context pooling**. ")
    st.markdown("### Main conclusions")

    st.markdown("""
    - **TF-IDF (normalized) + K-Means**: poor clustering metrics, no meaningful structure at interpretable number of clusters.
    - **Latent Dirichlet Allocation (LDA)**: 
        - Top number of topics K = 4
        - Model quality plateaus quickly and for larger K improvements are modest
    - **Embedding-based clustering**:
        - Good separation
        - More consistent structure
        - Item-click 32 dimensional embeddings form small well separated clusters in space.
    """)

    st.markdown("### LDA")
    lda_logs = pd.DataFrame({
        "K": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "perplexity": [4.991, 4.919, 4.952, 4.930, 4.947, 4.971, 4.967, 4.967, 4.997, 5.000, 5.062, 5.076],
        "log_likelihood": [-825953, -813965, -819451, -815740, -818550, -822538, -821880, -821914, -826799, -827338, -837572, -839868],
    
    })
    best = lda_logs.loc[lda_logs["perplexity"].idxmin()]
    st.info(f"Best LDA configuration: **K = {int(best['K'])}**, "    f"perplexity = **{best['perplexity']:.3f}**, "    f"log-likelihood = **{best['log_likelihood']:.0f}**")
    col1, col2 = st.columns(2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
    with col1:    
        ax[0].plot(lda_logs["K"], lda_logs["perplexity"], marker="o")
        ax[0].axvline(best["K"], linestyle="--") 
        ax[0].set_title("Perplexity by Number of Topics")
        ax[0].set_xlabel("K")
        ax[0].set_ylabel("Perplexity") 
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(lda_logs["K"], -lda_logs["log_likelihood"], marker="o")
        ax[1].axvline(best["K"], linestyle="--")
        ax[1].set_title("-Log-Likelihood by Number of Topics")
        ax[1].set_xlabel("K")
        ax[1].set_ylabel("Log-Likelihood")
        ax[1].set_yscale("log")
        ax[1].grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.markdown(""" 
    - Topics align strongly with **product categories**
    - Sessions reflect **coherent browsing intent**
    - Similar product groups appear within the same session
    - Price ranges and categories are consistent within topics
    """)
    st.markdown("### Visual comparison")

    images = {
        "TF-IDF clustering performance": "tf-idf.png",
        "Word2Vec pooling clustering performance": "word2vec.png",
        "LDA topics": "lda_analysis.png",
    }

    selected = st.radio(
        "Select method",
        list(images.keys()),
        horizontal=True
    )

    st.image(path.joinpath(images[selected]), use_container_width=True)

    with st.expander("Summary"):
        st.markdown("""
        **TF-IDF + K-Means clustering**
        - Used normalized sparse vectors with cosine-based evaluation.
        - Represents sessions based on **product occurrence frequencies**.
        - Captures co-occurrence of products **within sessions**.
        - Performance yields approx. 0.09 cosine silhouette only at K = 20.

        **LDA**
        - Captures latent topics well
        - Good for interpretability (browsing topics = product categories)

        **Embeddings (Word2Vec / pooled vectors) + K-Means**
        - Capture semantic similarity between products
        - Produce dense, meaningful representations
        - Lead to significantly better clustering performance

        Session behavior is better captured through **semantic representations**, not sparse frequency features.
        """)