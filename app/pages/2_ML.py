import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("User-Entity Behavior Modeling")
st.write("This page provides an overview on Event- and Session-level clustering experiments, results, and performance indicators. Along with some complementary analyses.")
from pathlib import Path
path = Path(__file__).parent / 'figures'

tab2, tab3, tab4, tab5 = st.tabs([
    "Event-Level Experiments",
    "Session-Level Experiments",
    "Event-Level Cluster Analysis",
    "Session-Level Cluster Analysis"
])

with tab2:
    st.markdown("## Event-Level Clustering")

    st.success(
        "Best configuration - **Experiment 2**"
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
        "Experiment 2 (best) - No temporal features": "output2.png",
        "Experiment 1 - Linear day encoding": "output1.png",
        "Experiment 0 - Sin/Cos day encoding + month": "output.png",
    }

    selected = st.radio(
        "Select experiment",
        list(images.keys()),
        horizontal=True
    )

    st.image(path.joinpath(images[selected]))

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

        ### Gaussian Mixture Model (GMM)
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
    - There are as much unique click feature combinations as there are items (217 items, 218 combinations)
    - Country excluded
    - 218 tokens
    - **TF-IDF (normalized) + K-Means**: poor clustering metrics, no meaningful structure at interpretable number of clusters.
    - **Latent Dirichlet Allocation (LDA)**: 
        - Top number of topics K = 4
        - Model quality plateaus quickly and for larger K improvements are modest
    - **Embedding-based clustering (good window size 3-5)**:
        - Good separation: Peak K ~ 4-6 (depending on seed), cosine silhouette ~ 0.43
        - More consistent structure
        - Item-click 32 dimensional embeddings form small well separated clusters in space.
    """)

    st.markdown("### Visual comparison")

    images = {
        "TF-IDF performance": "tf-idf.png",
        "Word2Vec context pooling performance": "word2vec.png",
        "LDA performance": None
    }

    lda_logs = pd.DataFrame({
        "K": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "perplexity": [4.991, 4.919, 4.952, 4.930, 4.947, 4.971, 4.967, 4.967, 4.997, 5.000, 5.062, 5.076],
        "log_likelihood": [-825953, -813965, -819451, -815740, -818550, -822538, -821880, -821914, -826799, -827338, -837572, -839868],
    
    })
    best = lda_logs.loc[lda_logs["perplexity"].idxmin()]
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
    fig.tight_layout()

    selected = st.radio(
        "Select method",
        list(images.keys()),
        horizontal=True
    )
    if selected == "LDA performance":
        st.pyplot(fig, use_container_width=False)
    else:
        st.image(path.joinpath(images[selected]))

    st.markdown("### LDA")
    st.info(f"Best LDA configuration: **K = {int(best['K'])}**, "    f"perplexity = **{best['perplexity']:.3f}**, "    f"log-likelihood = **{best['log_likelihood']:.0f}**")
    st.image(path.joinpath("lda_analysis.png"))
    st.markdown(""" 
    - Topics align strongly with **product categories**
    - Sessions reflect **coherent browsing intent**
    - Similar product groups appear within the same session
    - Price ranges and categories are consistent within topics
    """)

    with st.expander("Summary"):
        st.markdown("""
        **TF-IDF + K-Means clustering**
        - Used normalized sparse vectors with cosine-based evaluation.
        - Captures co-occurrence of products within sessions.
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
    with tab4:
        st.markdown("## Event-Level Cluster Profiling")
        st.info(
            "This section profiles the resulting event-level clusters using product range, "
            "price behavior, colours, categories, and item-level distributions.  "
            "Clusters are formed using K-Means, according to the best model selection and feature experiments."
        )

        event_path = path / "event_level"

        images = {
            "Cluster size": "cluster_size.png",
            "Product range": "product_range.png",
            "Price profiling": "price_profiling.png",
            "Average depth & photography preference": "depth_photo.png",
            "Top colours": "top_colors.png",
            "Top items": "top_items.png",
            "Main categories": "categories.png",
        }

        selected = st.radio(
            "Select cluster analysis figure",
            list(images.keys()),
            horizontal=True,
            key="event_cluster_analysis"
        )

        st.image(event_path / images[selected])

        st.markdown("### Summary")

        st.markdown("""
        Event-level clustering primarily groups items based on **click frequency patterns** and **feature similarity**, rather than capturing user intent or session behavior.

        - Clusters are largely driven by **item popularity**, separating frequently clicked items from less popular ones.
        - High-volume clusters tend to contain **top-performing or highly exposed products**.
        - Lower-volume clusters group **long-tail or niche items** with similar feature representations.
        - Feature similarity (e.g., category, price, attributes) further refines grouping within similar popularity levels.
        - The model captures **structural patterns in item interactions**.
        - Clusters reflect **which items behave similarly in terms of clicks**, not why users clicked them.
        - “Exploration vs intent” is **not directly encoded** at this level.
        - Top country in all clusters is Poland (quite obviously)
                    
        Similar raw event-level clustering in particular was done by Artioli et al. "A comprehensive investigation of clustering algorithms for user and entity behavior analytics", as well as paper by Datta et al. “Real-time threat detection in ueba using unsupervised learning algorithms”.
        """)

    with tab5:
        st.markdown("## Session-Level Cluster Profiling (Word2Vec)")

        st.info(
            "This section profiles session-level clusters built from Word2Vec embeddings. "
            "Unlike raw event-level clustering, these clusters are based on contextual co-occurrence between products within sessions."
        )

        session_path = path / "session_level"

        images = {
            "Projected context embeddings": "context_embeddings.png",
            "Projected click embeddings": "click_embeddings.png",
            "Cluster size": "clicks_session_counts.png",
            "Average depth & photography preference": "avg_depth_photo.png",
            "Price profiling": "prices.png",
            "Top colours": "top_colors.png",
            "Top products": "top_products.png",
        }

        selected = st.radio(
            "Select Word2Vec cluster analysis figure",
            list(images.keys()),
            horizontal=True,
            key="word2vec_cluster_analysis"
        )

        st.image(session_path / images[selected])

        st.markdown("### Summary")

        st.markdown("""
        Session-level clustering based on Word2Vec embeddings captures **co-occurrence structure**
        and **contextual similarity between products**, rather than only global click frequency.

        - The embedding space reflects products that appear in similar session contexts.
        - Clusters represent groups of sessions with similar browsing context.
        - Product, colour, price, and depth profiles help explain what each cluster contains.
        - Popularity effects may still exist, but they are less dominant than in raw event-level clustering.
        """)

        st.markdown("### Relation to LDA")

        st.markdown("""
        Represented clusters show strong parallels with **Latent Dirichlet Allocation (LDA)** results.

        - Both approaches rely on **co-occurrence patterns within sessions**:
            - LDA models sessions as mixtures of latent topics
            - Word2Vec embeddings capture similar structure through local context windows
        - Clusters derived from embeddings often align with **topic-like groupings** observed in LDA driven by product category
        - In practice, Word2Vec can be seen as a **continuous, geometry-based analogue of topic modeling**

        This explains why both methods highlight similar product groupings and browsing patterns (top products),
        even though they are based on different mathematical formulations.
                    
        Resulting clusters are suitable for session-based recommendation, contextual product analysis, and understanding latent browsing patterns.           
        """)

        st.markdown("### Bonus - Recommender sanity check with 80-20 split stratified by session lengths")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**No filtering (double interactions)**")
            st.metric("Recall@10", "0.456")
            st.metric("NDCG@10", "0.221")

        with col2:
            st.markdown("**Filtered (strict)**")
            st.metric("Recall@10", "0.328")
            st.metric("NDCG@10", "0.162")
