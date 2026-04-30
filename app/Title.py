import streamlit as st
from pathlib import Path
import base64

st.set_page_config(page_title="Title", layout="centered")

logo_path = Path(__file__).parent / "pages" / "figures" / "unioulu.png"

# encode logo so we can center it with html
with open(logo_path, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .title-page {{
        text-align: center;
        font-family: 'Georgia', serif;
        padding-top: 1rem;
    }}
    .title-page .logo {{
        margin-bottom: 2rem;
    }}
    .title-page .faculty {{
        font-variant: small-caps;
        font-size: 1.15rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        margin-bottom: 4rem;
        color: #222;
    }}
    .title-page .course {{
        font-size: 1rem;
        font-style: italic;
        margin-bottom: 1.5rem;
        color: #333;
    }}
    .title-page .main-title {{
        font-variant: small-caps;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.3;
        letter-spacing: 0.03em;
        margin-bottom: 4rem;
        color: #111;
    }}
    .title-page .authors {{
        font-variant: small-caps;
        font-size: 1.7rem;
        letter-spacing: 0.05em;
        line-height: 2;
        margin-bottom: 3rem;
        color: #222;
    }}
    .title-page .date {{
        font-size: 1rem;
        color: #444;
    }}
    </style>

    <div class="title-page">
        <div class="logo">
            <img src="data:image/png;base64,{logo_b64}" width="420">
        </div>
        <div class="faculty">Faculty of Information Technology and Electrical Engineering</div>
        <div class="course">Big Data Processing and Applications</div>
        <div class="main-title">Clickstream-based User Behavior<br>Analytics and Recommender System</div>
        <div class="authors">
            Nam Do<br>
            Sajjad Ghaeminejad<br>
            Seyyedhamid Azimidokht<br>
            Leo Davidov
        </div>
        <div class="date">April 30, 2026</div>
    </div>
    """,
    unsafe_allow_html=True,
)