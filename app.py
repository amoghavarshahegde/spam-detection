
import os
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# --------------------------------
# CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="NeuroShield | AI Spam Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------
# MODEL PATH
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_spam"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background: #080B14;
    color: #F4F7FF;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Main title */

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(
        90deg, #1677FF, #FFFFFF, #FF2E74
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9CAAC2;
    font-size: 16px;
    margin-bottom: 35px;
}

/* Section cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111827;
    border: 1px solid #25334D;
    border-radius: 16px;
}

/* Text area */

.stTextArea textarea {
    background: #0B1220 !important;
    color: #FFFFFF !important;
    border: 1px solid #1677FF !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}

.stTextArea textarea:focus {
    border: 1px solid #FF2E74 !important;
    box-shadow: 0 0 12px #FF2E7444 !important;
}

.stTextArea label {
    color: #DCE6F8 !important;
    font-weight: 600;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg, #1677FF, #FF2E74
    );
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    box-shadow: 0 0 20px #FF2E7466;
    transform: translateY(-2px);
    color: white;
    border: none;
}

/* Result cards */

.result-spam {
    background: #24101C;
    border: 1px solid #FF2E74;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 0 18px #FF2E7422;
}

.result-safe {
    background: #0B1D2B;
    border: 1px solid #1677FF;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 0 18px #1677FF22;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 8px;
}

.confidence {
    font-size: 35px;
    font-weight: 800;
    color: #FFFFFF;
}

/* Footer */

.footer {
    text-align: center;
    color: #74829A;
    font-size: 13px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD DISTILBERT MODEL
# --------------------------------

@st.cache_resource
def load_model():

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_model()

except Exception as e:
    st.error("Unable to load the DistilBERT model.")
    st.code(str(e))
    st.stop()


# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">NEUROSHIELD AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Email Security | Powered by DistilBERT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------
# MAIN APPLICATION
# --------------------------------

st.markdown("### 🛡️ Email Threat Analyzer")

st.write(
    "Analyze your email content using our fine-tuned "
    "AI language model."
)

email = st.text_area(
    "EMAIL CONTENT",
    height=220,
    placeholder="Paste your email message here..."
)

st.write("")

if st.button("⚡ ANALYZE EMAIL", use_container_width=True):

    if not email.strip():

        st.warning("Please enter an email before analyzing.")

    else:

        with st.spinner("AI is analyzing your email..."):

            inputs = tokenizer(
                email,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():

                outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[0][prediction].item() * 100
            )

        st.markdown("---")

        st.markdown("### ANALYSIS RESULT")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-spam">

                    <div class="result-title"
                         style="color:#FF2E74;">
                        🚨 SPAM DETECTED
                    </div>

                    <p style="color:#C4A5B3;">
                        This email has been classified as suspicious.
                    </p>

                    <div class="confidence">
                        {confidence:.2f}%
                    </div>

                    <p style="color:#FF8CAE;">
                        SPAM CONFIDENCE
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-safe">

                    <div class="result-title"
                         style="color:#3295FF;">
                        🛡️ LEGITIMATE EMAIL
                    </div>

                    <p style="color:#A8C9E8;">
                        This email has been classified as not spam.
                    </p>

                    <div class="confidence">
                        {confidence:.2f}%
                    </div>

                    <p style="color:#68B2FF;">
                        NOT SPAM CONFIDENCE
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------
# FOOTER
# --------------------------------

st.markdown(
    """
    <div class="footer">
        NEUROSHIELD AI • DISTILBERT INTELLIGENCE
        <br><br>
        AI-powered email protection system
    </div>
    """,
    unsafe_allow_html=True
)
