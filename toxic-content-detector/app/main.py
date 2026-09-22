"""Streamlit UI for the toxic content detector."""

import streamlit as st

from app.classifier import classify_text

st.set_page_config(
    page_title="Toxic Content Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Toxic Content Detector")
st.caption("LangChain + OpenRouter + Streamlit")

st.write(
    "Enter text below. The application uses an LLM to classify it as "
    "**Toxic** or **Non-Toxic** and gives a short explanation."
)

text = st.text_area(
    "Text to analyze",
    height=180,
    placeholder="Example: I really enjoyed this course and learned a lot.",
)

if st.button("🔍 Analyze text", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing with the LLM..."):
            try:
                result = classify_text(text)
            except Exception as exc:
                st.error("The analysis could not be completed.")
                st.exception(exc)
            else:
                st.subheader("Classification")
                if result.classification == "Toxic":
                    st.error("⚠️ Toxic")
                else:
                    st.success("✅ Non-Toxic")

                st.subheader("Explanation")
                st.write(result.explanation)

st.divider()
st.caption("For educational use. LLM classifications can be imperfect.")
