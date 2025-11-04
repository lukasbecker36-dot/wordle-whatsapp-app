import streamlit as st

st.title("Wordle WhatsApp Analyzer 🟩🟨⬛ — v2 test")

st.write("Upload your exported WhatsApp chat to begin!")

uploaded = st.file_uploader("Upload _chat.txt", type=["txt"])
if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")
    st.success(f"Loaded {len(text):,} characters")
    st.text_area("Preview:", text[:500], height=200)
