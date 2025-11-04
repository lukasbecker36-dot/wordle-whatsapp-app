import streamlit as st
import pandas as pd
import re
from datetime import datetime

st.set_page_config(page_title="Wordle WhatsApp Analyzer", page_icon="🟩", layout="wide")
st.title("Wordle WhatsApp Analyzer 🟩🟨⬛")

uploaded = st.file_uploader("Upload your WhatsApp export (_chat.txt)", type=["txt"])
date_format = st.radio("Date format in your chat", ["dd/mm/yyyy", "mm/dd/yyyy"], horizontal=True)

if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")

    # Clean special characters often found in WhatsApp exports
    text = (text.replace("\u202f", " ")
                .replace("\u200e", "")
                .replace("\u00a0", " "))

    fmt = "%d/%m/%Y" if date_format == "dd/mm/yyyy" else "%m/%d/%Y"

    # Handle both Android and iOS styles
    p1 = re.compile(r"\[(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?::\d{2})?\]\s+([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6")
    p2 = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}),\s*\d{1,2}:\d{2}(?:\s*[APMapm]{2})?\s*-\s*([^:]+?):\s+Wordle\s+[\d,]+\s+([1-6xX])/6")

    rows = []
    for (date_str, name, score) in p1.findall(text) + p2.findall(text):
        try:
            date = datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
        name = re.sub(r"^[^A-Za-z]*", "", name).strip()
        numeric = 7 if score.upper() == "X" else int(score)
        rows.append({"Date": date, "Name": name, "Score": numeric})

    if not rows:
        st.warning("No Wordle lines found. Make sure your export contains lines like 'Wordle #### 3/6'.")
    else:
        df = pd.DataFrame(rows).sort_values(["Date", "Name"])
        st.success(f"Parsed {df['Date'].nunique()} days, {df['Name'].nunique()} players, {len(df)} entries.")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download parsed CSV",
            df.to_csv(index=False).encode("utf-8"),
            "wordle_parsed.csv",
            "text/csv"
        )
