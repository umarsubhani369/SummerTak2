import streamlit as st
from support_ticket_tagger import get_ticket_tags

st.set_page_config(
    page_title="Support Ticket Auto-Tagger",
    layout="centered",
)

# Custom CSS for clean design
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9f9;
        color: #222;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 6px;
        font-size: 15px;
    }
    .stButton>button {
        background-color: #0A84FF;
        color: white;
        font-weight: 300;
        border-radius: 3px;
        padding: 5px 8px;
        transition: 0.2s all;
    }
    .stButton>button:hover {
        background-color: #006DD2;
    }
    .stMarkdown h1 {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🎫 Support Ticket Auto-Tagger")
st.write("Paste your support ticket below and get the top 3 predicted tags instantly.")

with st.form("tag_form"):
    ticket = st.text_area("Enter your support ticket message", height=160)
    submitted = st.form_submit_button("Predict Tags")

if submitted:
    if ticket.strip():
        with st.spinner("Analyzing ticket..."):
            tags = get_ticket_tags(ticket)
        st.markdown("#### 🔖 Predicted Tags")
        for tag in tags:
            st.markdown(f"- **{tag}**")
    else:
        st.warning("Please enter a support ticket.")
