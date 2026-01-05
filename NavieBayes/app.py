import streamlit as st
from model import predict_message

st.set_page_config(page_title="College Message Classifier")

st.title("🏫 College Message Classification")

user_msg = st.text_area("Enter college-related message")

if st.button("Classify Message"):
    if user_msg.strip() == "":
        st.warning("Please enter a message")
    else:
        result = predict_message(user_msg)
        if result == "complaint":
            st.error("📛 Complaint Message")
        else:
            st.success("✅ Feedback Message")
