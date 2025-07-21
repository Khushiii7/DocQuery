import streamlit as st
import requests

st.set_page_config(page_title="Hugging Face Transformers QA", layout="wide")
st.title("Hugging Face Transformers Documentation QA")
st.write("Ask any question about the Transformers library documentation!")

API_URL = "http://localhost:8000/query"

with st.form("qa_form"):
    question = st.text_input("Enter your question:", "How do I install Transformers using pip?")
    submit = st.form_submit_button("Get Answer")

if submit and question.strip():
    with st.spinner("Searching for the answer..."):
        try:
            response = requests.post(API_URL, json={"question": question})
            if response.status_code == 200:
                data = response.json()
                if "answer" in data:
                    st.success(f"**Answer:** {data['answer']}")
                    if data.get("context"):
                        with st.expander("Show supporting context"):
                            st.write(data["context"])
                elif "error" in data:
                    st.error(f"Error: {data['error']}")
                else:
                    st.warning("No answer found.")
            else:
                st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io/) · Powered by Hugging Face Transformers · Retrieval-Augmented Generation (RAG)") 