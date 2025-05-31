
import streamlit as st

# Experiment: Testing different page layouts
# st.set_page_config(page_title="Test Chatbot", layout="centered")
# st.title("Centered Layout Test")
# st.write("Testing centered layout")
# st.set_page_config(page_title="Test Chatbot", layout="wide")
# st.title("Wide Layout Test")
# st.write("Testing wide layout")  # Preferred wide layout for more space
#
# # Experiment: Testing multiple file upload
# uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
# if uploaded_files:
#     print(f"Uploaded {len(uploaded_files)} files")


def setup_streamlit_ui():
    st.set_page_config(page_title="Resume Chatbot", layout="wide")
    st.title(" Resume Analysis Chatbot")
    st.write("Upload a resume (PDF) and ask questions about the candidate's qualifications, experience, or skills.")

    with st.sidebar:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")
    return uploaded_file
