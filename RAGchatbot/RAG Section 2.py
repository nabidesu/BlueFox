
import streamlit as st

# Experiment: Testing basic chat display
# st.chat_message("user").markdown("Test user message")
# st.chat_message("assistant").markdown("Test assistant response")
#
# # Experiment: Testing chat history persistence
# if "temp_chat" not in st.session_state:
#     st.session_state.temp_chat = []
# st.session_state.temp_chat.append({"role": "user", "content": "Test question"})
# st.session_state.temp_chat.append({"role": "assistant", "content": "Test answer"})
# for msg in st.session_state.temp_chat:
#     st.chat_message(msg["role"]).markdown(msg["content"])


def initialize_chat():
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_chat_history():
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(qa_chain):
    # Handle user input and generate response
    if prompt := st.chat_input("Ask a question about the resume"):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing resume..."):
            result = qa_chain({"query": prompt})
            response = result["result"]
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)
                with st.expander("Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(
                            f"**Page {doc.metadata.get('page', 'N/A')}**: {doc.page_content[:200]}...")
