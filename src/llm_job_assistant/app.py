import streamlit as st
from core import create_qa_chain


# Set the page title and a relevant icon
st.set_page_config(page_title="JobSift", page_icon=" briefcase")

# --- PAGE HEADER ---
st.title(" JobSift: Your Personal Job Insights Assistant")
st.write(
    "Welcome to JobSift! Ask any question about the job descriptions in your "
    "knowledge base to get instant, AI-powered answers."
)


# --- CORE APPLICATION LOGIC ---
# Use st.cache_resource to create the QA chain once and cache it
# This prevents re-creating the chain on every user interaction
@st.cache_resource
def load_qa_chain():
    """
    Loads the RetrievalQA chain.
    The @st.cache_resource decorator ensures this resource-intensive
    function is only run once.
    """
    return create_qa_chain()


# Load the QA chain
qa_chain = load_qa_chain()

# --- USER INTERFACE ---
# Create a text input box for the user's question
question = st.text_input(
    "Ask a question:",
    placeholder="What are the common skills required for a data scientist?",
)

# If the user has entered a question, run the QA chain and display the result
if question:
    with st.spinner("Searching for the answer..."):
        try:
            # Invoke the QA chain with the user's question
            result = qa_chain.invoke({"query": question})

            # Display the answer
            st.header("Answer")
            st.write(result["result"])

            # (Optional) Display the source documents for traceability
            with st.expander("Show Sources"):
                st.write("The following sources were used to generate the answer:")
                for doc in result["source_documents"]:
                    # Display the source and a snippet of the content
                    st.info(f"Source: {doc.metadata['source']}")
                    st.write(f"Content: {doc.page_content[:250]}...")
        except Exception as e:
            st.error(f"An error occurred: {e}")
