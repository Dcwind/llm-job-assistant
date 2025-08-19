import streamlit as st
from core import create_qa_chain


def main():
    """
    Main function to run the Streamlit application.
    """
    # --- PAGE CONFIGURATION ---
    st.set_page_config(page_title="JobSift", page_icon=" briefcase")

    # --- PAGE HEADER ---
    st.title(" JobSift: Your Personal Job Insights Assistant")
    st.write(
        "Welcome to JobSift! Ask any question about the job descriptions in your "
        "knowledge base to get instant, AI-powered answers."
    )

    # --- CORE APPLICATION LOGIC ---
    @st.cache_resource
    def load_qa_chain():
        """
        Loads the RetrievalQA chain.
        The @st.cache_resource decorator ensures this resource-intensive
        function is only run once.
        """
        return create_qa_chain()

    qa_chain = load_qa_chain()

    # --- USER INTERFACE ---
    question = st.text_input(
        "Ask a question:",
        placeholder="What are the common skills required for a data scientist?",
    )

    if question:
        with st.spinner("Searching for the answer..."):
            try:
                result = qa_chain.invoke({"query": question})

                st.header("Answer")
                st.write(result["result"])

                with st.expander("Show Sources"):
                    st.write("The following sources were used to generate the answer:")
                    # Check if source_documents exists and is not empty
                    if "source_documents" in result and result["source_documents"]:
                        for doc in result["source_documents"]:
                            st.info(f"Source: {doc.metadata['source']}")
                            st.write(f"Content: {doc.page_content[:250]}...")
                    else:
                        st.warning("No source documents found for this answer.")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
