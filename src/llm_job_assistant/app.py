import streamlit as st
from core import create_qa_chain, NOT_FOUND_MESSAGE


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="JobSift", page_icon="briefcase")
    st.title(" JobSift: Your Personal Job Insights Assistant")
    st.write(
        "Welcome to JobSift! Ask any question about the job descriptions in your "
        "knowledge base to get AI-powered answers."
    )

    @st.cache_resource
    def load_qa_chain():
        return create_qa_chain()

    qa_chain = load_qa_chain()

    question = st.text_input(
        "Ask a question:",
        placeholder="What are the common skills required for a data scientist?",
    )

    if question:
        with st.spinner("Searching for the answer..."):
            try:
                result = qa_chain.invoke({"question": question})

                st.header("Answer")
                st.write(result["answer"])

                # Conditionally display sources only if the answer is not the "not found" message
                if result["answer"] != NOT_FOUND_MESSAGE and result["context"]:
                    with st.expander("Show Sources"):
                        st.write(
                            "The following sources were used to generate the answer:"
                        )
                        for doc in result["context"]:
                            st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"Content: {doc.page_content[:250]}...")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
