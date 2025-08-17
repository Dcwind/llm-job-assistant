from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# --- CONSTANTS ---
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
"""


def main():
    """
    Main function to set up and run the RAG QA chain.
    This is for demonstration and testing purposes.
    """
    # Create the QA chain
    qa_chain = create_qa_chain()

    # Ask a question for testing
    question = "What are the key skills required for a data scientist?"
    print(f"Question: {question}")

    # Get the answer
    answer = qa_chain.invoke({"query": question})
    print("Answer:")
    print(answer)


def create_qa_chain():
    """
    Creates and returns a RetrievalQA chain.

    This function initializes the necessary components for the RAG pipeline:
    - The LLM (ChatOpenAI).
    - The vector store (ChromaDB) with an embedding function.
    - The retriever to fetch relevant documents.
    - A prompt template for structuring the query to the LLM.
    """
    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Initialize the embedding function and vector store
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Create the retriever
    retriever = db.as_retriever()

    # Create the prompt from a template
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


if __name__ == "__main__":
    # This block allows the script to be run directly for testing
    main()
