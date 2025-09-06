from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# --- CONSTANTS ---
CHROMA_PATH = "chroma"


def create_qa_chain():
    """
    Creates and returns a RAG chain with query rewrite and source document return.
    """
    # Initialize the models
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()

    # Load the Chroma vector store and create a retriever
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever()

    # The Rewriting Chain
    # Optimizes the user's question for vector database search.
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert query optimizer. Rewrite the following user question to be a detailed, specific query for a vector database containing job descriptions. Focus on extracting key skills, technologies, and experience levels.",
            ),
            ("user", "{question}"),
        ]
    )

    query_rewriter = rewrite_prompt | llm | StrOutputParser()

    # The Answering Chain
    # Synthesizes the final answer from the retrieved documents.
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer the user's question based on the below context.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        Context:
        {context}
        """,
            ),
            ("user", "{question}"),
        ]
    )

    # The Full RAG Chain with Source Return
    # This chain now also returns the rewritten question for debugging.
    chain = (
        RunnablePassthrough.assign(
            rewritten_question=query_rewriter,
        )
        .assign(context=itemgetter("rewritten_question") | retriever)
        .assign(answer=answer_prompt | llm | StrOutputParser())
    )

    return chain


def main():
    """
    Main function to test the QA chain's full invocation.
    """
    # Create the QA chain
    qa_chain = create_qa_chain()

    # Ask a question for testing
    question = "What are the key skills required for a data scientist?"
    print(f"Original Question: {question}\n")

    # Get the result from the chain
    result = qa_chain.invoke({"question": question})

    # Print the rewritten query
    print("Rewritten Query:")
    print(result["rewritten_question"])

    # Print the answer
    print("\nAnswer:")
    print(result["answer"])

    # Print the source documents
    print("\nSource Documents:")
    for doc in result["context"]:
        print(f"- Source: {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()
