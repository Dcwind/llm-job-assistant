from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.retrievers import MultiQueryRetriever

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- CONSTANTS ---
CHROMA_PATH = "chroma"
NOT_FOUND_MESSAGE = (
    "This information is not available in the source files I have access to."
)


def create_qa_chain():
    """
    Creates and returns a RAG chain using the Multi-Query Retriever pattern.
    """
    # Initialize models
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()

    # Load retriever from the absolute path
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    base_retriever = db.as_retriever()

    # The Multi-Query Retriever
    # This automatically generates multiple queries from different perspectives
    # and fuses the results to get a richer set of documents.
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    # The Answering Chain
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""Answer the user's question based on the below context.
                If the context does not contain the answer, reply with the exact phrase: {NOT_FOUND_MESSAGE}

                Context:
                {{context}}
                """,
            ),
            ("user", "{question}"),
        ]
    )

    # The Full RAG Chain with Source Return
    # This chain is now simpler and more robust due to the Multi-Query Retriever.
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    # We need a separate chain to pass through the context for source display
    chain_with_sources = RunnablePassthrough.assign(
        answer=chain,
        context=itemgetter("question") | retriever,
    )

    return chain_with_sources


def main():
    """
    Main function to test the QA chain's full invocation.
    """
    qa_chain = create_qa_chain()
    question = "What are the key skills required for a data scientist?"
    print(f"Original Question: {question}\n")

    result = qa_chain.invoke({"question": question})

    print("Answer:")
    print(result["answer"])

    print("\nSource Documents:")
    if result["context"]:
        for doc in result["context"]:
            print(f"- Source: {doc.metadata.get('source', 'Unknown')}")
    else:
        print("No source documents found.")


if __name__ == "__main__":
    main()


# def create_qa_chain():
#     """
#     Creates and returns a RAG chain with query rewrite and source document return.
#     """
#     # Initialize models
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     embeddings = OpenAIEmbeddings()

#     # Load retriever
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
#     retriever = db.as_retriever()

#     # The Rewriting Chain
#     rewrite_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are an expert query optimizer. Rewrite the user question to be a detailed, specific query for a vector database containing job descriptions.",
#             ),
#             ("user", "{question}"),
#         ]
#     )
#     query_rewriter = rewrite_prompt | llm | StrOutputParser()

#     # The Answering Chain
#     answer_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 f"""Answer the user's question based on the below context.
#                 If the context does not contain the answer, reply with the exact phrase: {NOT_FOUND_MESSAGE}

#                 Context:
#                 {{context}}
#                 """,
#             ),
#             ("user", "{question}"),
#         ]
#     )

#     # The Full RAG Chain with Source Return
#     chain = (
#         RunnablePassthrough.assign(rewritten_question=query_rewriter)
#         .assign(context=itemgetter("rewritten_question") | retriever)
#         .assign(answer=answer_prompt | llm | StrOutputParser())
#     )

#     return chain


# def main():
#     """
#     Main function to test the QA chain's full invocation.
#     """
#     qa_chain = create_qa_chain()
#     question = "What are the common skills for a plumber?"
#     # question = "What are the key skills required for a data scientist?"
#     print(f"Original Question: {question}\n")
#     result = qa_chain.invoke({"question": question})

#     print("Rewritten Query:")
#     print(result["rewritten_question"])

#     print("\nAnswer:")
#     print(result["answer"])

#     print("\nSource Documents:")
#     if result["context"]:
#         for doc in result["context"]:
#             print(f"- Source: {doc.metadata.get('source', 'Unknown')}")
#     else:
#         print("No sources found (as expected for this query).")


# if __name__ == "__main__":
#     main()
