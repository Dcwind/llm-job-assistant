import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- CONSTANTS ---
# Use the sample_data directory by default
# The user can override this by creating a 'data' directory
DATA_PATH = "data" if os.path.exists("data") else "sample_data"
CHROMA_PATH = "chroma"


# --- MAIN INGESTION LOGIC ---
def main():
    """
    Main function to handle the data ingestion process.
    - Loads documents from the specified data path.
    - Splits the documents into manageable chunks.
    - Creates a Chroma vector store with OpenAI embeddings.
    - Saves the vector store to disk for persistence.
    """
    # Check if the vector store already exists
    if os.path.exists(CHROMA_PATH):
        logging.info("Vector store already exists. Skipping ingestion.")
        return

    # Load documents from the data directory
    logging.info(f"Loading documents from {DATA_PATH}...")
    # Use TextLoader for individual .txt files to avoid potential metadata issues
    document_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = document_loader.load()
    if not documents:
        logging.warning("No documents found in the data directory. Exiting.")
        return
    logging.info(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")

    # Create the Chroma vector store with OpenAI embeddings
    logging.info("Creating vector store with OpenAI embeddings...")
    # The from_documents method handles embedding creation and storage
    Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    logging.info("Vector store created successfully.")

    logging.info(f"Vector store saved to {CHROMA_PATH}. Ingestion complete.")


if __name__ == "__main__":
    main()
