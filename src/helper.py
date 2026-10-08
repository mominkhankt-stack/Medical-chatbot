from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings



def load_pdf_files(Data):
    loader = DirectoryLoader(
        Data,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )
    Documents = loader.load()
    return Documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document objects containg
    only "source" in the metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    
    for doc in docs:
        src = doc.metadata.get("source")
        
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
        
    return minimal_docs


# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks


def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings
