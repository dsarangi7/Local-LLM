import subprocess
import importlib
from typing import Optional, List
import os
import hashlib
import shutil
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import requests

# Constants
PROMPT_TEMPLATE = """
You are an expert assistant specializing in battery technology and project proposals. Use only the following context from uploaded PDFs to answer the question. The PDFs include:
- **Proposals**: Each has a project number starting with "BABESS" (e.g., BABESS1843) and a project name as the document title.
- **Technotes**: These detail technical information about battery systems and their components.

Context:
{context}

Question: {question}
Answer: Provide a precise and concise response based solely on the context. If the question relates to a specific project (e.g., BABESS number or project name) or technote, reference the relevant details (e.g., project number, name, or component) explicitly. If the context lacks sufficient information, say "The provided context does not contain enough information to answer this question."
"""

# Ollama configuration
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def check_ollama_model():
    """Check if Ollama server is running by hitting a valid endpoint."""
    try:
        print("Attempting to connect to Ollama server...")
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        print(f"Server response status code: {response.status_code}")
        print(f"Server response text: {response.text}")
        
        if response.status_code == 200:
            print("✓ Ollama server is running and responding")
            return True
        print(f"Error: Ollama server returned status {response.status_code}")
        return False
    except requests.ConnectionError:
        print("Error: Could not connect to Ollama server at 127.0.0.1:11434. Is it running?")
        return False
    except Exception as e:
        print(f"Error connecting to Ollama server: {str(e)}")
        return False

def check_cuda_availability():
    """Check if CUDA is available for GPU support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            subprocess.check_call(["pip", "install", "--no-cache-dir", "torch"])
            import torch
            return torch.cuda.is_available()
        except:
            return False

def install_required_packages():
    """Install required packages if not already installed."""
    packages = [
        ("langchain", "LLM Library"),
        ("pypdf", "PDF support"),
        ("tqdm", "Progress bars"),
        ("sentence_transformers", "Embeddings generation"),
        ("langchain-community", "LangChain community components"),
        ("langchain-core", "LangChain core functionality"),
        ("chromadb", "Vector store"),
        ("langchain-chroma", "Chroma integration"),
        ("requests", "HTTP client")
    ]
    
    for package, comment in packages:
        try:
            if not check_package(package.split('[')[0]):
                print(f"Installing {package}...")
                subprocess.check_call(["pip", "install", "--no-cache-dir", package])
                print(f"Successfully installed {package} # {comment}")
        except Exception as e:
            print(f"Error installing {package}: {str(e)}")
            raise

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False

class DocumentProcessor:
    def __init__(self, data_dir: str = "data", db_dir: str = "db"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None
        self.qa_chain = None
        
        os.makedirs(self.db_dir, exist_ok=True)

    def process_single_pdf(self, file_path: str) -> List[Document]:
        """Process a single PDF file and return pages as documents."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            processed_docs = []
            for i, doc in enumerate(documents):
                unique_id_input = f"{file_path}_{doc.metadata.get('page', 0)}"
                doc.metadata['doc_id'] = self.generate_document_id(unique_id_input)
                doc.metadata['source'] = file_path
                doc.metadata['page_number'] = doc.metadata.get('page', 0)
                processed_docs.append(doc)
            
            print(f"✓ Processed {os.path.basename(file_path)} - {len(processed_docs)} pages")
            return processed_docs
        except Exception as e:
            print(f"✗ Error processing {file_path}: {str(e)}")
            return []

    def load_and_process_documents(self):
        """Load and process documents with parallel processing."""
        print("Scanning for PDF files...")
        pdf_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.data_dir}")
        
        print(f"Found {len(pdf_files)} PDF files. Processing...")
        all_documents = []
        
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(pdf_files))) as executor:
            futures = [executor.submit(self.process_single_pdf, pdf_file) for pdf_file in pdf_files]
            for future in futures:
                chunks = future.result()
                all_documents.extend(chunks)
        
        print(f"\nTotal chunks created: {len(all_documents)}")
        return all_documents

    @staticmethod
    def generate_document_id(content: str) -> str:
        """Generate a unique ID for a document chunk based on content and position."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def initialize_vector_store(self, documents: Optional[List[Document]] = None):
        """Initialize or update the Chroma vector store."""
        print("Initializing/updating vector store...")
        
        try:
            self.db = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            if documents:
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.db.add_documents(batch)
                    print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                print("Vector store successfully updated.")
            else:
                print("Using existing Chroma database.")
                
        except Exception as e:
            print(f"Error during vector store initialization: {e}")
            self.db = None
            raise

    def initialize_rag(self):
        """Initialize the RAG system."""
        if not self.db:
            self.initialize_vector_store()

        print("\nInitializing Ollama model...")
        if not check_ollama_model():
            raise RuntimeError(
                "Could not connect to Ollama server. Please ensure it is running with 'ollama serve'."
            )
        
        print("Initializing LLM with Ollama...")
        llm = Ollama(
            model=OLLAMA_MODEL,
            temperature=0.1,
            top_k=5,
            num_ctx=8192,
            verbose=True,
            base_url=OLLAMA_BASE_URL
        )
        
        # Configure Chroma retriever without score_threshold in search_kwargs
        retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Removed score_threshold
        )
        
        prompt = PromptTemplate(
            template="""
Answer the question based on the following pages from the documents:
{context}

Question: {question}
Answer: Let me provide a comprehensive answer based on the relevant pages:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    def query_rag(self, query: str) -> tuple[str, Optional[None]]:
        """Process a query through the RAG system."""
        if not self.qa_chain:
            self.initialize_rag()

        try:
            response = self.qa_chain.invoke({"query": query})
            return response["result"], None
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg, None

    def similarity_search(self, query: str, k: int = 3):
        """Perform similarity search using Chroma."""
        if not self.db:
            self.initialize_vector_store()

        results = self.db.similarity_search_with_score(query, k=k)
        
        print("\nTop relevant pages with scores:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Page: {doc.metadata['page_number'] + 1}")
            print("Content:", doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            print("-" * 80)

def main():
    install_required_packages()
    
    processor = DocumentProcessor(
        data_dir="C:/Users/dsara/Desktop/Local LLM/data",
        db_dir="db"
    )
    
    print("Loading and splitting documents...")
    documents = processor.load_and_process_documents()
    print(f"Number of documents after splitting: {len(documents)}")
    
    print("\nChecking and updating vector database...")
    processor.initialize_vector_store(documents)
    
    print("\nEntering interactive mode...")
    print("1. RAG Query Mode")
    print("2. Similarity Search Mode")
    print("3. Exit")
    
    while True:
        try:
            mode = input("\nSelect mode (1-3): ").strip()
            
            if mode == "3":
                break
                
            query = input("\nEnter your question: ")
            
            if mode == "1":
                print("\nProcessing RAG query...")
                response, _ = processor.query_rag(query)
                print("\nResponse:", response)
            elif mode == "2":
                print("\nPerforming similarity search...")
                processor.similarity_search(query)
            else:
                print("Invalid mode selected. Please choose 1-3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()

