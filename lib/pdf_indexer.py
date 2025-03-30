import os
import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

# Document processing
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store and embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# For future agent integration
from langchain.schema import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_indexer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf_indexer")

class PDFIndexer:
    """
    Indexes PDF documents using vector embeddings for semantic search capabilities.
    Handles document loading, text splitting, embedding generation, and vector storage.
    """
    
    def __init__(
        self,
        pdf_folder: str = "pdfs",
        vector_store_path: str = "vector_store",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the PDF indexer with configuration parameters.
        
        Args:
            pdf_folder: Directory containing PDF files
            vector_store_path: Directory to save the vector store
            embedding_model_name: Name of the HuggingFace embedding model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.pdf_folder = Path(pdf_folder)
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings with specifically all-MiniLM-L6-v2 using PyTorch
        logger.info(f"Initializing embeddings model: sentence-transformers/all-MiniLM-L6-v2")
        try:
            # Direct PyTorch implementation for MiniLM
            from transformers import AutoTokenizer, AutoModel
            import torch
            import numpy as np

            # Try different possible import paths for Embeddings base class
            try:
                from langchain.embeddings.base import Embeddings  # older versions
            except ImportError:
                try:
                    from langchain_core.embeddings import Embeddings  # newer versions
                except ImportError:
                    # Define our own minimal base class if we can't import
                    class Embeddings:
                        """Base class for embeddings"""
                        def embed_documents(self, texts): 
                            """Embed documents"""
                            pass
                        def embed_query(self, text): 
                            """Embed query"""
                            pass
            
            # Custom implementation specifically for all-MiniLM-L6-v2 with PyTorch
            class MiniLMPyTorchEmbeddings(Embeddings):
                def __init__(self):
                    # Load model with explicit PyTorch configuration
                    self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                    self.model = AutoModel.from_pretrained(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        torch_dtype=torch.float32,  # Explicitly use float32
                        trust_remote_code=False
                    )
                    # Move to GPU if available
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model.to(self.device)
                    logger.info(f"MiniLM PyTorch model loaded successfully on {self.device}")
                
                def _mean_pooling(self, model_output, attention_mask):
                    # Mean pooling to get sentence embeddings
                    token_embeddings = model_output.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                def _get_embeddings(self, texts):
                    # Tokenize and prepare inputs
                    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                    
                    # Get model output
                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                    
                    # Perform mean pooling
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    return embeddings.cpu().numpy().tolist()
                
                def embed_documents(self, texts):
                    """Generate embeddings for a list of documents."""
                    # Process in batches for memory efficiency
                    batch_size = 32
                    embeddings = []
                    
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        batch_embeddings = self._get_embeddings(batch)
                        embeddings.extend(batch_embeddings)
                    
                    return embeddings
                
                def embed_query(self, text):
                    """Generate embeddings for a query."""
                    return self._get_embeddings([text])[0]
                
            # Use our custom PyTorch implementation
            self.embeddings = MiniLMPyTorchEmbeddings()
            logger.info("MiniLM PyTorch embeddings initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MiniLM PyTorch embeddings: {str(e)}")
            raise ValueError(f"Failed to initialize MiniLM PyTorch embeddings: {str(e)}")

        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Vector store reference
        self.vector_store = None
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a PDF file and extract its text content with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with text and metadata
        """
        logger.info(f"Loading PDF: {file_path}")
        try:
            # Extract text from PDF
            pdf_reader = pypdf.PdfReader(file_path)
            documents = []
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Skip empty pages
                    metadata = {
                        "file_name": file_path.name,
                        "page": i + 1,
                        "total_pages": len(pdf_reader.pages)
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
            
            logger.info(f"Successfully extracted {len(documents)} pages from {file_path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better indexing and retrieval.
        Preserves and updates metadata during splitting.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects after splitting
        """
        logger.info(f"Splitting {len(documents)} documents with chunk size {self.chunk_size} and overlap {self.chunk_overlap}")
        try:
            splits = []
            for doc in documents:
                doc_splits = self.text_splitter.split_documents([doc])
                
                # Update metadata for each split to include paragraph information
                for i, split in enumerate(doc_splits):
                    split.metadata["paragraph"] = i + 1
                    split.metadata["total_paragraphs"] = len(doc_splits)
                
                splits.extend(doc_splits)
            
            logger.info(f"Created {len(splits)} splits from {len(documents)} documents")
            return splits
        
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        try:
            start_time = time.time()
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Vector store created successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self) -> None:
        """Save the vector store to disk for future use"""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        logger.info(f"Saving vector store to {self.vector_store_path}")
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Save vector store
            self.vector_store.save_local(str(self.vector_store_path))
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> bool:
        """
        Load a previously saved vector store.
        
        Returns:
            True if vector store was loaded successfully
        """
        vector_store_index = self.vector_store_path / "index.faiss"
        if not vector_store_index.exists():
            logger.warning(f"Vector store not found at {self.vector_store_path}")
            return False
        
        logger.info(f"Loading vector store from {self.vector_store_path}")
        try:
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path), 
                self.embeddings
            )
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant document chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing document text and metadata
        """
        if self.vector_store is None:
            logger.error("No vector store available for querying")
            raise ValueError("Vector store not initialized. Run index_documents() first.")
        
        logger.info(f"Querying vector store with: '{query}'")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                }
                formatted_results.append(result)
            
            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise

    def search_by_document(self, query, top_k_per_doc=3):
        """
        Query the vector store and group results by document.
        
        Args:
            query: Search query string
            top_k_per_doc: Number of top results to return per document
            
        Returns:
            Dictionary with document names as keys and lists of results as values
        """
        # Get more results to ensure coverage across documents
        all_results = self.query(query, top_k=50)  # Get more results to ensure coverage
        
        # Group by document name
        doc_results = {}
        for result in all_results:
            doc_name = result['metadata']['file_name']
            if doc_name not in doc_results:
                doc_results[doc_name] = []
            
            if len(doc_results[doc_name]) < top_k_per_doc:
                doc_results[doc_name].append(result)
        
        return doc_results
        
    
    def index_documents(self, force_rebuild: bool = False) -> None:
        """
        Index all PDF documents in the specified folder.
        
        Args:
            force_rebuild: If True, rebuild the vector store even if it exists
        """
        # Check if vector store already exists
        if not force_rebuild and self.load_vector_store():
            logger.info("Using existing vector store")
            return
        
        logger.info(f"Indexing documents from {self.pdf_folder}")
        
        try:
            # Check if PDF folder exists
            if not self.pdf_folder.exists():
                logger.error(f"PDF folder {self.pdf_folder} does not exist")
                raise FileNotFoundError(f"PDF folder {self.pdf_folder} does not exist")
            
            # Get list of PDF files
            pdf_files = list(self.pdf_folder.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.pdf_folder}")
                return
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Process each PDF file
            all_documents = []
            for pdf_file in pdf_files:
                try:
                    # Load and process each PDF
                    documents = self.load_pdf(pdf_file)
                    splits = self.split_documents(documents)
                    all_documents.extend(splits)
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                    logger.info("Continuing with next file...")
                    continue
            
            if not all_documents:
                logger.warning("No documents were successfully processed")
                return
            
            # Create and save vector store
            self.create_vector_store(all_documents)
            self.save_vector_store()
            
            logger.info(f"Successfully indexed {len(all_documents)} document chunks from {len(pdf_files)} PDFs")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

def main():
    """Main function to demonstrate usage"""
    try:
        # Initialize the indexer with parameters
        indexer = PDFIndexer(
            pdf_folder="/content/drive/MyDrive/Datasets/pdf-excels/data/pdfs",
            vector_store_path="vector_store",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Index documents
        indexer.index_documents()
        
        # Example query
        results = indexer.query("What are the main topics discussed in these documents?", top_k=5)
        
        # Print results
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Document: {result['metadata']['file_name']}, Page: {result['metadata']['page']}, Paragraph: {result['metadata']['paragraph']}")
            print(f"   Score: {result['similarity_score']:.4f}")
            print(f"   Excerpt: {result['content'][:150]}...")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

