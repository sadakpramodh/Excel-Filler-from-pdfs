from lib.pdf_indexer import PDFIndexer
import logging
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