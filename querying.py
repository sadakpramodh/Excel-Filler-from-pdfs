# Import the PDFIndexer class
from lib.pdf_indexer import PDFIndexer
import pandas as pd
from collections import defaultdict
import logging
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Vector store and embeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # For future agent integration
# from langchain.schema import Document

# Set up logging if not already configured
logging.basicConfig(level=logging.INFO)

# Initialize the indexer with paths to your existing vector store
indexer = PDFIndexer(
    vector_store_path="vector_store",  # Path to your saved vector store
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create a subclass with the safe loading method
class SafePDFIndexer(PDFIndexer):
    def load_vector_store(self):
        """Load vector store with pickle deserialization allowed."""
        vector_store_index = self.vector_store_path / "index.faiss"
        if not vector_store_index.exists():
            print(f"Vector store not found at {self.vector_store_path}")
            return False
        
        print(f"Loading vector store from {self.vector_store_path}")
        try:
            from langchain_community.vectorstores import FAISS
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully")
            return True
                
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False

# Use the modified class
indexer = SafePDFIndexer(
    vector_store_path="vector_store",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Basic search function
def search_docs(query, top_k=5):
    print(f"\n📝 Searching for: '{query}'")
    results = indexer.query(query, top_k=top_k)
    
    # Group by document
    docs = defaultdict(list)
    for r in results:
        docs[r['metadata']['file_name']].append(r)
    
    print(f"✅ Found {len(results)} results from {len(docs)} documents\n")
    
    # Print results by document
    for doc_name, doc_results in docs.items():
        print(f"📄 DOCUMENT: {doc_name} ({len(doc_results)} matches)")
        print("-" * 80)
        
        for i, result in enumerate(doc_results):
            meta = result['metadata']
            print(f"  Match #{i+1} | Page: {meta['page']}/{meta['total_pages']} | Score: {result['similarity_score']:.4f}")
            print(f"  {result['content'][:200]}...")
            print()
        
    return results

# Search with document-grouped results
def search_by_document(query, top_k_per_doc=2, max_docs=100):
    """Search across all documents and return top results from each document."""
    # Get more results to ensure we have enough for each document
    all_results = indexer.query(query, top_k=50)
    
    # Group by document
    doc_results = defaultdict(list)
    for result in all_results:
        doc_name = result['metadata']['file_name']
        doc_results[doc_name].append(result)
    
    # Get top results per document
    organized_results = {}
    for doc_name, results in list(doc_results.items())[:max_docs]:
        organized_results[doc_name] = results[:top_k_per_doc]
    
    # Print organized results
    print(f"\n📝 Searching for: '{query}' (top {top_k_per_doc} results from up to {max_docs} documents)")
    print(f"✅ Found results in {len(organized_results)} documents\n")
    
    for doc_name, results in organized_results.items():
        print(f"📄 DOCUMENT: {doc_name}")
        print("-" * 80)
        
        for i, result in enumerate(results):
            meta = result['metadata']
            print(f"  Match #{i+1} | Page: {meta['page']}/{meta['total_pages']} | Score: {result['similarity_score']:.4f}")
            print(f"  {result['content'][:200]}...")
            print()
    
    return organized_results

# Compare search results
def analyze_results(results, query):
    """Convert search results to a DataFrame for analysis."""
    rows = []
    for r in results:
        rows.append({
            'file': r['metadata']['file_name'],
            'page': r['metadata']['page'],
            'paragraph': r['metadata']['paragraph'],
            'score': r['similarity_score'],
            'content': r['content'][:100] + "...",
            'query': query
        })
    return pd.DataFrame(rows)

# Try loading the vector store
if indexer.load_vector_store():
    # # Example 1: Basic search across all documents
    # print("\n===== EXAMPLE 1: STANDARD SEARCH =====")
    # results1 = search_docs("What are the main requirements for tender submission?", top_k=7)
    
    # Example 2: Search grouped by document
    print("\n===== EXAMPLE 2: SEARCH BY DOCUMENT =====")
    results2 = search_by_document("payment terms and conditions", top_k_per_doc=2, max_docs=100)
    
    # # Example 3: Compare different queries
    # print("\n===== EXAMPLE 3: MULTI-QUERY ANALYSIS =====")
    # queries = [
    #     "project timeline requirements",
    #     "technical specifications",
    #     "budget limitations"
    # ]
    
    # all_results = []
    # for query in queries:
    #     results = indexer.query(query, top_k=3)
    #     df = analyze_results(results, query)
    #     all_results.append(df)
    
    # combined_df = pd.concat(all_results)
    # print(combined_df)
    
    # # Example 4: Advanced document-specific search
    # print("\n===== EXAMPLE 4: DOCUMENT-SPECIFIC SEARCH =====")
    # # Get all unique document names
    # all_results = indexer.query("", top_k=100)  # Empty query to get a sample
    # doc_names = list(set(r['metadata']['file_name'] for r in all_results))
    
    # if doc_names:
    #     target_doc = doc_names[0]  # Use the first document as an example
    #     print(f"Searching specifically in document: {target_doc}")
        
    #     # Filter results to just this document
    #     results = indexer.query("contract terms", top_k=20)
    #     doc_results = [r for r in results if r['metadata']['file_name'] == target_doc]
        
    #     for i, result in enumerate(doc_results[:3]):  # Show top 3
    #         meta = result['metadata']
    #         print(f"\nMatch #{i+1} | Page: {meta['page']}/{meta['total_pages']} | Score: {result['similarity_score']:.4f}")
    #         print(f"{result['content'][:200]}...")
else:
    print("Failed to load vector store. You may need to index documents first.")