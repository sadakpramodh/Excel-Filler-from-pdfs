import logging
import time
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import pandas as pd
import numpy as np
import requests
from tqdm.notebook import tqdm  # For Jupyter notebook progress bars

# Assuming PDFIndexer is imported from your module
from pdf_indexer import PDFIndexer

class LLMAgentRAG:
    """
    An LLM-powered agent for enhanced Retrieval Augmented Generation with PDF documents.
    
    Features:
    - Query planning and decomposition using LLM
    - Advanced context retrieval from vector store
    - Multi-step reasoning for complex questions
    - Answer generation and synthesis
    - Evidence-based responses with document citations
    - Support for OpenAI, Anthropic, and local Llama models
    """
    
    def __init__(self, 
                 vector_store_path: str = "vector_store",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 llm_model: str = "gpt-3.5-turbo",  # or "claude-3-opus-20240229" or "llama-3.2-3b-instruct"
                 llama_model_path: Optional[str] = None,  # Path to local Llama model weights if using Llama
                 similarity_cutoff: float = 0.5,
                 max_results_per_query: int = 10,
                 max_tokens: int = 1000,
                 temperature: float = 0.3,
                 verbose: bool = True):
        """
        Initialize the LLM-powered RAG agent.
        
        Args:
            vector_store_path: Path to the vector store directory
            embedding_model_name: Name of the embedding model used
            openai_api_key: OpenAI API key (can also be set in env var OPENAI_API_KEY)
            anthropic_api_key: Anthropic API key (can also be set in env var ANTHROPIC_API_KEY)
            llm_model: LLM model to use (OpenAI, Anthropic, or local Llama models supported)
            llama_model_path: Path to local Llama model weights if using Llama
            similarity_cutoff: Minimum similarity score to include results (0-1)
            max_results_per_query: Maximum results to consider per query
            max_tokens: Maximum tokens in LLM response
            temperature: Temperature for LLM responses (0-1)
            verbose: Whether to print detailed progress information
        """
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model
        self.llama_model_path = llama_model_path
        self.similarity_cutoff = similarity_cutoff
        self.max_results_per_query = max_results_per_query
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Set API keys
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger("llm_agent_rag")
        
        # Initialize LLM-related components
        self._setup_llm()
        
        # Initialize the PDF indexer with safe loading
        self.indexer = self._create_safe_indexer()
        
        # Store metadata about indexed documents
        self.document_metadata = {}
        
        # Load vector store
        self._load_vector_store()
        
        # Analyze vector store content for metadata
        self._analyze_vector_store()
        
        # History of interactions
        self.history = []
    
    def _setup_llm(self):
        """Set up the LLM based on the specified model."""
        # Determine LLM provider
        if "gpt" in self.llm_model.lower():
            self.llm_provider = "openai"
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for GPT models. Set openai_api_key or OPENAI_API_KEY env var.")
                
        elif "claude" in self.llm_model.lower():
            self.llm_provider = "anthropic"
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key required for Claude models. Set anthropic_api_key or ANTHROPIC_API_KEY env var.")
                
        elif "llama" in self.llm_model.lower():
            self.llm_provider = "llama"
            
            # Set up the Llama model if using it
            if self.llm_provider == "llama":
                self._setup_llama()
                
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm_model}. Use OpenAI, Anthropic, or Llama models.")
    
    def _setup_llama(self):
        """Set up Llama model for local inference."""
        try:
            # Define model path if not provided
            if not self.llama_model_path and "3b" in self.llm_model.lower():
                # Default location for Llama 3.2 3B Instruct model
                self.llama_model_path = "meta-llama/Llama-3.2-3B-Instruct"
            
            self.logger.info(f"Setting up Llama model from {self.llama_model_path}")
            
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                # Check for CUDA availability
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.logger.info(f"Using device: {device}")
                
                # Use 8-bit quantization for faster loading and less memory usage
                self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_path)
                
                # For limited GPU memory, use quantization options
                if device == "cuda":
                    try:
                        import bitsandbytes
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.llama_model_path,
                            device_map="auto",
                            load_in_8bit=True,  # Use 8-bit quantization
                            torch_dtype=torch.float16
                        )
                    except ImportError:
                        self.logger.warning("bitsandbytes not available, using standard loading")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.llama_model_path,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.llama_model_path,
                        device_map="auto"
                    )
                
                self.llama_pipeline = pipeline(
                    "text-generation", 
                    model=self.model, 
                    tokenizer=self.tokenizer
                )
                
                self.logger.info("Llama model loaded successfully")
                
            except ImportError as e:
                self.logger.error(f"Failed to import required packages for Llama: {str(e)}")
                raise ImportError(f"To use Llama models, install transformers, torch, and bitsandbytes: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error setting up Llama model: {str(e)}")
            raise
    
    def _create_safe_indexer(self):
        """Create a PDFIndexer subclass with safe loading capabilities."""
        class SafePDFIndexer(PDFIndexer):
            def load_vector_store(self_indexer):
                """Load vector store with pickle deserialization allowed."""
                vector_store_index = Path(self_indexer.vector_store_path) / "index.faiss"
                if not vector_store_index.exists():
                    self.logger.warning(f"Vector store not found at {self_indexer.vector_store_path}")
                    return False
                
                self.logger.info(f"Loading vector store from {self_indexer.vector_store_path}")
                try:
                    from langchain_community.vectorstores import FAISS
                    self_indexer.vector_store = FAISS.load_local(
                        str(self_indexer.vector_store_path), 
                        self_indexer.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.logger.info("Vector store loaded successfully")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error loading vector store: {str(e)}")
                    return False
        
        # Create and return the safe indexer
        return SafePDFIndexer(
            vector_store_path=self.vector_store_path,
            embedding_model_name=self.embedding_model_name
        )
    
    def _load_vector_store(self):
        """Load the vector store and verify it's ready for querying."""
        if not self.indexer.load_vector_store():
            self.logger.error("Failed to load vector store. Please check the path and permissions.")
            raise ValueError(f"Vector store could not be loaded from {self.vector_store_path}")
        self.logger.info("Vector store loaded successfully")
    
    def _analyze_vector_store(self):
        """Analyze the vector store to gather metadata about indexed documents."""
        # Use an empty query to get a sample of documents
        try:
            # Get a sample of results to analyze metadata
            sample_results = self.indexer.query("document", top_k=100)
            
            # Extract document metadata
            doc_names = set()
            for result in sample_results:
                doc_name = result['metadata']['file_name']
                doc_names.add(doc_name)
                
                # Store document metadata if not already present
                if doc_name not in self.document_metadata:
                    self.document_metadata[doc_name] = {
                        'total_pages': result['metadata']['total_pages'],
                        'sample_content': result['content'][:100]
                    }
            
            self.logger.info(f"Found {len(doc_names)} documents in the vector store")
            for doc_name in doc_names:
                self.logger.info(f"Document: {doc_name}, Pages: {self.document_metadata[doc_name]['total_pages']}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing vector store: {str(e)}")
    
    def call_llm(self, 
                prompt: str, 
                system_message: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: User prompt for the LLM
            system_message: Optional system message
            temperature: Optional override for temperature
            max_tokens: Optional override for max tokens
            
        Returns:
            LLM response text
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.llm_provider == "openai":
            return self._call_openai(prompt, system_message, temp, tokens)
        elif self.llm_provider == "anthropic":
            return self._call_anthropic(prompt, system_message, temp, tokens)
        else:  # llama
            return self._call_llama(prompt, system_message, temp, tokens)
    
    def _call_openai(self, 
                    prompt: str, 
                    system_message: Optional[str] = None,
                    temperature: float = 0.3,
                    max_tokens: int = 1000) -> str:
        """Call OpenAI's API with the given prompt."""
        try:
            import openai
            
            # Set API key
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Call API
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("OpenAI Python package not installed. Install with pip install openai")
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _call_anthropic(self, 
                       prompt: str, 
                       system_message: Optional[str] = None,
                       temperature: float = 0.3,
                       max_tokens: int = 1000) -> str:
        """Call Anthropic's API with the given prompt."""
        try:
            import anthropic
            
            # Set API key
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # Prepare system and user messages
            system = system_message if system_message else "You are a helpful AI assistant."
            
            # Call API
            response = client.messages.create(
                model=self.llm_model,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.content[0].text
            
        except ImportError:
            raise ImportError("Anthropic Python package not installed. Install with pip install anthropic")
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            raise
    
    def _call_llama(self, 
                  prompt: str, 
                  system_message: Optional[str] = None,
                  temperature: float = 0.3,
                  max_tokens: int = 1000) -> str:
        """Call the local Llama model with the given prompt."""
        try:
            # Format prompt with system message for Llama models
            if system_message:
                formatted_prompt = f"<|system|>\n{system_message}\n<|user|>\n{prompt}\n<|assistant|>"
            else:
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            
            # Generate text
            result = self.llama_pipeline(
                formatted_prompt,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Extract the generated text
            generated_text = result[0]['generated_text']
            
            # Remove the prompt from the response
            response = generated_text[len(formatted_prompt):]
            
            # Clean up the response
            if "<|user|>" in response:
                response = response.split("<|user|>")[0].strip()
                
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[1].strip()
                
            if "<|endoftext|>" in response:
                response = response.split("<|endoftext|>")[0].strip()
                
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating text with Llama: {str(e)}")
            raise
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Use the LLM to decompose a complex query into simpler sub-queries.
        
        Args:
            query: The complex query to decompose
            
        Returns:
            List of simpler sub-queries
        """
        system_message = """
        You are an AI assistant that decomposes complex questions into simpler, atomic sub-questions.
        Your goal is to break down the main question into 2-5 specific, focused sub-questions
        that would help answer the original question when combined.
        
        Return your response as a numbered list with ONLY the sub-questions, nothing else.
        Each sub-question should be self-contained and focused on a specific aspect.
        """
        
        prompt = f"""
        Please decompose the following question into 2-5 specific sub-questions:
        
        Question: {query}
        
        Return ONLY the numbered list of sub-questions, nothing else.
        """
        
        try:
            response = self.call_llm(prompt, system_message)
            
            # Parse the response to extract sub-queries
            lines = response.strip().split('\n')
            sub_queries = []
            
            for line in lines:
                # Remove numbering and any other artifacts
                clean_line = line.strip()
                # Remove numbering formats like "1.", "1)", "[1]", etc.
                if clean_line and (clean_line[0].isdigit() or clean_line[0] in ['•', '-', '*']):
                    # Find the first non-digit, non-punctuation character
                    for i, char in enumerate(clean_line):
                        if char.isalpha():
                            clean_line = clean_line[i:].strip()
                            break
                
                # Only add non-empty lines that seem like actual questions
                if clean_line and len(clean_line) > 10:
                    sub_queries.append(clean_line)
            
            # Always include the original query
            if query not in sub_queries:
                sub_queries.insert(0, query)
                
            self.logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            self.logger.error(f"Error decomposing query: {str(e)}")
            # Fallback to simple decomposition
            return [query]
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        self.logger.info(f"Retrieving documents for query: '{query}'")
        
        # Decompose query into sub-queries using LLM
        sub_queries = self.decompose_query(query)
        
        if self.verbose:
            print(f"Searching with {len(sub_queries)} queries:")
            for i, q in enumerate(sub_queries):
                print(f"  {i+1}. {q}")
        
        # Retrieve documents for each sub-query
        all_results = []
        for sub_query in sub_queries:
            try:
                results = self.indexer.query(sub_query, top_k=self.max_results_per_query)
                
                # Filter by similarity threshold
                filtered_results = [r for r in results if r['similarity_score'] > self.similarity_cutoff]
                
                # Add source query to results
                for result in filtered_results:
                    result['source_query'] = sub_query
                    all_results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error retrieving for sub-query '{sub_query}': {str(e)}")
        
        # Deduplicate results
        unique_results = {}
        for result in all_results:
            content = result['content']
            
            # If we already have this content, keep the one with higher score
            if content in unique_results:
                if result['similarity_score'] > unique_results[content]['similarity_score']:
                    unique_results[content] = result
            else:
                unique_results[content] = result
        
        # Sort by score and limit to top_k
        sorted_results = sorted(
            unique_results.values(), 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )
        
        top_results = sorted_results[:top_k]
        self.logger.info(f"Retrieved {len(top_results)} unique relevant documents")
        
        return top_results
    
    def generate_response(self, 
                        query: str, 
                        context: List[Dict[str, Any]], 
                        with_citations: bool = True) -> str:
        """
        Generate a response using the LLM based on retrieved context.
        
        Args:
            query: The original query
            context: The retrieved document chunks
            with_citations: Whether to include citations in the response
            
        Returns:
            Generated response
        """
        # Format context for the LLM
        formatted_context = ""
        for i, doc in enumerate(context):
            doc_name = doc['metadata']['file_name']
            page = doc['metadata']['page']
            
            formatted_context += f"\n[{i+1}] From {doc_name}, Page {page}:\n{doc['content']}\n"
        
        # Create system message
        system_message = """
        You are a knowledgeable assistant that provides accurate, comprehensive answers based on 
        the document excerpts provided. Your goal is to directly answer the question using ONLY 
        the information in the provided document excerpts.
        
        Guidelines:
        1. Base your response SOLELY on the provided document excerpts.
        2. If the documents don't contain enough information to answer the question fully, acknowledge this limitation.
        3. Do NOT make up or infer information that isn't supported by the provided excerpts.
        4. Cite your sources using the format [X] where X is the number of the excerpt.
        5. Be precise, clear, and direct in your answer.
        6. Structure your response in a well-organized way.
        """
        
        if not with_citations:
            system_message = system_message.replace("4. Cite your sources using the format [X] where X is the number of the excerpt.\n", "")
        
        # Create user prompt
        prompt = f"""
        Question: {query}
        
        Please answer the question based on these document excerpts:
        {formatted_context}
        
        {"Please include citation numbers [X] after sentences that use information from source X." if with_citations else ""}
        """
        
        # Call LLM
        response = self.call_llm(prompt, system_message, temperature=0.3)
        
        return response
    
    def query(self, 
             query: str, 
             top_k: int = 8, 
             with_citations: bool = True, 
             with_evidence: bool = False) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve, generate, and return response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            with_citations: Whether to include citations in response
            with_evidence: Whether to include retrieved evidence in response
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        self.logger.info(f"Processing query: '{query}'")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Step 2: Generate response
        response = self.generate_response(query, retrieved_docs, with_citations)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Record this interaction
        interaction = {
            "query": query,
            "response": response,
            "retrieved_docs": retrieved_docs,
            "elapsed_time": elapsed_time
        }
        self.history.append(interaction)
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "elapsed_time": elapsed_time
        }
        
        if with_evidence:
            result["evidence"] = retrieved_docs
            
        self.logger.info(f"Query processed in {elapsed_time:.2f} seconds")
        return result
    
    def print_response(self, 
                      result: Dict[str, Any], 
                      show_evidence: bool = False, 
                      max_evidence: int = 3):
        """
        Print the query response in a formatted way.
        
        Args:
            result: Result dictionary from query()
            show_evidence: Whether to show the evidence used
            max_evidence: Maximum number of evidence items to show
        """
        print("\n" + "=" * 80)
        print(f"🔍 Query: {result['query']}")
        print("=" * 80)
        
        print("\n📝 Response:")
        print("-" * 80)
        print(result['response'])
        print("-" * 80)
        print(f"Generated in {result['elapsed_time']:.2f} seconds")
        
        if show_evidence and "evidence" in result:
            print("\n📚 Supporting Evidence:")
            print("-" * 80)
            
            evidence = result["evidence"][:max_evidence]
            for i, doc in enumerate(evidence):
                doc_name = doc['metadata']['file_name']
                page = doc['metadata']['page']
                
                print(f"[{i+1}] From {doc_name}, Page {page} (Score: {doc['similarity_score']:.4f})")
                print(f"    {doc['content'][:200]}...")
                print()
    
    def get_document_list(self) -> List[str]:
        """Get a list of all documents in the vector store."""
        return list(self.document_metadata.keys())
    
    def explore_document(self, doc_name: str, page_limit: Optional[int] = None):
        """
        Explore the content of a specific document.
        
        Args:
            doc_name: Name of the document to explore
            page_limit: Maximum number of pages to explore (None for all)
        """
        if doc_name not in self.document_metadata:
            print(f"Document '{doc_name}' not found in the vector store.")
            return
            
        print(f"\n📄 Exploring Document: {doc_name}")
        print(f"Total Pages: {self.document_metadata[doc_name]['total_pages']}")
        print("-" * 80)
        
        # Get content from all pages of the document
        all_results = self.indexer.query("", top_k=1000)  # Empty query to get samples
        doc_results = [r for r in all_results if r['metadata']['file_name'] == doc_name]
        
        # Group by page
        pages = defaultdict(list)
        for r in doc_results:
            page_num = r['metadata']['page']
            pages[page_num].append(r)
            
        # Sort pages
        sorted_pages = sorted(pages.items())
        
        # Limit pages if requested
        if page_limit:
            sorted_pages = sorted_pages[:page_limit]
            
        # Print page samples
        for page_num, results in sorted_pages:
            print(f"\nPage {page_num}:")
            print("-" * 40)
            
            # Get the first chunk from the page as a sample
            if results:
                sample = results[0]['content']
                if len(sample) > 200:
                    sample = sample[:200] + "..."
                print(sample)
            else:
                print("(No content available)")
    
    def summarize_document(self, doc_name: str) -> str:
        """
        Generate a summary of a specific document using LLM.
        
        Args:
            doc_name: Name of the document to summarize
            
        Returns:
            Document summary
        """
        if doc_name not in self.document_metadata:
            return f"Document '{doc_name}' not found in the vector store."
            
        # Get content from the document
        all_results = self.indexer.query("", top_k=1000)  # Empty query to get samples
        doc_results = [r for r in all_results if r['metadata']['file_name'] == doc_name]
        
        # Sort by page and paragraph
        sorted_results = sorted(
            doc_results, 
            key=lambda x: (x['metadata']['page'], x['metadata'].get('paragraph', 0))
        )
        
        # Get a representative sample (first chunk from different pages)
        sample_chunks = []
        seen_pages = set()
        
        for result in sorted_results:
            page = result['metadata']['page']
            if page not in seen_pages and len(sample_chunks) < 5:
                sample_chunks.append(result['content'])
                seen_pages.add(page)
        
        # Join chunks
        document_sample = "\n\n".join(sample_chunks)
        
        # Create system message
        system_message = """
        You are an AI assistant specialized in document summarization. Your task is to provide a 
        concise yet comprehensive summary of the document excerpts provided. Focus on the key 
        topics, main points, and apparent purpose of the document.
        """
        
        # Create user prompt
        prompt = f"""
        Please provide a summary of this document based on these excerpts:
        
        Document: {doc_name}
        
        Content excerpts:
        {document_sample}
        
        Please provide:
        1. A concise 2-3 sentence overall summary
        2. The apparent document type and purpose
        3. Main topics or sections covered (based on these excerpts)
        4. A note acknowledging that this summary is based on limited excerpts
        """
        
        # Call LLM
        summary = self.call_llm(prompt, system_message)
        
        return summary
    
    def compare_documents(self, doc_names: List[str]) -> str:
        """
        Generate a comparison between multiple documents using LLM.
        
        Args:
            doc_names: List of document names to compare
            
        Returns:
            Comparison text
        """
        if len(doc_names) < 2:
            return "Need at least two documents to compare."
            
        # Get summaries for each document
        summaries = {}
        for doc_name in doc_names:
            if doc_name in self.document_metadata:
                summaries[doc_name] = self.summarize_document(doc_name)
            else:
                return f"Document '{doc_name}' not found in the vector store."
        
        # Format summaries for the prompt
        formatted_summaries = ""
        for doc_name, summary in summaries.items():
            formatted_summaries += f"\nDocument: {doc_name}\nSummary: {summary}\n"
        
        # Create system message
        system_message = """
        You are an AI assistant specialized in document analysis and comparison. Your task is to 
        compare multiple documents and identify similarities, differences, and relationships 
        between them. Provide a structured, detailed comparison that helps the user understand 
        how these documents relate to each other.
        """
        
        # Create user prompt
        prompt = f"""
        Please compare the following documents based on their summaries:
        {formatted_summaries}
        
        In your comparison, please include:
        1. Key similarities between the documents
        2. Notable differences in content, scope, or purpose
        3. How the documents might complement or contradict each other
        4. A concise overall assessment of their relationship
        
        Organize your comparison in a clear, structured format.
        """
        
        # Call LLM
        comparison = self.call_llm(prompt, system_message, max_tokens=1500)
        
        return comparison
    
    def answer_with_reasoning(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate a response with explicit reasoning steps.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with reasoned response
        """
        start_time = time.time()
        self.logger.info(f"Processing reasoned query: '{query}'")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Format context for the LLM
        formatted_context = ""
        for i, doc in enumerate(retrieved_docs):
            doc_name = doc['metadata']['file_name']
            page = doc['metadata']['page']
            
            formatted_context += f"\n[{i+1}] From {doc_name}, Page {page}:\n{doc['content']}\n"
        
        # Create system message for reasoned response
        system_message = """
        You are a knowledgeable assistant that provides accurate, well-reasoned answers based on 
        document excerpts. Your response should show your reasoning process step by step.
        
        Guidelines:
        1. Base your reasoning SOLELY on the provided document excerpts.
        2. First identify the key information needed to answer the question.
        3. Then analyze this information using explicit reasoning steps.
        4. Finally, provide a direct answer to the question.
        5. Cite your sources using the format [X] where X is the number of the excerpt.
        6. Be precise, clear, and logical in your reasoning.
        """
        
        # Create user prompt
        prompt = f"""
        Question: {query}
        
        Please answer the question based on these document excerpts, showing your reasoning process:
        {formatted_context}
        
        Structure your response in this format:
        
        RELEVANT INFORMATION:
        (Identify the key pieces of information from the documents that are relevant to answering this question)
        
        REASONING:
        (Show your step-by-step reasoning process)
        
        ANSWER:
        (Provide the direct answer to the question)
        
        Include citation numbers [X] after information from source X.
        """
        
        # Call LLM with higher max tokens for detailed reasoning
        response = self.call_llm(prompt, system_message, max_tokens=2000)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Record this interaction
        interaction = {
            "query": query,
            "response": response,
            "retrieved_docs": retrieved_docs,
            "elapsed_time": elapsed_time,
            "type": "reasoned"
        }
        self.history.append(interaction)
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "elapsed_time": elapsed_time,
            "evidence": retrieved_docs
        }
        
        self.logger.info(f"Reasoned query processed in {elapsed_time:.2f} seconds")
        return result
    
    def generate_report(self, topic: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report on a topic based on all available documents.
        
        Args:
            topic: The topic to create a report about
            
        Returns:
            Dictionary with report content and metadata
        """
        start_time = time.time()
        self.logger.info(f"Generating report on topic: '{topic}'")
        
        # Step 1: Retrieve a large set of relevant documents
        topic_query = f"comprehensive information about {topic}"
        retrieved_docs = self.retrieve(topic_query, top_k=20)
        
        # Step 2: Extract documents that have information about the topic
        if not retrieved_docs:
            return {
                "topic": topic,
                "report": f"Insufficient information available about '{topic}' in the indexed documents.",
                "elapsed_time": time.time() - start_time,
                "sources_used": []
            }
        
        # Format context for the LLM
        formatted_context = ""
        for i, doc in enumerate(retrieved_docs):
            doc_name = doc['metadata']['file_name']
            page = doc['metadata']['page']
            
            formatted_context += f"\n[{i+1}] From {doc_name}, Page {page}:\n{doc['content']}\n"
        
        # Create system message
        system_message = """
        You are an expert research assistant tasked with creating comprehensive, well-structured 
        reports on specific topics based on the provided document excerpts. Your report should 
        synthesize all relevant information into a cohesive, informative document.
        
        Guidelines:
        1. Use ONLY information from the provided excerpts.
        2. Organize the information in a logical structure with clear sections and headings.
        3. Provide a comprehensive overview while being concise and avoiding repetition.
        4. Cite sources using the format [X] where X is the excerpt number.
        5. Include an executive summary at the beginning.
        6. Use professional, clear language appropriate for a formal report.
        """
        
        # Create user prompt
        prompt = f"""
        Please create a comprehensive report on the topic of "{topic}" based on these document excerpts:
        
        {formatted_context}
        
        Your report should include:
        1. Executive Summary (brief overview of key findings)
        2. Introduction (context and scope)
        3. Main Sections (organize information logically by subtopics)
        4. Conclusion (synthesis of key points)
        
        Format the report with clear headings and subheadings. Include citation numbers [X] after 
        information from source X.
        """
        
        # Call LLM with higher max tokens for a detailed report
        report_content = self.call_llm(prompt, system_message, max_tokens=3000, temperature=0.2)
        
        # Extract sources used
        sources_used = set()
        for doc in retrieved_docs:
            doc_name = doc['metadata']['file_name']
            sources_used.add(doc_name)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Prepare result
        result = {
            "topic": topic,
            "report": report_content,
            "elapsed_time": elapsed_time,
            "sources_used": list(sources_used),
            "evidence": retrieved_docs
        }
        
        self.logger.info(f"Report generated in {elapsed_time:.2f} seconds")
        return result
    
    def answer_follow_up(self, follow_up_query: str, with_citations: bool = True) -> Dict[str, Any]:
        """
        Answer a follow-up question using the context of previous interactions.
        
        Args:
            follow_up_query: The follow-up query
            with_citations: Whether to include citations
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.history:
            # If no history, treat as regular query
            return self.query(follow_up_query, with_citations=with_citations)
        
        start_time = time.time()
        self.logger.info(f"Processing follow-up query: '{follow_up_query}'")
        
        # Get the most recent interaction
        last_interaction = self.history[-1]
        
        # Format previous interaction
        previous_context = f"""
        Previous question: {last_interaction['query']}
        
        Previous answer: {last_interaction['response']}
        """
        
        # Step 1: Retrieve relevant documents for the follow-up query
        new_docs = self.retrieve(follow_up_query, top_k=8)
        
        # Step 2: Combine with some documents from the previous query for continuity
        previous_docs = last_interaction.get('retrieved_docs', [])
        
        # Use a set to track unique content
        unique_content = set()
        combined_docs = []
        
        # First add new docs
        for doc in new_docs:
            content = doc['content']
            if content not in unique_content:
                combined_docs.append(doc)
                unique_content.add(content)
        
        # Then add some previous docs if they're not duplicates
        for doc in previous_docs[:3]:  # Limit to first 3 previous docs
            content = doc['content']
            if content not in unique_content:
                combined_docs.append(doc)
                unique_content.add(content)
        
        # Format context for the LLM
        formatted_context = ""
        for i, doc in enumerate(combined_docs):
            doc_name = doc['metadata']['file_name']
            page = doc['metadata']['page']
            
            formatted_context += f"\n[{i+1}] From {doc_name}, Page {page}:\n{doc['content']}\n"
        
        # Create system message
        system_message = """
        You are a knowledgeable assistant that provides accurate, helpful answers to follow-up questions.
        Consider the previous question and answer for context, but focus on directly addressing 
        the new follow-up question based on the provided document excerpts.
        
        Guidelines:
        1. Base your response SOLELY on the provided document excerpts.
        2. Consider the context from the previous Q&A, but don't repeat information unnecessarily.
        3. If the documents don't contain enough information to answer the follow-up fully, acknowledge this limitation.
        4. Cite your sources using the format [X] where X is the number of the excerpt.
        5. Be precise, clear, and direct in your answer.
        """
        
        if not with_citations:
            system_message = system_message.replace("4. Cite your sources using the format [X] where X is the number of the excerpt.\n", "")
        
        # Create user prompt
        prompt = f"""
        Previous interaction for context:
        {previous_context}
        
        Follow-up question: {follow_up_query}
        
        Please answer the follow-up question based on these document excerpts:
        {formatted_context}
        
        {"Please include citation numbers [X] after sentences that use information from source X." if with_citations else ""}
        """
        
        # Call LLM
        response = self.call_llm(prompt, system_message)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Record this interaction
        interaction = {
            "query": follow_up_query,
            "previous_query": last_interaction['query'],
            "response": response,
            "retrieved_docs": combined_docs,
            "elapsed_time": elapsed_time,
            "type": "follow_up"
        }
        self.history.append(interaction)
        
        # Prepare result
        result = {
            "query": follow_up_query,
            "response": response,
            "elapsed_time": elapsed_time,
            "previous_query": last_interaction['query']
        }
        
        if with_citations:
            result["evidence"] = combined_docs
            
        self.logger.info(f"Follow-up query processed in {elapsed_time:.2f} seconds")
        return result

