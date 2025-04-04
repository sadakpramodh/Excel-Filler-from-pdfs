import streamlit as st
import pandas as pd
import os
import io
import tempfile
from pathlib import Path
import uuid
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader

# Set page configuration
st.set_page_config(page_title="Intelligent Data Extractor (CAD Files)​", layout="wide")

# Try to initialize OpenAI client based on available version
def get_openai_client(api_key):
    try:
        # First try the new client (OpenAI 1.0.0+)
        from openai import OpenAI
        return OpenAI(api_key=api_key), "new"
    except (ImportError, TypeError) as e:
        try:
            # Fall back to legacy client (pre-1.0.0)
            import openai
            openai.api_key = api_key
            return openai, "legacy"
        except ImportError:
            raise ImportError("Failed to initialize OpenAI client. Please check your installation.")

# Helper functions for different OpenAI client versions
def generate_questions_with_client(parameter, client, client_type):
    """Generate questions using the appropriate client type."""
    system_content = "You are a helpful assistant that generates questions."
    user_content = f"Generate 4 concise questions for the parameter '{parameter}'. Include one 'what' question, one 'when' question, one 'how' question, and one 'who' question. Format as a Python list."
    
    try:
        if client_type == "new":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        else:  # legacy
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate questions: {str(e)}"

def summarize_results_with_client(results, client, client_type):
    """Summarize the search results using the appropriate client type."""
    context = "\n\n".join([doc.page_content for doc in results])
    sources = [f"{doc.metadata['source']} (Page {doc.metadata['page']}, Paragraph {doc.metadata['paragraph']})" for doc in results]
    
    system_content = "You are a helpful assistant that summarizes information."
    user_content = f"Summarize the following information concisely:\n\n{context}"
    
    try:
        if client_type == "new":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.5,
                max_tokens=200
            )
            return response.choices[0].message.content, sources
        else:  # legacy
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.5,
                max_tokens=200
            )
            return response.choices[0].message.content, sources
    except Exception as e:
        return f"Failed to summarize results: {str(e)}", sources

# Define process_pdfs function with improved error handling and FAISS fallback
def process_pdfs(uploaded_pdfs, parameter_list, embedding_model, collection_name="parameter_search"):
    """Process PDFs and store chunks in vector database (ChromaDB or FAISS)."""
    # Create a temporary directory to store the PDFs
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    
    try:
        # Save uploaded PDFs to the temporary directory
        for pdf in uploaded_pdfs:
            pdf_path = os.path.join(temp_dir, pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())
            pdf_paths.append(pdf_path)
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Process each PDF
        documents = []
        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_documents = loader.load()
                
                # Add metadata for tracking source
                for i, doc in enumerate(pdf_documents):
                    # Extract page number from metadata (0-indexed in PyPDFLoader)
                    page_num = doc.metadata.get("page", 0) + 1
                    
                    # Assign paragraph number based on position in page
                    paragraph_num = i % 10 + 1  # Simplified paragraph numbering
                    
                    doc.metadata.update({
                        "source": pdf_name,
                        "page": page_num,
                        "paragraph": paragraph_num
                    })
                
                documents.extend(pdf_documents)
            except Exception as e:
                st.warning(f"Error processing {pdf_name}: {str(e)}")
                continue
                
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.warning("No text content could be extracted from the PDFs.")
            return None, temp_dir
            
        # Try to use ChromaDB first, with fallback to FAISS
        vectorstore = None
        try:
            # Try using Chroma
            from langchain.vectorstores import Chroma
            persist_directory = os.path.join(temp_dir, "chroma_db")
            
            # Ensure the persistence directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            st.success("Using ChromaDB for vector storage.")
        except Exception as e:
            st.warning(f"ChromaDB initialization failed: {str(e)}")
            st.info("Falling back to FAISS vector store...")
            
            # Fall back to FAISS
            try:
                from langchain.vectorstores import FAISS
                vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
                st.success("Using FAISS for vector storage.")
            except Exception as faiss_error:
                st.error(f"FAISS initialization also failed: {str(faiss_error)}")
                return None, temp_dir
            
        return vectorstore, temp_dir
        
    except Exception as e:
        st.error(f"Error in PDF processing: {str(e)}")
        st.error(traceback.format_exc())
        return None, temp_dir

def search_answers(parameter, vectorstore, num_results=3):
    """Search for answers in the vectorstore based on parameter."""
    try:
        results = vectorstore.similarity_search(parameter, k=num_results)
        return results
    except Exception as e:
        st.error(f"Error searching for answers: {str(e)}")
        return []

# Main app layout
st.title("Intelligent Data Extractor (CAD Files)​")

# Add debug mode toggle
with st.sidebar:
    st.header("Configuration")
    debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed error information")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.divider()
    
    st.header("Upload Files")
    excel_file = st.file_uploader("Upload Excel File with Parameters", type=["xlsx", "xls"])
    pdf_files = st.file_uploader("Upload PDF Files for Reference", type=["pdf"], accept_multiple_files=True)

# Main processing
if excel_file and pdf_files and openai_api_key:
    # Initialize OpenAI client with version handling
    try:
        client, client_type = get_openai_client(openai_api_key)
        st.success(f"Successfully initialized OpenAI client (Type: {client_type})")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        st.stop()
    
    # Process Excel file
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        st.stop()
    
    # Validate Excel columns
    required_columns = ["parameters", "description", "Reference"]
    if not any(col.lower() in [c.lower() for c in df.columns] for col in required_columns):
        st.error("Excel file must contain at least one of these columns: parameters, description, or Reference")
        st.stop()
    else:
        # Standardize column names (case-insensitive matching)
        column_mapping = {}
        for col in df.columns:
            if col.lower() == 'parameters':
                column_mapping[col] = 'parameters'
            elif col.lower() == 'description':
                column_mapping[col] = 'description'
            elif col.lower() == 'reference':
                column_mapping[col] = 'Reference'
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist, create if missing
        for required_col in required_columns:
            if required_col not in df.columns:
                df[required_col] = ""
        
        # Extract parameters
        parameters = df['parameters'].dropna().tolist()
        
        if not parameters:
            st.error("No parameters found in the Excel file. Please ensure the 'parameters' column contains values.")
            st.stop()
        
        # Process PDFs and build vector store
        with st.spinner("Processing PDFs and building vector store..."):
            try:
                vectorstore, temp_dir = process_pdfs(pdf_files, parameters, "sentence-transformers/all-MiniLM-L6-v2")
                if vectorstore is None:
                    st.error("Failed to create vector store. Please check your PDF files and try again.")
                    st.stop()
                st.success("PDFs processed and vector store created!")
            except Exception as e:
                st.error(f"Error during PDF processing: {str(e)}")
                if debug_mode:
                    st.error(traceback.format_exc())
                st.stop()
        
        # Process each parameter
        st.header("Processing Parameters")
        progress_bar = st.progress(0)
        
        results_df = df.copy()
        for i, parameter in enumerate(parameters):
            st.subheader(f"Parameter: {parameter}")
            
            # Generate questions
            with st.spinner(f"Generating questions for '{parameter}'..."):
                try:
                    questions = generate_questions_with_client(parameter, client, client_type)
                    st.write("Generated Questions:")
                    st.text(questions)
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    if debug_mode:
                        st.error(traceback.format_exc())
                    questions = "Failed to generate questions."
                    st.text(questions)
            
            # Search for answers
            with st.spinner(f"Searching for information about '{parameter}'..."):
                try:
                    search_results = search_answers(parameter, vectorstore)
                    
                    if search_results:
                        summary, references = summarize_results_with_client(search_results, client, client_type)
                        
                        # Find the row index for this parameter
                        param_indices = results_df[results_df['parameters'] == parameter].index
                        
                        if len(param_indices) > 0:
                            idx = param_indices[0]
                            results_df.at[idx, 'description'] = summary
                            results_df.at[idx, 'Reference'] = " | ".join(references)
                        
                        st.write("Summary:")
                        st.write(summary)
                        
                        st.write("References:")
                        for ref in references:
                            st.write(f"- {ref}")
                    else:
                        st.warning(f"No search results found for '{parameter}'")
                except Exception as e:
                    st.error(f"Error searching for answers: {str(e)}")
                    if debug_mode:
                        st.error(traceback.format_exc())
            
            progress_bar.progress((i + 1) / len(parameters))
        
        # Save results
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False)
            
            # Download button
            st.download_button(
                label="Download Processed Excel File",
                data=output.getvalue(),
                file_name="processed_parameters.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error saving results: {str(e)}")
            if debug_mode:
                st.error(traceback.format_exc())
        
        # Clean up
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            if debug_mode:
                st.warning(f"Error during cleanup: {str(e)}")
        
        st.success("Processing complete!")

else:
    st.info("Please upload an Excel file with parameters, PDF files for reference, and provide your OpenAI API key to begin processing.")
    
    # Display sample information
    st.header("How to Use This App")
    st.markdown("""
    1. **Upload an Excel file** with three columns:
       - parameters: List of parameters to generate questions for
       - description: Will be filled with summaries from PDF search
       - Reference: Will be filled with source references
       
    2. **Upload PDF files** that contain information related to the parameters
    
    3. **Enter your OpenAI API Key** in the sidebar
    
    4. The app will:
       - Generate questions for each parameter
       - Process PDFs and store them in a vector database
       - Search for relevant information for each parameter
       - Summarize the information and add it to the Excel file
       - Allow you to download the completed Excel file
    """)
    
    # Add requirements info
    st.header("Requirements")
    st.markdown("""
    This app requires the following Python packages:
    ```
    streamlit
    pandas
    openpyxl
    openai
    langchain
    langchain-community
    sentence-transformers
    chromadb (or faiss-cpu as fallback)
    pypdf
    ```
    
    Make sure these are installed in your environment.
    """)