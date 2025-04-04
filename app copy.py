import streamlit as st
import pandas as pd
import os
import io
import tempfile
from pathlib import Path
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
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

# Define other functions
def process_pdfs(uploaded_pdfs, parameter_list, embedding_model, collection_name="parameter_search"):
    """Process PDFs and store chunks in ChromaDB."""
    # Create a temporary directory to store the PDFs
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    
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
    
    # Process each PDF and add to vector store
    documents = []
    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)
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
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    persist_directory = os.path.join(temp_dir, "chroma_db")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return vectorstore, temp_dir

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

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
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
        st.stop()
    
    # Process Excel file
    df = pd.read_excel(excel_file)
    
    # Validate Excel columns
    required_columns = ["parameters", "description", "Reference"]
    if not all(col.lower() in [c.lower() for c in df.columns] for col in required_columns):
        st.error("Excel file must contain columns: parameters, description, and Reference")
    else:
        # Standardize column names
        column_mapping = {col: 'parameters' for col in df.columns if col.lower() == 'parameters'}
        column_mapping.update({col: 'description' for col in df.columns if col.lower() == 'description'})
        column_mapping.update({col: 'Reference' for col in df.columns if col.lower() == 'reference'})
        df = df.rename(columns=column_mapping)
        
        # Extract parameters
        parameters = df['parameters'].dropna().tolist()
        
        # Process PDFs and build vector store
        with st.spinner("Processing PDFs and building vector store..."):
            vectorstore, temp_dir = process_pdfs(pdf_files, parameters, "sentence-transformers/all-MiniLM-L6-v2")
            st.success("PDFs processed and vector store created!")
        
        # Process each parameter
        st.header("Processing Parameters")
        progress_bar = st.progress(0)
        
        results_df = df.copy()
        for i, parameter in enumerate(parameters):
            st.subheader(f"Parameter: {parameter}")
            
            # Generate questions
            with st.spinner(f"Generating questions for '{parameter}'..."):
                questions = generate_questions_with_client(parameter, client, client_type)
                st.write("Generated Questions:")
                st.text(questions)
            
            # Search for answers
            with st.spinner(f"Searching for information about '{parameter}'..."):
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
            
            progress_bar.progress((i + 1) / len(parameters))
        
        # Save results
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
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
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
       - Process PDFs and store them in ChromaDB
       - Search for relevant information for each parameter
       - Summarize the information and add it to the Excel file
       - Allow you to download the completed Excel file
    """)