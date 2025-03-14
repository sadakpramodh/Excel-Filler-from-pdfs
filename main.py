#!/usr/bin/env python
"""
Project: Tender Document Mapping with Smolagents & GPT-4
Description:
  This script implements a multi-agent system that:
    - Recursively reads all PDF files from a specified folder.
    - Extracts detailed text (with file, page, paragraph, and line metadata) from each PDF.
    - Reads an Excel template containing a "Parameters" column.
    - For each parameter, searches the extracted PDF data for all occurrences and tags each with a detailed reference.
    - Uses OpenAI GPT-4 to rank the occurrences.
    - Updates the Excel file with the ranked details and document references.
    - Saves the final Excel file for review.
    
Agents:
  A. Document Extraction Agent:
     - Recursively reads PDFs from a given folder.
     - Extracts text, splits pages into paragraphs and lines, and records metadata.
     
  B. Excel Reader Agent:
     - Reads the Excel template file.
     
  C. Parameter Mapping Agent:
     - Searches for each parameter in the detailed PDF records.
     - Collects all occurrences and detailed references.
     
  D. Ranking Agent:
     - Uses GPT-4 to rank occurrences for each parameter.
     
  E. Excel Filler Agent:
     - Updates the Excel DataFrame with the mapped values and references.
     
  F. Review & Iteration Agent:
     - Saves the updated Excel file.
     
Configuration and API keys are stored in .env and config.json.
     
Requirements:
    - Python packages: smolagents, pandas, openpyxl, python-dotenv, PyPDF2, openai
Author: Your Name
Date: 2025-03-14
"""

import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
import openai

# Import smolagents components:
from smolagents import CodeAgent, tool

from transformers import pipeline

class LocalLLMModel:
    def __init__(self, model_id, temperature=0.2):
        self.temperature = temperature
        self.pipeline = pipeline("text-generation", model=model_id, tokenizer=model_id)

    def __call__(self, messages, temperature=None):
        # Convert messages into a single prompt string.
        prompt = "\n".join([msg["content"] for msg in messages])
        response = self.pipeline(prompt, max_length=500, temperature=temperature or self.temperature)
        return {"content": response[0]["generated_text"]}



# -----------------------------------------------------------------------------
# 1. Load Environment Variables and Configuration
# -----------------------------------------------------------------------------
load_dotenv()  # Load API keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in .env file.")

openai.api_key = OPENAI_API_KEY  # Set API key for OpenAI library

# Load configuration from config.json
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

PDF_FOLDER = config.get("pdf_folder")
EXCEL_PATH = config.get("excel_path")
OUTPUT_EXCEL_PATH = config.get("output_excel_path")
MODEL_ID = config.get("model_id")

# Use Local Llama Model Instead of OpenAI API
model = LocalLLMModel(MODEL_ID, temperature=0.2)

# -----------------------------------------------------------------------------
# Utility: Recursively list all PDF files in a folder
# -----------------------------------------------------------------------------
def get_pdf_files(pdf_folder):
    """
    Recursively traverse the given folder and return a list of full paths for all PDF files.
    """
    pdf_files = []
    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# -----------------------------------------------------------------------------
# 2. Document Extraction Agent
# -----------------------------------------------------------------------------
class DocumentExtractionAgent:
    """
    Extracts detailed text from PDF files.
    For each PDF, the agent reads page by page, splits text into paragraphs (using double newlines)
    and then splits each paragraph into lines.
    Each record includes:
      - 'page': Page number (1-indexed)
      - 'paragraph': Paragraph number (1-indexed)
      - 'line': Line number (1-indexed)
      - 'text': Text content of the line
      - 'full_paragraph': The complete paragraph (for context)
    """
    def __init__(self, pdf_files):
        self.pdf_files = pdf_files
        self.pdf_details = {}  # {filename: [record, ...]}

    def extract_detailed_text_from_pdf(self, pdf_path):
        from PyPDF2 import PdfReader
        detailed_records = []
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                page_number = i + 1
                text = page.extract_text()
                if not text:
                    continue
                # Split page text into paragraphs
                paragraphs = text.split("\n\n")
                for p_index, para in enumerate(paragraphs):
                    para = para.strip()
                    if not para:
                        continue
                    # Split paragraph into lines
                    lines = para.split("\n")
                    for l_index, line in enumerate(lines):
                        record = {
                            "page": page_number,
                            "paragraph": p_index + 1,
                            "line": l_index + 1,
                            "text": line.strip(),
                            "full_paragraph": para
                        }
                        detailed_records.append(record)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        return detailed_records

    def run(self):
        for file_path in self.pdf_files:
            filename = os.path.basename(file_path)
            self.pdf_details[filename] = self.extract_detailed_text_from_pdf(file_path)
        return self.pdf_details

# -----------------------------------------------------------------------------
# 3. Excel Template Reader Agent
# -----------------------------------------------------------------------------
class ExcelReaderAgent:
    """
    Reads the Excel template file which contains the mapping template.
    Expected columns: "Parameters", "Details", "Document reference".
    """
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = None

    def run(self):
        self.df = pd.read_excel(self.excel_path)
        return self.df

# -----------------------------------------------------------------------------
# 4. Parameter Mapping & Reference Tagging Agent
# -----------------------------------------------------------------------------
class ParameterMappingAgent:
    """
    For each parameter (from the Excel template), searches the detailed PDF records to extract
    all occurrences. Each occurrence is a tuple (snippet, reference) where the reference contains
    file name, page, paragraph, and line numbers.
    """
    def __init__(self, parameters, pdf_details):
        self.parameters = parameters
        self.pdf_details = pdf_details  # Detailed records from DocumentExtractionAgent
        self.mapping = {}  # {parameter: {"occurrences": [(snippet, reference), ...]}}

    def search_parameter_in_records(self, parameter, records, context_chars=100):
        """
        Search for a parameter in a list of detailed records.
        Returns a list of (snippet, reference) tuples.
        """
        matches = []
        pattern = re.compile(re.escape(parameter), re.IGNORECASE)
        for rec in records:
            text = rec["text"]
            if not text:
                continue
            for match in pattern.finditer(text):
                full_para = rec["full_paragraph"]
                start = max(match.start() - context_chars, 0)
                end = min(match.end() + context_chars, len(full_para))
                snippet = full_para[start:end].replace("\n", " ").strip()
                reference = f"Page {rec['page']} | Paragraph {rec['paragraph']} | Line {rec['line']}"
                matches.append((snippet, reference))
        return matches

    def run(self):
        for param in self.parameters:
            all_occurrences = []
            for pdf_filename, records in self.pdf_details.items():
                occs = self.search_parameter_in_records(param, records)
                for snippet, ref in occs:
                    all_occurrences.append((snippet, f"{pdf_filename} - {ref}"))
            self.mapping[param] = {"occurrences": all_occurrences}
        return self.mapping

# -----------------------------------------------------------------------------
# 5. Ranking Agent (Using OpenAI GPT-4 via smolagents)
# -----------------------------------------------------------------------------
class RankingAgent:
    """
    Uses OpenAI GPT-4 (via HfApiModel) to rank occurrences for each parameter.
    It prompts GPT-4 to rank the occurrences by relevance and then returns the ranked list.
    """
    def __init__(self, mapping, model, temperature=0.2):
        self.mapping = mapping
        self.model = model
        self.temperature = temperature

    def llm_rank_occurrences(self, param, occurrences):
        prompt = f"""Rank the following occurrences for the parameter "{param}" by relevance.
Return the ranked list in JSON format with key "ranked_occurrences". Each element should be an object with keys "snippet" and "reference".
Occurrences:
"""
        for i, (snippet, reference) in enumerate(occurrences):
            prompt += f"{i+1}. Snippet: {snippet} (Reference: {reference})\n"
        prompt += "\nReturn the JSON now."
        try:
            messages = [
                {"role": "system", "content": "You are an expert in ranking textual information by relevance."},
                {"role": "user", "content": prompt}
            ]
            response = self.model(messages, temperature=self.temperature)
            content = response["content"]
            data = json.loads(content)
            ranked_occurrences = data.get("ranked_occurrences", occurrences)
            return [(item["snippet"], item["reference"]) for item in ranked_occurrences]
        except Exception as e:
            print(f"LLM ranking failed for parameter '{param}': {e}")
            return occurrences

    def run(self):
        for param, data in self.mapping.items():
            occurrences = data.get("occurrences", [])
            if occurrences:
                ranked = self.llm_rank_occurrences(param, occurrences)
                combined_value = "\n\n".join([f"{i+1}. {s}" for i, (s, _) in enumerate(ranked)])
                combined_ref = "\n\n".join([f"{i+1}. {r}" for i, (_, r) in enumerate(ranked)])
                self.mapping[param]["value"] = combined_value
                self.mapping[param]["reference"] = combined_ref
            else:
                self.mapping[param]["value"] = "Not found"
                self.mapping[param]["reference"] = "Not found"
        return self.mapping

# -----------------------------------------------------------------------------
# 6. Excel Filler Agent
# -----------------------------------------------------------------------------
class ExcelFillerAgent:
    """
    Updates the Excel DataFrame with the mapped values and document references.
    It populates the "Details" and "Document reference" columns based on the mapping.
    """
    def __init__(self, df, mapping):
        self.df = df
        self.mapping = mapping

    def run(self):
        details_col = "Details"
        reference_col = "Document reference"
        for idx, row in self.df.iterrows():
            param = str(row["Parameters"]).strip()
            if param in self.mapping:
                self.df.at[idx, details_col] = self.mapping[param].get("value", "Not found")
                self.df.at[idx, reference_col] = self.mapping[param].get("reference", "Not found")
        return self.df

# -----------------------------------------------------------------------------
# 7. Review & Iteration Agent
# -----------------------------------------------------------------------------
class ReviewAgent:
    """
    Saves the updated Excel DataFrame to a file for review.
    """
    def __init__(self, df, output_path):
        self.df = df
        self.output_path = output_path

    def run(self):
        self.df.to_excel(self.output_path, index=False)
        print(f"Updated Excel file saved to {self.output_path}")
        return self.output_path

# -----------------------------------------------------------------------------
# 8. Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    # Recursively get all PDF files from the PDF folder.
    def get_pdf_files(pdf_folder):
        pdf_files = []
        for root, dirs, files in os.walk(pdf_folder):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    pdf_files = get_pdf_files(PDF_FOLDER)
    print(f"Found {len(pdf_files)} PDF files.")

    # Agent A: Document Extraction
    doc_agent = DocumentExtractionAgent(pdf_files)
    print("Extracting detailed text from PDFs...")
    pdf_details = doc_agent.run()

    # Agent B: Excel Reader
    excel_agent = ExcelReaderAgent(EXCEL_PATH)
    print("Reading Excel template...")
    df = excel_agent.run()

    # Extract parameters (assumes "Parameters" column exists)
    parameters = df["Parameters"].dropna().tolist()

    # Agent C: Parameter Mapping & Reference Tagging
    mapping_agent = ParameterMappingAgent(parameters, pdf_details)
    print("Mapping parameters across PDFs...")
    mapping = mapping_agent.run()

    # Instantiate the LLM model using HfApiModel with the configured model ID
    model = HfApiModel(model_id=MODEL_ID)

    # Agent D: Ranking Agent using GPT-4
    ranking_agent = RankingAgent(mapping, model, temperature=0.2)
    print("Ranking occurrences using GPT-4...")
    ranked_mapping = ranking_agent.run()

    # Agent E: Excel Filler
    filler_agent = ExcelFillerAgent(df, ranked_mapping)
    print("Updating Excel template with mapping data...")
    updated_df = filler_agent.run()

    # Agent F: Review Agent (Save the updated Excel file)
    review_agent = ReviewAgent(updated_df, OUTPUT_EXCEL_PATH)
    review_agent.run()

    print("All agents have completed processing.")

if __name__ == "__main__":
    main()
