CV Creation using LLMs
This project automates the creation of professional, ATS-optimized Curriculum Vitae (CVs) using Large Language Models (LLMs). It converts unstructured user profile data into formatted and coherent CVs in .docx format through a two-stage pipeline.


Overview
Stage 1 (Extraction): Extracts structured information (JSON) from unstructured text using Gemma-2B-IT.

Stage 2 (Generation): Generates a polished CV using LLaMA-2-7B-Chat (with Mistral-7B-Instruct as fallback).

Output: Clean, standardized, ATS-friendly CVs in .docx format.

Ten sample unstructured profiles were processed to produce ten complete CVs, demonstrating the capability of LLMs to automate document creation with precision and consistency.


Features
Automated extraction of personal, educational, and professional data
Generation of formatted CV text
Conversion to .docx file format
Support for multiple input profiles in batch mode
Automatic fallback to alternative model if preferred model fails


Install dependencies
pip install -r requirements.txt

Since gated models like LLaMA-2 and Gemma are being used a Hugging Face access token is needed.
python main.py --hf_token "hf_YOUR_TOKEN_HERE"


Usage
Run the execution.txt file, it will install dependencies and run the main.py

main.py will:
Load all .json files
Extract structured information
Generate formatted CVs
Save the final .docx files in the same directory



