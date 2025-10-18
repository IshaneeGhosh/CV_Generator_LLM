#Imports
import os, json, re, torch, glob
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from docx import Document
from huggingface_hub import login

# ==== Setup ====
# Automatically select GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ==== Argument parsing ====
# Creates a command-line interface for optional Hugging Face authentication token
parser = argparse.ArgumentParser()
parser.add_argument("--hf_token", required=False, help="Optional Hugging Face token for gated models")
args = parser.parse_args()

# If user provides a Hugging Face token, log in for access to private/gated models.
if args.hf_token:
    login(token=args.hf_token)
    print("Authenticated with Hugging Face.")
else:
    print("No Hugging Face token provided. Proceeding without authentication.")

#Model names
GEMMA = "google/gemma-2b-it"                 # extractor
LLAMA = "meta-llama/Llama-2-7b-chat-hf"      # generator
FALLBACK_GEN = "mistralai/Mistral-7B-Instruct-v0.2" #fallback model

#Load extractor (Gemma)
print("Loading Gemma (JSON extractor)...")
# Load tokenizer and model for Gemma
tokenizer_ex = AutoTokenizer.from_pretrained(GEMMA, use_fast=True)
model_ex = AutoModelForCausalLM.from_pretrained(
    GEMMA,
    torch_dtype=torch.float16,  # Use 16-bit precision to save VRAM
    device_map="auto"           # Automatically distribute across available devices
)
model_ex.eval() # Put the model in inference mode
print("Gemma extractor loaded.")

## Use bitsandbytes 4-bit quantization to reduce memory for large models like LLaMA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Lazy loading placeholders
tokenizer_llama = None
model_llama = None
llama_model_name = None

# ==== Lazy load generator ====
def load_generation_model(preferred=LLAMA, fallback=FALLBACK_GEN):
    """
    Loads a generation model (like LLaMA or Mistral) for CV text generation.
    Falls back to a secondary model if the preferred one fails.
    """
    global tokenizer_llama, model_llama, llama_model_name
    # If model already loaded, skip loading again
    if model_llama is not None:
        return True
    try:
        # Try loading preferred model first (LLaMA)
        print(f"Loading generation model: {preferred}")
        tokenizer_llama = AutoTokenizer.from_pretrained(preferred, use_fast=True)
        model_llama = AutoModelForCausalLM.from_pretrained(
            preferred,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model_llama.eval()
        llama_model_name = preferred
        print("LLaMA loaded successfully.")
        return True
    
    # If LLaMA loading fails, fallback to Mistral model
    except Exception as e:
        print(f"Failed to load preferred model ({preferred}): {e}")
        print("Falling back to open model...")
        tokenizer_llama = AutoTokenizer.from_pretrained(fallback, use_fast=True)
        model_llama = AutoModelForCausalLM.from_pretrained(
            fallback,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model_llama.eval()
        llama_model_name = fallback
        print("Fallback model loaded successfully.")
        return True

# ==== Prompt templates ====
# Template prompt for Gemma: used to extract structured JSON data from unstructured text (like resumes).
gemma_prompt_template = """
You are a data extraction assistant. Your task is to convert the following unstructured resume text into a clean, well-formed JSON object with these fields:

{{
  "personal_details": {{
    "name": "",
    "age": "",
    "email": "",
    "phone": "",
    "linkedin": "",
    "location":""
  }},
  "professional_summary": "",
  "work_experience": [
    {{
      "role": "",
      "company": "",
      "duration": "",
      "location": "",
      "achievements": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "skills": {{
    "technical": [],
    "soft": []
  }},
  "certifications": [],
  "projects": []
}}

Rules:
- Extract only what exists; leave missing fields empty or as empty arrays.
- Be concise and factual.
- Use text inference where needed (e.g., company names, job titles).
- Never include explanations or text outside the JSON.
-Always detect and extract any text following the words “Certification” or “Certifications”, and store it as a list of certification names.
- Output ONLY valid JSON.

Input text:
\"\"\"{raw_input}\"\"\"

"""
#Template prompt for LLaMA: used to convert the structured JSON back into a formatted CV text.
llama_prompt_template = """
You are a professional CV writer and ATS specialist. Using ONLY the structured JSON below, generate a clean, polished, and ATS-optimized CV.
Skip sections that have no data.
Formatting rules:
- Start with "CURRICULUM VITAE"
- PERSONAL DETAILS:
    Name:
    Age:
    Email:
    Phone:
    LinkedIn:
    Location:

- PROFESSIONAL SUMMARY: 2-3 concise sentences from 'professional_summary' or other fields
- WORK EXPERIENCE: reverse chronological order, each with Role — Company — Duration — Location, and 2–4 bullet points
- EDUCATION: each on its own line (Degree — Institution — Year)
- SKILLS: Technical Skills (comma-separated), Soft Skills (comma-separated)
- CERTIFICATIONS and PROJECTS: bullet lists if available
- Skip sections that have no data.
- Output ONLY the resume text (no markdown, no JSON, no explanation).

Structured JSON:
{json}

Resume text:
"""

# ==== Extraction ====
def extract_with_gemma(raw_input, max_new_tokens=512):
    """
    Extract structured information (JSON) from raw resume text using Gemma model.
    If parsing fails, fallback to minimal JSON.
    """    
    prompt = gemma_prompt_template.format(raw_input=raw_input)
    inputs = tokenizer_ex(prompt, return_tensors="pt", truncation=True).to(device)
    # Generate model output deterministically (temperature=0)
    with torch.no_grad():
        output = model_ex.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
    # Decode text output
    text = tokenizer_ex.decode(output[0], skip_special_tokens=True)

    # Try to extract JSON substring
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        # Fallback minimal JSON if extraction fails
        return {
            "personal_details": {"name":"", "age":"", "email":"", "phone":"", "linkedin":"", "location":""},
            "professional_summary": raw_input.strip()[:512],
            "work_experience": [], "education": [], "skills":{"technical":[],"soft":[]},
            "certifications": [], "projects": []
        }

#Function for CV generation
def generate_cv(structured_json, max_new_tokens=512):
    """
    Generate formatted CV text using structured JSON via LLaMA or fallback model.
    """    
    load_generation_model()
    prompt = llama_prompt_template.format(json=json.dumps(structured_json, ensure_ascii=False))
    inputs = tokenizer_llama(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        gen = model_llama.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=2)
    text = tokenizer_llama.decode(gen[0], skip_special_tokens=True).strip()

    # Keep only text after "Resume text:" in case the model repeats the prompt
    if "Resume text:" in text:
        text = text.split("Resume text:", 1)[1].strip()

    # Ensure "CURRICULUM VITAE" is first line
    if not re.match(r"^\s*CURRICULUM\s+VITAE", text, re.IGNORECASE):
        text = "CURRICULUM VITAE\n" + text.strip()

    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text

# Function to write CV to .docx
def write_docx(cv_text, path):
    doc = Document()
    for line in cv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Headings (section titles) are uppercase or end with a colon
        if line.isupper() or line.endswith(":"):
            doc.add_heading(line, level=1)
        # Bullet points    
        elif line.startswith("•") or line.startswith("-"):
            doc.add_paragraph(line.lstrip("•- ").strip(), style="List Bullet")
        # Normal paragraph text
        else:
            doc.add_paragraph(line)
    doc.save(path)

#function to process a single profile
def process_profile(json_path):
    print(f"\nProcessing: {json_path}")
    data = json.load(open(json_path))
    raw = data.get("raw_text", "")

    # Step 1: Extract information
    structured = extract_with_gemma(raw)
    print("Extraction complete.")

    # Step 2: Generate CV text
    cv_text = generate_cv(structured)
    print("CV generation complete.")

    # Step 3: Write CV to Word file
    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(json_path))[0] + "_CV.docx")
    write_docx(cv_text, out_path)
    print(f"Saved CV to: {out_path}")
    return out_path



if __name__ == "__main__":


    # ==== Directory structure ====
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory

    # No subdirectories - files in same directory
    PROFILES_DIR = BASE_DIR  # Profiles are in same directory as main.py
    OUT_DIR = BASE_DIR       # Output to same directory as main.py

    print(f"Profiles from: {BASE_DIR}")
    print(f"CVs will be saved to: {BASE_DIR}")

    #Search for all .json files (profiles) in current directory
    profile_files = glob.glob(os.path.join(PROFILES_DIR, "*.json"))
    print(f"Found {len(profile_files)} profiles to process...")
    
    # Process each JSON profile
    for profile_file in profile_files:
        process_profile(profile_file)
    
    print("CV generation completed for all profiles.")



