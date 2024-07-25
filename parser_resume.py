from docx import Document
import chardet
import pdfplumber
import re






def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_information(text: str) -> dict:
    """Extract key information from resume text."""
    skills_pattern = re.compile(r'Skills:\s*(.*)', re.IGNORECASE)
    experience_pattern = re.compile(r'Experience:\s*(.*)', re.IGNORECASE)
    education_pattern = re.compile(r'Education:\s*(.*)', re.IGNORECASE)

    skills = skills_pattern.findall(text)
    experience = experience_pattern.findall(text)
    education = education_pattern.findall(text)

    return {
        "skills": skills,
        "experience": experience,
        "education": education
    }

def load_text(file_path: str) -> str:
    """Load text from a file with automatic encoding detection.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        str: Contents of the file.
    """
    try:
        # Detect the file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Read the file with the detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return ""


def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    text = text.strip()  # Remove leading and trailing spaces
    return text



   
