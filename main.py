from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import os
import aiofiles
import uvicorn
import numpy as np
from fastapi.staticfiles import StaticFiles

from username import set_name, get_name
from parser_resume import extract_text_from_pdf, load_text, preprocess_text, extract_text_from_txt, extract_text_from_docx
from NLP import extract_skills,compute_skill_match_score, compare_skills, generate_feedback,extract_education

app = FastAPI()

STATIC_DIR = Path("static")
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main():
    index_file = os.path.join("static", "index.html")
    async with aiofiles.open(index_file, 'r') as file:
        content = await file.read()
    return content

@app.post("/upload")
async def upload_file(resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        # Ensure the directory for the resume exists
        name = resume.filename.split(".")[0]
        set_name(name)

        resume_dir = UPLOAD_DIR / name
        resume_dir.mkdir(parents=True, exist_ok=True)

        # Save the uploaded resume file
        resume_loc = resume_dir / resume.filename
        with open(resume_loc, "wb") as f:
            f.write(resume.file.read())

        # Save the job description
        job_description_file = "job_description.txt"
        job_description_loc = resume_dir / job_description_file
        with open(job_description_loc, 'w') as f:
            f.write(job_description)

        # Process uploaded files
        result = process_uploaded_files(resume.filename, job_description_loc)

        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {e}"})

def get_file_extension(file_path: str) -> str:
    """Get the file extension."""
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def process_uploaded_files(filename, job_description_path):
    name = get_name()
    resume_path = UPLOAD_DIR / name / filename
    extension = get_file_extension(resume_path)

    try:
        # Extract text from resume based on file type
        if extension == '.pdf':
            resume_text = extract_text_from_pdf(resume_path)
        elif extension in ['.docx', '.doc']:
            resume_text = extract_text_from_docx(resume_path)
        elif extension == '.txt':
            resume_text = extract_text_from_txt(resume_path)
        else:
            return {"error": "Unsupported file type."}

        # Load and preprocess job description
        job_description_text = preprocess_text(load_text(job_description_path))

       
        
        print(resume_text)
        # Extract and flatten skills
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description_text)

        education_in_resume = extract_education(resume_text)
        education_in_jd = extract_education(job_description_text)
        # print(education_in_resume,education_in_jd)
        print("education resume" ,education_in_resume)
        print("education jd" ,education_in_jd)
        resume_skills_text = ' '.join(resume_skills)
        jd_skills_text = ' '.join(jd_skills)

        print(f"Flattened Resume Skills: {resume_skills_text}")
        print(f"Flattened Job Description Skills: {jd_skills_text}")
       

        # resume_edu_text = ' '.join(education_in_resume)
        # jd_edu_text = ' '.join(education_in_jd)

        # print(f"Flattened Resume Skills: {resume_edu_text}")
        # print(f"Flattened Job Description Skills: {jd_edu_text}")
        # similarity_score_edu = compute_education_score(resume_edu_text,jd_edu_text)
        # print(similarity_score_edu)
        similarity_score = compute_skill_match_score(resume_skills_text,jd_skills_text)
        print(similarity_score)

        # Compare skills
        skills_comparison = compare_skills(resume_skills, jd_skills)
        matched_skills = ','.join(skills_comparison["matched_skills"])
        missing_skills = ','.join(skills_comparison['missing_skills'])

        print(f"Matched Skills: {matched_skills}")
        print(f"Missing Skills: {missing_skills}")

        # Generate feedback
        feedback = generate_feedback(skills_comparison["matched_skills"], skills_comparison["missing_skills"])
        feedback["similarity_score"] = similarity_score
        print(feedback["similarity_score"])
        return feedback

    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": "Error processing files."}

   


if __name__ == "__main__":
    uvicorn.run("0.0.0.0",port=3000)



