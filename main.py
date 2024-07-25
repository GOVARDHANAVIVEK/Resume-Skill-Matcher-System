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
from NLP import extract_skills,vectorize_text, compare_skills, generate_feedback, vectorize_text, compute_cosine_similarity,extract_information,prepare_text_for_comparison

app = FastAPI()

STATIC_DIR = Path("static")
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main():
    index_file = os.path.join("static", "frontend", "index.html")
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

        # Extract information
        resume_info = extract_information(resume_text)
        jd_info = extract_information(job_description_text)

        print(f"Resume Info: {resume_info}")
        print(f"Job Description Info: {jd_info}")

        # Extract and flatten skills
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description_text)
        resume_skills_text = ' '.join(resume_skills)
        jd_skills_text = ' '.join(jd_skills)

        print(f"Flattened Resume Skills: {resume_skills_text}")
        print(f"Flattened Job Description Skills: {jd_skills_text}")

        # Vectorize the flattened text
        vectors = vectorize_text([resume_skills_text, jd_skills_text])

        print(f"TF-IDF Vectors: {vectors}")

        # Compute cosine similarity
        similarity_matrix = compute_cosine_similarity(vectors[0:1], vectors[1:2])
        print(f"Similarity Matrix: {similarity_matrix}")

        # Handle similarity_matrix based on its type
        if isinstance(similarity_matrix, np.ndarray):
            similarity_score = similarity_matrix[0, 0] if similarity_matrix.size > 0 else None
        else:
            similarity_score = similarity_matrix  # Scalar value directly

        print(f"Similarity Score: {similarity_score}")

        # Compare skills
        skills_comparison = compare_skills(resume_skills, jd_skills)
        matched_skills = ','.join(skills_comparison["matched_skills"])
        missing_skills = ','.join(skills_comparison['missing_skills'])

        print(f"Matched Skills: {matched_skills}")
        print(f"Missing Skills: {missing_skills}")

        # Generate feedback
        feedback = generate_feedback(skills_comparison["matched_skills"], skills_comparison["missing_skills"])
        feedback["similarity_score"] = similarity_score

        return feedback

    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": "Error processing files."}

   









































# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import HTMLResponse, JSONResponse
# from pathlib import Path
# import os
# import aiofiles
# import uvicorn
# from fastapi.staticfiles import StaticFiles

# from username import set_name, get_name
# from parser_resume import extract_text_from_pdf, load_text, preprocess_text,extract_text_from_txt,extract_text_from_docx
# from NLP import extract_skills, compare_skills, generate_feedback
# app = FastAPI()

# STATIC_DIR = Path("static")
# UPLOAD_DIR = STATIC_DIR / "uploads"
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def main():
#     index_file = os.path.join("static", "frontend", "index.html")
#     async with aiofiles.open(index_file, 'r') as file:
#         content = await file.read()
#     return content

# @app.post("/upload")
# async def upload_file(resume: UploadFile = File(...), job_description: str = Form(...)):
#     try:
#         # Ensure the directory for the resume exists
#         name = resume.filename.split(".")[0]
#         set_name(name)

#         resume_dir = UPLOAD_DIR / name
#         resume_dir.mkdir(parents=True, exist_ok=True)

#         # Save the uploaded file
#         resume_loc = resume_dir / resume.filename
#         with open(resume_loc, "wb") as f:
#             f.write(resume.file.read())

#         # Save the job description
#         job_description_file = "job_description.txt"
#         job_description_loc = resume_dir / job_description_file
#         with open(job_description_loc, 'w') as f:
#             f.write(job_description)

#         # Process uploaded files
#         result = process_uploaded_files(resume.filename)

#         return JSONResponse(content=result)
    
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {e}"})

# def get_file_extension(file_path: str) -> str:
#     _, extension = os.path.splitext(file_path)
#     return extension.lower()

# def process_uploaded_files(filename):
#     name = get_name()
#     resume_path = os.path.join("static", "uploads", name)
#     job_description_text_path = os.path.join("static", "uploads", name, "job_description.txt")
#     extension = get_file_extension(filename)
#     if extension in ['.pdf']:
#         resume_text = extract_text_from_pdf(resume_path+f"\\{name}.pdf")
#     elif extension in ['.docx', '.doc']:
#         resume_text = extract_text_from_docx(resume_path+f"\\{name}.docx")
#     elif extension in ['.txt']:
#         resume_text = extract_text_from_txt(resume_path+f"\\{name}.txt")
#     else:
#         print("Unsupported file type.")

    
#     job_description_text = load_text(job_description_text_path)

#     processed_resume = preprocess_text(resume_text)
#     print(f"processed_resume : {processed_resume}")
#     processed_JD = preprocess_text(job_description_text)
#     print(processed_JD)

#     skills_resume = extract_skills(processed_resume)
#     skills_job_description = extract_skills(processed_JD)

#     skills_comparison = compare_skills(skills_resume, skills_job_description)

#     feedback = generate_feedback(skills_comparison["matched_skills"], skills_comparison["missing_skills"])
#     print(feedback)
#     return feedback

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)

















# # from fastapi import FastAPI, File, UploadFile, Form
# # from fastapi.responses import HTMLResponse, JSONResponse
# # from pathlib import Path
# # import uvicorn
# # from fastapi.staticfiles import StaticFiles
# # import os
# # import aiofiles
# # from username import set_name, get_name
# # from parser_resume import extract_text_from_pdf, load_text, preprocess_text
# # from NLP import extract_skills, compare_skills, generate_feedback, categorize_skills

# # app = FastAPI()

# # STATIC_DIR = Path("static")
# # UPLOAD_DIR = STATIC_DIR / "uploads"
# # UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# # app.mount("/static", StaticFiles(directory="static"), name="static")

# # @app.get("/", response_class=HTMLResponse)
# # async def main():
# #     index_file = os.path.join("static", "frontend", "index.html")
    
# #     async with aiofiles.open(index_file, 'r') as file:
# #         content = await file.read()
    
# #     return content

# # @app.post("/upload")
# # async def upload_file(resume: UploadFile = File(...), job_description: str = Form(...)):
# #     try:
# #         # Ensure the directory for the resume exists
# #         name = resume.filename.split(".")[0]
# #         set_name(name)

# #         resume_dir = UPLOAD_DIR / name
# #         resume_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

# #         # Save the uploaded file
# #         resume_loc = resume_dir / resume.filename
# #         with open(resume_loc, "wb") as f:
# #             f.write(resume.file.read())

# #         # Save the job description
# #         job_description_file = "job_description.txt"
# #         job_description_loc = resume_dir / job_description_file
# #         with open(job_description_loc, 'w') as f:
# #             f.write(job_description)

# #         # Process uploaded files
# #         result = process_uploaded_files()

# #         return JSONResponse(content=result)
    
# #     except Exception as e:
# #         # Log the exception
# #         print(f"Error occurred: {e}")
# #         # Return a HTTP 500 response with the error message
# #         return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {e}"})

# # def process_uploaded_files():
# #     name = get_name()
# #     resume_path = os.path.join("static", "uploads", name, f"{name}.pdf")
# #     job_description_text_path = os.path.join("static", "uploads", name, "job_description.txt")

# #     resume_text = extract_text_from_pdf(resume_path)
# #     job_description_text = load_text(job_description_text_path)

# #     processed_resume = preprocess_text(resume_text)
# #     processed_JD = preprocess_text(job_description_text)

# #     skills_resume = extract_skills(processed_resume)
# #     print(skills_resume)
# #     skills_job_description = extract_skills(processed_JD)

# #     skills_comparison = compare_skills(skills_resume, skills_job_description)

# #     matched_skills = categorize_skills(skills_comparison["matched_skills"])
# #     missing_skills = categorize_skills(skills_comparison["missing_skills"])

# #     feedback = generate_feedback(matched_skills, missing_skills)
# #     print(feedback)
# #     return feedback

# # if __name__ == "__main__":
# #     uvicorn.run(app, host="127.0.0.1", port=8000)
