import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[■]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str) -> list:
    """Tokenize text and remove stop words."""
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def extract_skills(text: str) -> list:
    """Extract skills from the text."""
    skills_list = [
    'python', 'java', 'c++', 'sql', 'javascript', 'ruby', 'php', 'swift', 'objective-c', 'rust', 'go', 'bash', 'perl', 'r', 'sas', 'matlab', 'julia', 'stata', 'spss',
    'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'django', 'flask', 'ruby on rails', 'laravel',
    'docker', 'kubernetes', 'git', 'linux', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 'puppet', 'chef', 'jenkins', 'travis ci', 'circleci', 'jira', 'confluence', 'salesforce', 'sap', 'oracle',
    'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'ibm db2', 'cobol', 'fortran', 'assembly', 'vhdl', 'verilog', 'fpga', 'arduino', 'raspberry pi',
    'hadoop', 'spark', 'scala', 'tableau', 'power bi', 'informatica', 'talend', 'mulesoft', 'etl', 'data warehousing', 'data engineering', 'big data',
    'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'h2o.ai', 'tpot', 'auto-sklearn', 'pyspark', 'sagemaker',
    'tensorflow', 'keras', 'pytorch', 'caffe', 'theano', 'mxnet', 'chainer', 'cntk', 'dl4j',
    'dialogflow', 'microsoft bot framework', 'ibm watson assistant', 'rasa', 'botpress', 'snatchbot', 'tars', 'chatfuel', 'manychat',
    'openai gpt', 'hugging face transformers', 'spacy', 'nltk', 'allennlp', 'stanford nlp', 'fasttext', 'automl', 'rapidminer', 'datarobot', 'alteryx',
    'jupyter notebook', 'google colab', 'kaggle kernels', 'azure machine learning', 'amazon sagemaker', 'ibm watson studio',
    'openai gym', 'stable baselines', 'rllib', 'unity ml-agents',
    'deepmind lab', 'dota 2 ai', 'tensorflow extended (tfx)', 'apache singa', 'vowpal wabbit', 'pycaret', 'optuna',
    'networking', 'cybersecurity', 'penetration testing', 'ethical hacking', 'blockchain', 'cryptography',
    'virtual reality', 'augmented reality', 'unity', 'unreal engine', 'game development', 'opencv', 'computer vision', 'robotics', 'autocad', 'solidworks',
    'ci/cd', 'communication', 'teamwork', 'problem-solving', 'adaptability', 'critical thinking', 'time management', 'leadership', 'creativity', 'emotional intelligence', 'negotiation', 'conflict resolution', 'decision making', 'attention to detail', 'analytical skills', 'interpersonal skills', 'organizational skills', 'presentation skills', 'strategic thinking', 'customer service', 'multitasking', 'self-motivation', 'work ethic', 'collaboration',
    'cloud computing', 'aws', 'azure', 'gcp', 'cloud architecture', 'serverless computing', 'cloud security', 'cloud migration',
    'internet of things (iot)', 'iot security', 'iot architecture', 'embedded systems', 'sensor networks', 'edge computing',
    'artificial intelligence', 'machine learning', 'deep learning', 'reinforcement learning', 'natural language processing', 'computer vision', 'speech recognition', 'ai ethics',
    'data science', 'data analysis', 'data visualization', 'data mining', 'big data', 'data engineering',
    'cybersecurity', 'network security', 'information security', 'application security', 'threat detection', 'incident response', 'forensics',
    'blockchain', 'cryptocurrency', 'smart contracts', 'decentralized applications (dapps)', 'distributed ledger technology',
    'devops', 'continuous integration', 'continuous deployment', 'infrastructure as code', 'monitoring and logging',
    'digital twins', 'simulation', 'predictive maintenance', 'digital manufacturing',
    'quantum computing', 'quantum algorithms', 'quantum cryptography', 'quantum machine learning',
    'autonomous systems', 'self-driving cars', 'drones', 'robotics', 'robot operating system (ros)',
    'biotechnology', 'genomics', 'proteomics', 'bioinformatics', 'synthetic biology',
    'fintech', 'mobile payments', 'cryptocurrencies', 'financial algorithms', 'regtech',
    'edtech', 'learning management systems', 'educational content creation', 'remote learning technologies',
    'healthtech', 'telemedicine', 'digital health records', 'wearable health tech',
    'cleantech', 'renewable energy', 'energy storage', 'sustainable technology',
    'martech', 'marketing automation', 'customer data platforms', 'digital advertising technologies',
    # Libraries and Frameworks for Specific Domains
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'statsmodels',  # Data Science
    'flask', 'django', 'fastapi',  # Web Development
    'keras', 'pytorch', 'tensorflow', 'theano', 'mxnet', 'caffe', 'chainer', 'cntk', 'dl4j',  # Deep Learning
    'nltk', 'spacy', 'gensim', 'transformers',  # NLP
    'opencv', 'dlib', 'scikit-image',  # Computer Vision
    'ansible', 'terraform', 'chef', 'puppet',  # DevOps
    'spark', 'hadoop', 'hive', 'pig', 'kafka',  # Big Data
    'selenium', 'cypress', 'puppeteer', 'webdriver',  # Testing
    'boto3', 'azure-sdk', 'google-cloud-sdk',  # Cloud
    'docker-compose', 'kubernetes', 'istio', 'prometheus', 'grafana',  # Containerization and Orchestration
    'hyperledger', 'web3.js', 'ethers.js',  # Blockchain
    'opencv', 'pcl', 'ros', 'gazebo', 'vrep'  # Robotics and Simulation
]


    text = preprocess_text(text)
    tokens = tokenize_text(text)
    skills = [token for token in tokens if token in skills_list]
    return list(set(skills))  # Remove duplicates

def compare_skills(resume_skills, jd_skills):
    """Compare skills from resume and job description."""
    matched_skills = list(set(resume_skills).intersection(jd_skills))
    missing_skills = list(set(jd_skills).difference(resume_skills))
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }

def calculate_similarity_score(matched_skills, missing_skills):
    """Calculate a similarity score between matched and missing skills."""
    total_skills = len(matched_skills) + len(missing_skills)
    if total_skills == 0:
        return 0
    score = len(matched_skills) / total_skills
    return score

def generate_feedback(matched_skills, missing_skills):
    """Generate feedback based on matched and missing skills."""
    feedback = {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": "Consider adding these skills to your resume or gaining experience in these areas to better align with the job requirements.",
        "similarity_score": calculate_similarity_score(matched_skills, missing_skills),
        "notes": [
            "The match similarity score reflects how well your skills align with the job description. A higher score indicates a closer match.",
            "Regularly update your skills and experience to stay competitive in the job market."
        ]
    }
    return feedback

def vectorize_text(texts):
    """Vectorize texts using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

def compute_cosine_similarity(vector1, vector2):
    """Compute cosine similarity between two vectors."""
    return cosine_similarity(vector1, vector2)[0][0]

def calculate_combined_similarity(resume_text: str, job_description_text: str) -> dict:
    """Calculate combined similarity based on skills, education, and experience."""
    resume_info = extract_information(resume_text)
    jd_info = extract_information(job_description_text)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description_text)
    
    skills_comparison = compare_skills(resume_skills, jd_skills)
    
    resume_prepared = prepare_text_for_comparison(resume_info)
    jd_prepared = prepare_text_for_comparison(jd_info)
    
    vectors = vectorize_text([resume_prepared, jd_prepared])
    
    similarity_score = compute_cosine_similarity(vectors[0], vectors[1])
    
    feedback = generate_feedback(skills_comparison["matched_skills"], skills_comparison["missing_skills"])
    
    return {
        "skills_comparison": skills_comparison,
        "similarity_score": similarity_score,
        "feedback": feedback
    }

# Helper functions for extracting information
def extract_information(text: str) -> dict:
    """Extract key information from resume or job description text."""
    skills_pattern = re.compile(r'TECHNICALSKILLS:\s*(.*?)(?=Experience:|Education:|$)', re.IGNORECASE | re.DOTALL)
    experience_pattern = re.compile(r'Experience:\s*(.*?)(?=Education:|$)', re.IGNORECASE | re.DOTALL)
    education_pattern = re.compile(r'Education:\s*(.*)', re.IGNORECASE | re.DOTALL)

    skills = skills_pattern.findall(text)
    experience = experience_pattern.findall(text)
    education = education_pattern.findall(text)

    return {
        "skills": skills,
        "experience": experience,
        "education": education
    }

def prepare_text_for_comparison(extracted_info: dict) -> str:
    """Prepare text for comparison by concatenating skills, experience, and education."""
    return f"Skills: {', '.join(extracted_info.get('skills', []))} " \
           f"Experience: {', '.join(extracted_info.get('experience', []))} " \
           f"Education: {', '.join(extracted_info.get('education', []))}"











# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Download NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')

# def preprocess_text(text: str) -> str:
#     """Clean and preprocess text."""
#     text = text.lower()
    
#     text = re.sub(r'[■]', '', text)
#     # Remove extra whitespace and line breaks
#     text = re.sub(r'\s+', ' ', text)
    
#     # Remove unnecessary punctuation
#     text = re.sub(r'[^\w\s,.-]', '', text)
    
#     # Remove redundant spaces
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def tokenize_text(text: str) -> list:
#     """Tokenize text and remove stop words."""
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word not in stop_words]
#     return filtered_words

# def extract_skills(text: str) -> list:
#     """Extract skills from the text."""
#     skills_list = [
#         # Programming Languages
#         'python', 'java', 'c++', 'sql', 'javascript', 'ruby', 'php', 'swift', 'objective-c', 'rust', 'go', 'bash', 'perl', 'r', 'sas', 'matlab', 'julia', 'stata', 'spss',

#         # Web Development
#         'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'django', 'flask', 'ruby on rails', 'laravel',

#         # Tools and Platforms
#         'docker', 'kubernetes', 'git', 'linux', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 'puppet', 'chef', 'jenkins', 'travis ci', 'circleci', 'jira', 'confluence', 'salesforce', 'sap', 'oracle',

#         # Databases
#         'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'ibm db2', 'cobol', 'fortran', 'assembly', 'vhdl', 'verilog', 'fpga', 'arduino', 'raspberry pi',

#         # Data Engineering and Big Data
#         'hadoop', 'spark', 'scala', 'tableau', 'power bi', 'informatica', 'talend', 'mulesoft', 'etl', 'data warehousing', 'data engineering', 'big data',

#         # Machine Learning Frameworks
#         'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'h2o.ai', 'tpot', 'auto-sklearn','pyspark','sagemaker',

#         # Deep Learning Frameworks
#         'tensorflow', 'keras', 'pytorch', 'caffe', 'theano', 'mxnet', 'chainer', 'cntk', 'dl4j',

#         # Chatbots
#         'dialogflow', 'microsoft bot framework', 'ibm watson assistant', 'rasa', 'botpress', 'snatchbot', 'tars', 'chatfuel', 'manychat',

#         # AI Platforms & Tools
#         'openai gpt', 'hugging face transformers', 'spacy', 'nltk', 'allennlp', 'stanford nlp', 'fasttext', 'automl', 'rapidminer', 'datarobot', 'alteryx',

#         # AI Development Environments
#         'jupyter notebook', 'google colab', 'kaggle kernels', 'azure machine learning', 'amazon sagemaker', 'ibm watson studio',

#         # Reinforcement Learning
#         'openai gym', 'stable baselines', 'rllib', 'unity ml-agents',

#         # Specialized AI Tools
#         'deepmind lab', 'dota 2 ai', 'tensorflow extended (tfx)', 'apache singa', 'vowpal wabbit', 'pycaret', 'optuna',

#         # Networking and Security
#         'networking', 'cybersecurity', 'penetration testing', 'ethical hacking', 'blockchain', 'cryptography',

#         # Miscellaneous
#         'virtual reality', 'augmented reality', 'unity', 'unreal engine', 'game development', 'opencv', 'computer vision', 'robotics', 'autocad', 'solidworks', 'business intelligence', 'data science', 'ai', 'natural language processing', 'speech recognition', 'reinforcement learning', 'time series analysis', 'biometrics',

#         # Software Testing
#         'selenium', 'junit', 'mocha', 'chai', 'jest', 'cypress', 'appium', 'uipath', 'blue prism', 'automation',

#         # Project Management and Methodologies
#         'project management', 'agile', 'scrum', 'kanban', 'lean', 'product management', 'business analysis',

#         # Design and Multimedia
#         'ux design', 'ui design', 'wireframing', 'prototyping', 'usability testing', 'a/b testing', 'growth hacking', 'seo', 'sem', 'content marketing', 'social media marketing', 'email marketing', 'digital marketing', 'brand management', 'copywriting', 'technical writing', 'creative writing', 'translation', 'localization', 'language skills',

#         # GIS and Spatial Analysis
#         'gis', 'arcgis', 'qgis', 'remote sensing', 'spatial analysis',

#         # Multimedia Tools
#         'photoshop', 'illustrator', 'indesign', 'after effects', 'premiere pro', '3d modeling', 'animation', 'graphics design',
#         'ci/cd'
#     ]

#     text = preprocess_text(text)
#     tokens = tokenize_text(text)
#     skills = [token for token in tokens if token in skills_list]
#     return list(set(skills))  # Remove duplicates

# def compare_skills(resume_skills, jd_skills):
#     """Compare skills from resume and job description."""
#     matched_skills = list(set(resume_skills).intersection(jd_skills))
#     missing_skills = list(set(jd_skills).difference(resume_skills))
#     return {
#         "matched_skills": matched_skills,
#         "missing_skills": missing_skills
#     }

# def calculate_similarity_score(matched_skills, missing_skills):
#     """Calculate a similarity score between matched and missing skills."""
#     total_skills = len(matched_skills) + len(missing_skills)
#     if total_skills == 0:
#         return 0
#     score = len(matched_skills) / total_skills
#     return score

# def generate_feedback(matched_skills, missing_skills):
#     """Generate feedback based on matched and missing skills."""
#     feedback = {
#         "matched_skills": matched_skills,
#         "missing_skills": missing_skills,
#         "recommendations": "Consider adding these skills to your resume or gaining experience in these areas to better align with the job requirements.",
#         "similarity_score": calculate_similarity_score(matched_skills, missing_skills),
#         "notes": [
#             "The match similarity score reflects how well your skills align with the job description. A higher score indicates a closer match.",
#             "Regularly update your skills and experience to stay competitive in the job market."
#         ]
#     }
    
#     return feedback



# import re
# import spacy
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Load spaCy model
# nlp = spacy.load('en_core_web_md')

# def preprocess_text(text: str) -> str:
#     """Clean and preprocess text."""
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     text = text.strip()  # Remove extra whitespace
#     return text

# def tokenize_text(text: str) -> list:
#     """Tokenize text and remove stop words."""
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word not in stop_words]
#     return filtered_words

# def extract_skills(text: str) -> list:
#     """Extract skills from the text."""
#     skills_list = [
#         # Skills list as defined previously
#         'python', 'java', 'c++', 'sql', 'javascript', 'ruby', 'php', 'swift', 'objective-c', 'rust', 'go', 'bash', 'perl', 'r', 'sas', 'matlab', 'julia', 'stata', 'spss',
#         'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'django', 'flask', 'ruby on rails', 'laravel',
#         'docker', 'kubernetes', 'git', 'linux', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 'puppet', 'chef', 'jenkins', 'travis ci', 'circleci', 'jira', 'confluence', 'salesforce', 'sap', 'oracle',
#         'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'ibm db2', 'cobol', 'fortran', 'assembly', 'vhdl', 'verilog', 'fpga', 'arduino', 'raspberry pi',
#         'hadoop', 'spark', 'scala', 'tableau', 'power bi', 'informatica', 'talend', 'mulesoft', 'etl', 'data warehousing', 'data engineering', 'big data',
#         'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'h2o.ai', 'tpot', 'auto-sklearn', 'pyspark', 'sagemaker',
#         'tensorflow', 'keras', 'pytorch', 'caffe', 'theano', 'mxnet', 'chainer', 'cntk', 'dl4j',
#         'dialogflow', 'microsoft bot framework', 'ibm watson assistant', 'rasa', 'botpress', 'snatchbot', 'tars', 'chatfuel', 'manychat',
#         'openai gpt', 'hugging face transformers', 'spacy', 'nltk', 'allennlp', 'stanford nlp', 'fasttext', 'automl', 'rapidminer', 'datarobot', 'alteryx',
#         'jupyter notebook', 'google colab', 'kaggle kernels', 'azure machine learning', 'amazon sagemaker', 'ibm watson studio',
#         'openai gym', 'stable baselines', 'rllib', 'unity ml-agents',
#         'deepmind lab', 'dota 2 ai', 'tensorflow extended (tfx)', 'apache singa', 'vowpal wabbit', 'pycaret', 'optuna',
#         'networking', 'cybersecurity', 'penetration testing', 'ethical hacking', 'blockchain', 'cryptography',
#         'virtual reality', 'augmented reality', 'unity', 'unreal engine', 'game development', 'opencv', 'computer vision', 'robotics', 'autocad', 'solidworks', 'business intelligence', 'data science', 'ai', 'natural language processing', 'speech recognition', 'reinforcement learning', 'time series analysis', 'biometrics',
#         'selenium', 'junit', 'mocha', 'chai', 'jest', 'cypress', 'appium', 'uipath', 'blue prism', 'automation',
#         'project management', 'agile', 'scrum', 'kanban', 'lean', 'product management', 'business analysis',
#         'ux design', 'ui design', 'wireframing', 'prototyping', 'usability testing', 'a/b testing', 'growth hacking', 'seo', 'sem', 'content marketing', 'social media marketing', 'email marketing', 'digital marketing', 'brand management', 'copywriting', 'technical writing', 'creative writing', 'translation', 'localization', 'language skills',
#         'gis', 'arcgis', 'qgis', 'remote sensing', 'spatial analysis',
#         'photoshop', 'illustrator', 'indesign', 'after effects', 'premiere pro', '3d modeling', 'animation', 'graphics design', 'ci/cd'
#     ]

#     text = preprocess_text(text)
#     tokens = tokenize_text(text)
#     skills = [token for token in tokens if token in skills_list]
#     return list(set(skills))  # Remove duplicates

# def compare_skills(resume_skills, jd_skills):
#     """Compare skills from resume and job description."""
#     matched_skills = list(set(resume_skills).intersection(jd_skills))
#     missing_skills = list(set(jd_skills).difference(resume_skills))
#     return {
#         "matched_skills": matched_skills,
#         "missing_skills": missing_skills
#     }

# def calculate_similarity_spacy(resume_text, job_description_text):
#     """Calculate similarity using spaCy."""
#     resume_doc = nlp(resume_text)
#     job_description_doc = nlp(job_description_text)
#     return resume_doc.similarity(job_description_doc)

# def generate_feedback(matched_skills, missing_skills):
#     """Generate feedback based on matched and missing skills."""
#     feedback = {
#         "matched_skills": matched_skills,
#         "missing_skills": missing_skills,
#         "recommendations": "Consider adding these skills to your resume or gaining experience in these areas to better align with the job requirements.",
#         "similarity_score": calculate_similarity_spacy(preprocess_text(resume_text), preprocess_text(job_description_text)),
#         "notes": [
#             "The match similarity score reflects how well your skills align with the job description. A higher score indicates a closer match.",
#             "Regularly update your skills and experience to stay competitive in the job market."
#         ]
#     }
    
#     return feedback

# # Example usage:
# resume_text = "Your resume text here."
# job_description_text = "Job description text here."

# resume_skills = extract_skills(resume_text)
# job_description_skills = extract_skills(job_description_text)

# skills_comparison = compare_skills(resume_skills, job_description_skills)
# similarity_score = calculate_similarity_spacy(resume_text, job_description_text)
# feedback = generate_feedback(skills_comparison['matched_skills'], skills_comparison['missing_skills'])

# print("Skills Comparison:", skills_comparison)
# print("Similarity Score (spaCy):", similarity_score)
# print("Feedback:", feedback)



