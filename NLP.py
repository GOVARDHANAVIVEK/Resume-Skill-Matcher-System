# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.stem import WordNetLemmatizer
# # Download NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')

# def preprocess_text(text: str) -> str:
#     """Clean and preprocess text."""
#     text = text.lower()
#     text = re.sub(r'[■]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s,.-]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess_edu(text: str) -> str:
#     """Clean and preprocess educational qualifications text."""
#     text = text.lower()
#     text = re.sub(r'[■]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s,.-]', '', text)
#     # text = re.sub(r'\b(bachelor(?:\'s| of)?|b\.?sc|b\.?tech|b\.?eng|b\.?e|bachelor of science|bachelor of technology|bachelor of engineering)\b|\
#     #                 \b(master(?:\'s| of)?|m\.?sc|m\.?tech|m\.?eng|m\.?e|master of science|master of engineering)\b|\
#     #                 \b(ph\.?d|doctor(?:ate)?|doctoral)\b|\
#     #                 \b(computer science|information technology|software engineering|data science|artificial intelligence|machine learning|cybersecurity|\
#     #                 network engineering|electrical engineering|electronics engineering|systems engineering|robotics|mathematics|statistics|physics|bioinformatics|\
#     #                 informatics|computer engineering|web development|database management|cloud computing|big data|analytics|embedded systems|human-computer interaction|\
#     #                 software development|game development|mobile development)\b|\
#     #                 \b(bachelors|masters|phd|doctorate|doctoral)\b', '', text, flags=re.IGNORECASE).strip()
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
#         'virtual reality', 'augmented reality', 'unity', 'unreal engine', 'game development', 'opencv', 'computer vision', 'robotics', 'autocad', 'solidworks',
#         'ci/cd', 'communication', 'teamwork', 'problem-solving', 'adaptability', 'critical thinking', 'time management', 'leadership', 'creativity', 'emotional intelligence', 'negotiation', 'conflict resolution', 'decision making', 'attention to detail', 'analytical skills', 'interpersonal skills', 'organizational skills', 'presentation skills', 'strategic thinking', 'customer service', 'multitasking', 'self-motivation', 'work ethic', 'collaboration',
#         'cloud computing', 'aws', 'azure', 'gcp', 'cloud architecture', 'serverless computing', 'cloud security', 'cloud migration',
#         'internet of things (iot)', 'iot security', 'iot architecture', 'embedded systems', 'sensor networks', 'edge computing',
#         'artificial intelligence', 'machine learning', 'deep learning', 'reinforcement learning', 'natural language processing', 'computer vision', 'speech recognition', 'ai ethics',
#         'data science', 'data analysis', 'data visualization', 'data mining', 'big data', 'data engineering',
#         'cybersecurity', 'network security', 'information security', 'application security', 'threat detection', 'incident response', 'forensics',
#         'blockchain', 'cryptocurrency', 'smart contracts', 'decentralized applications (dapps)', 'distributed ledger technology',
#         'devops', 'continuous integration', 'continuous deployment', 'infrastructure as code', 'monitoring and logging',
#         'digital twins', 'simulation', 'predictive maintenance', 'digital manufacturing',
#         'quantum computing', 'quantum algorithms', 'quantum cryptography', 'quantum machine learning',
#         'autonomous systems', 'self-driving cars', 'drones', 'robotics', 'robot operating system (ros)',
#         'biotechnology', 'genomics', 'proteomics', 'bioinformatics', 'synthetic biology',
#         'fintech', 'mobile payments', 'cryptocurrencies', 'financial algorithms', 'regtech',
#         'edtech', 'learning management systems', 'educational content creation', 'remote learning technologies',
#         'healthtech', 'telemedicine', 'digital health records', 'wearable health tech',
#         'cleantech', 'renewable energy', 'energy storage', 'sustainable technology',
#         'martech', 'marketing automation', 'customer data platforms', 'digital advertising technologies',
#         # Libraries and Frameworks for Specific Domains
#         'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'statsmodels',  # Data Science
#         'flask', 'django', 'fastapi',  # Web Development
#         'keras', 'pytorch', 'tensorflow', 'theano', 'mxnet', 'caffe', 'chainer', 'cntk', 'dl4j',  # Deep Learning
#         'nltk', 'spacy', 'gensim', 'transformers',  # NLP
#         'opencv', 'dlib', 'scikit-image',  # Computer Vision
#         'ansible', 'terraform', 'chef', 'puppet',  # DevOps
#         'spark', 'hadoop', 'hive', 'pig', 'kafka',  # Big Data
#         'selenium', 'cypress', 'puppeteer', 'webdriver',  # Testing
#         'boto3', 'azure-sdk', 'google-cloud-sdk',  # Cloud
#         'docker-compose', 'kubernetes', 'istio', 'prometheus', 'grafana',  # Containerization and Orchestration
#         'hyperledger', 'web3.js', 'ethers.js',  # Blockchain
#         'opencv', 'pcl', 'ros', 'gazebo', 'vrep'  # Robotics and Simulation
#     ]
    
#     text = preprocess_text(text)
#     tokens = tokenize_text(text)
#     skills = [token for token in tokens if token in skills_list]
#     return list(set(skills))  # Remove duplicates

# def extract_education(text: str) -> list:
#     """
#     Extract educational qualifications from the text.

#     :param text: The text to extract qualifications from.
#     :return: List of identified educational qualifications.
#     """
#     text = preprocess_edu(text)
#     print(text)
#     return extract_education_qualifications(text)

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

# def vectorize_text(texts):
#     """Vectorize texts using TF-IDF."""
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     return tfidf_matrix

# def compute_skill_match_score(resume_skills_str, jd_skills_str):
#     """
#     Compute skill match score based on how many skills are matched with the job description.

#     :param resume_skills_str: String of skills in the resume, separated by commas.
#     :param jd_skills_str: String of skills in the job description, separated by commas.
#     :return: Skill match score as a ratio of matched skills to total JD skills.
#     """
#     resume_skills = set(map(str.strip, resume_skills_str.split(',')))
#     jd_skills = set(map(str.strip, jd_skills_str.split(',')))
    
#     matched_skills = list(resume_skills.intersection(jd_skills))
    
#     # Calculate the skill match score
#     score = len(matched_skills) / len(jd_skills) if jd_skills else 0
    
#     return score

# def extract_education_qualifications(text):
    
#    # Define patterns for educational qualifications
#     education_patterns = [
#         r'\b(bachelor(?:\'s| of)?|b\.?sc|b\.?tech|b\.?eng|b\.?e|bachelor of science|bachelor of technology|bachelor of engineering)\b',
#         r'\b(master(?:\'s| of)?|m\.?sc|m\.?tech|m\.?eng|m\.?e|master of science|master of engineering)\b',
#         r'\b(ph\.?d|doctor(?:ate)?|doctoral)\b',
#         r'\b(computer science|information technology|software engineering|data science|artificial intelligence|machine learning|cybersecurity|network engineering|electrical engineering|electronics engineering|systems engineering|robotics|mathematics|statistics|physics|bioinformatics|informatics|computer engineering|web development|database management|cloud computing|big data|analytics|embedded systems|human-computer interaction|software development|game development|mobile development)\b',
#         r'\b(bachelors|masters|phd|doctorate|doctoral)\b',
#         r'\bBachelor\s*of\s*Technology|Master\s*of\s*Technology|\bbacheloroftechnology|(Bachelor\s*of\s*[Science]*\s*in\s*[a-zA-z]*[a-zA-Z]*[^\s])\b|(\bBTech|\s*B-Tech|Bsc|Bcom\s*in\s*[a-zA-Z]*)'
#     ]
    
#     # Combine patterns into one regex pattern
#     combined_pattern = '|'.join(education_patterns)
    
#     # Find all matches in the text
#     matches = re.findall(combined_pattern, text, re.IGNORECASE)
    
#     # Flatten the matches and remove duplicates
#     flattened_matches = [item for sublist in matches for item in sublist if item]
    
#     # Tokenize and lemmatize the matches
#     lemmatized_matches = lemmatize_words(flattened_matches)
    
#     # Remove duplicates and clean up the matches
#     unique_matches = list(set(lemmatized_matches))
#     cleaned_matches = [match.strip().replace(' ', '') for match in unique_matches]
#     return cleaned_matches
#     # Initialize a set to hold unique words
    
# def lemmatize_words(words: list) -> list:
#     """Lemmatize words."""
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
#     return lemmatized_words   

# def compute_education_score(resume,job_description):
#     resume_str = set(map(str.strip, resume.split(',')))
#     jd_str= set(map(str.strip, job_description.split(',')))
    
#     matched_skills = list(resume_str.intersection(jd_str))
    
#     # Calculate the skill match score
#     score = len(matched_skills) / len(jd_str) if jd_str else 0
    
#     return score


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

def preprocess_edu(text: str) -> str:
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[■]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\bBachelor\s*of\s*Technology|Master\s*of\s*Technology|(Bachelor\s*of\s*[Science]*\s*in\s*[a-zA-z]*[a-zA-Z]*[^\s])\b|(\bBTech|\s*B-Tech|Bsc|Bcom\s*in\s*[a-zA-Z]*)','',text).strip()
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


def extract_education(text:str) ->list:
    education_list = [
    "bachelor's degree", "bsc", "btech", "b.eng", "b.e", "bachelor of science", "bachelor of engineering",
    "master's degree", "msc", "mtech", "m.eng", "m.e", "master of science", "master of engineering",
    "phd", "doctorate", "doctoral", "doctoral degree", "philosophy doctor", "ph.d",
    "computer science", "information technology", "software engineering", "data science",
    "artificial intelligence", "machine learning", "cybersecurity", "network engineering",
    "electrical engineering", "electronics engineering", "systems engineering", "robotics",
    "mathematics", "statistics", "physics", "applied mathematics", "quantitative analysis",
    "bioinformatics", "informatics", "computer engineering", "web development", "database management",
    "cloud computing", "big data", "analytics", "embedded systems", "human-computer interaction",
    "software development", "game development", "mobile development", "computer engineering"
]
    education_list =[l.strip() for l in education_list]
    
    text = preprocess_edu(text)
    print(text)
    return extract_education_qualifications(text)
    # tokens = tokenize_text(text)
    # print(f"tokens edu {tokens}")
    # education = [token for token in tokens if token in education_list]
    # print(f"edu text :{education}")
    # return list(set(education))



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

def compute_skill_match_score(resume_skills_str, jd_skills_str):
    """
    Compute skill match score based on how many skills are matched with the job description.

    :param resume_skills_str: String of skills in the resume, separated by commas.
    :param jd_skills_str: String of skills in the job description, separated by commas.
    :return: Skill match score as a ratio of matched skills to total JD skills.
    """
    # resume_skills = list(set(resume_skills_str.split()))
    # jd_skills = list(set(jd_skills_str.split()))
    
    # print(f"Resume Skills: {resume_skills}")
    # print(f"Job Description Skills: {jd_skills}")
    
    # matched_skills = list(resume_skills.intersection(jd_skills))
    # print(f"Matched Skills: {matched_skills}")
    
    # score = len(matched_skills) / len(jd_skills) if jd_skills else 0
    # print(f"Score: {score}")
    
    # return score

    resume_skills = set(map(str.strip, resume_skills_str.split()))
    jd_skills = set(map(str.strip, jd_skills_str.split()))
    
    print(f"Resume Skills: {resume_skills}")
    print(f"Job Description Skills: {jd_skills}")
    
    # Find matched skills
    matched_skills = list(resume_skills.intersection(jd_skills))
    print(f"Matched Skills: {matched_skills}")
    
    # Calculate the skill match score
    score = len(matched_skills) / len(jd_skills) if jd_skills else 0
    print(f"Score: {score}")
    
    return score






def extract_education_qualifications(text):
    """
    Extract educational qualifications from the given text.
    
    :param text: The job description text.
    :return: List of identified educational qualifications.
    """
    # Define patterns for educational qualifications
    education_patterns = [
        r'\b(bachelor(?:\'s| of)?|b\.?sc|b\.?tech|b\.?eng|b\.?e|bachelor of science|bachelor of technology|bachelor of engineering)\b',
        r'\b(master(?:\'s| of)?|m\.?sc|m\.?tech|m\.?eng|m\.?e|master of science|master of engineering)\b',
        r'\b(ph\.?d|doctor(?:ate)?|doctoral)\b',
        r'\b(computer science|information technology|software engineering|data science|artificial intelligence|machine learning|cybersecurity|network engineering|electrical engineering|electronics engineering|systems engineering|robotics|mathematics|statistics|physics|bioinformatics|informatics|computer engineering|web development|database management|cloud computing|big data|analytics|embedded systems|human-computer interaction|software development|game development|mobile development)\b',
        r'\b(bachelors|masters|phd|doctorate|doctoral)\b'
    ]
    
    # Combine patterns into one regex pattern
    combined_pattern = '|'.join(education_patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text, re.IGNORECASE)
    
    # Remove duplicates by converting to a set and back to a list
    unique_matches = list(set(matches))
    
    return unique_matches

