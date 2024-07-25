import requests

def search_jobs(role:list ,location:str):

    url = "https://linkedin-data-api.p.rapidapi.com/search-jobs"

    querystring = {"keywords":role,"locationId":location,"datePosted":"recent","sort":"mostRelevant"}

    headers = {
        "x-rapidapi-key": "250c405c9amsh550a4360d2fe6e0p1625ccjsn4e52826a4364",
        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.json())

search_jobs(["machine learning engineer","python","NLP"],"Hyderabad")    