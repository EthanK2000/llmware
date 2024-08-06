from llmware.library import Library
from llmware.parsers import Parser
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import urlparse, parse_qs
import requests
import os
import argparse

from dotenv import load_dotenv

load_dotenv()

def create_country_library(country: str):
    client = MongoClient(os.environ['MONGODB_ATLAS_URI'], server_api=ServerApi('1'))
    database = client["VisaInfo-Database"]
    collection = database["VisaInfo"]
    countries = []
    if country == "All":
        countries = collection.find({})
    else:
        countries = collection.find({"country" : country})
    for country in countries:
        travel_country_library = country["country"].replace(" ","_")
        library_name = f"VisaInfo_{travel_country_library}_Library"
        try:
            Library().delete_library(library_name, confirm_delete=True)
            print(f"Deleting and recreating new {library_name}...")
        except:
            print(f"Creating new {library_name}...")
        library = Library().create_new_library(library_name)
        for url in country["visa_info_urls"]:
            urlstring = url["url"]
            print(f"Parsing {urlstring}")
            if urlstring.endswith(".pdf"):
                pdfpath = parse_online_pdf(travel_country_library, urlstring)
                library.add_pdf(pdfpath)
            else:
                Parser(library).parse_website(urlstring, write_to_db=True, get_links=False, attrs=url["attrs"])

        print("Installing embeddings...")
        library.install_new_embedding(embedding_model_name=os.environ['OPEN_AI_EMBEDDING_MODEL'], model_api_key=os.environ['OPENAI_API_KEY'])
    return 1

def parse_online_pdf(country, url):
    parsed_url = urlparse(url)
    filename = parse_qs(parsed_url.query)['filename'][0]
    base_directory = "C:/Users/ethkr/OneDrive/Documents/GitHub/llmware/tmp/" + country + "/files"
    full_path_original = os.path.join(base_directory, filename)
    html = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
    if not os.path.exists(base_directory):
        os.makedirs(base_directory, exist_ok=True)
    f = open(full_path_original, 'wb')
    f.write(html.content)
    f.close()
    return base_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='Create Country Library',
                description='Call this script when a countries data sources have been updated to update the country\'s library.')
    parser.add_argument('country')
    args = parser.parse_args()
    country = args.country        
    print(f"Creating library for {country}...")
    create_country_library(country)
    print(f"Library created for {country}.")
    