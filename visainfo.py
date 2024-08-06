from llmware.library import Library
from llmware.configs import LLMWareConfig
from llmware.retrieval import Query
from llmware.prompts import Prompt
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from openai import OpenAI
from flask import Flask, request
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder="/home/ubuntu/", static_folder="/home/ubuntu/")

LLMWareConfig().set_active_db("mongo")
LLMWareConfig().set_vector_db("milvus")

client = MongoClient(os.environ['MONGODB_ATLAS_URI'], server_api=ServerApi('1'))
clientOpenAI = OpenAI()
database = client["VisaInfo-Database"]

async def get_travel_country(q, travelcountry):
    ## Check where user wants to travel to
    print("Getting travel destination...")
    collection = database["Country"]
    prompter = Prompt().load_model(os.environ['RAG_MODEL_P1'],api_key=os.environ['OPENAI_API_KEY'])
    prompter.clear_source_materials()
    response = prompter.prompt_main(f"Which country is the user travelling to?", context=q, prompt_name="answer_or_not_found")
    text_embedding = response["llm_response"]
    if text_embedding == "Not Found.":
        if travelcountry == "":
            return None
        else:
            text_embedding = travelcountry
    embedding = clientOpenAI.embeddings.create(input=text_embedding, model=os.environ['OPEN_AI_EMBEDDING_MODEL']).data[0].embedding
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding_vector",
                "queryVector": embedding,
                "limit": 1,
                "numCandidates": 50,
                "exact": False
            }
        }
    ]
    results = collection.aggregate(pipeline)
    for result in results:
        return result
    return None

async def get_client_passport(q, nationality):
    print("Getting client passport...")
    collection = database["Country"]
    prompter = Prompt().load_model(os.environ['RAG_MODEL_P1'],api_key=os.environ['OPENAI_API_KEY'])
    prompter.clear_source_materials()
    response = prompter.prompt_main(f"What passport does the user have?", context=q, prompt_name="answer_or_not_found")
    text_embedding = response["llm_response"]
    if text_embedding != "Not Found.":
        embedding = clientOpenAI.embeddings.create(input=text_embedding, model=os.environ['OPEN_AI_EMBEDDING_MODEL']).data[0].embedding
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding_vector",
                    "queryVector": embedding,
                    "limit": 1,
                    "numCandidates": 50,
                    "exact": False
                }
            }
        ]
        results = collection.aggregate(pipeline)
        for result in results:
            return result["nationality"]
    elif nationality != "":
        return nationality
    else:
        return None
    
async def get_visa_type(q, visatype):
    print("Getting visa type...")
    prompter = Prompt().load_model(os.environ['RAG_MODEL_P1'],api_key=os.environ['OPENAI_API_KEY'])
    prompter.clear_source_materials()
    response = prompter.prompt_main(f"What visa type does the user want?", context=q, prompt_name="answer_or_not_found")
    text_embedding = response["llm_response"]
    if text_embedding != "Not Found.":
        return text_embedding
    elif visatype != "":
        return visatype
    else:
        return "tourism visit"

async def get_key_context(topic, travel_country, passport, visatype, library):
    prompter = Prompt().load_model(os.environ['RAG_MODEL_P1'],api_key=os.environ['OPENAI_API_KEY'])
    prompter.clear_source_materials()
    results = Query(library).query(topic)
    source = prompter.add_source_query_results(results)
    if not source:
        return None
    topic_context = f"I am travelling to {travel_country}. I have a {passport} passport. I want {visatype} visa information regarding {topic}."
    return prompter.prompt_with_source(topic_context, prompt_name="default_with_context")[0]["llm_response"]

@app.route("/queryvisainfo")
async def query_visa_info():
    print("Incoming visa information query...")
    q = request.args.get('q')
    if not q:
        return "Please provide a query - q.", 400
    userid = request.args.get('userid')
    if not userid:
        return "Please provide a user id - userid.", 400
    collection = database["User"]
    query_filter = {'user_id': userid}
    user = collection.find_one(query_filter)
    if not user:
        return "User does not exist.", 422
    travelcountry = user["travel_country"]
    nationality = user["nationality"]
    visatype = user["visa_type"]
    ## Check where user wants to travel to
    ## Check for user's passport
    ## Check for visa type
    task1 = asyncio.create_task(get_travel_country(q, travelcountry))
    task2 = asyncio.create_task(get_client_passport(q, nationality))
    task3 = asyncio.create_task(get_visa_type(q, visatype))
    result_travel_country = await task1
    passport = await task2
    visa_type = await task3
    if not result_travel_country:
        return "Which country do you need a visa for?"
    travel_country = result_travel_country["country"]
    travel_country_url = result_travel_country["url"]
    
    if not passport:
        return "What passport do you have?"

    ## Update User Details
    collection = database["User"]
    query_filter = {"user_id" : userid}
    update_operation = { "$set" : 
        { 
            "nationality" : passport,
            "travel_country": travel_country,
            "visa_type": visa_type
        }
    }
    collection.update_one(query_filter, update_operation)

    ## Identify key topics for context
    print("Identifying key topics from client query...")  
    key_topics = ["how and where to apply", "holding a " + passport + "passport","visa exemptions","the cost","how long it is valid for",q,"important information"]

    ## Build context from key topics
    tasks = set()

    print("Building context...")
    try:
        ## Get country library source files
        print("Getting travel destination data...")
        travel_country_library = travel_country.replace(" ","_")
        library_name = f"VisaInfo_{travel_country_library}_Library"
        library = Library().load_library(library_name)
    except:
        return f"Unfortunately, I do not yet have information for visa's in {travel_country}."
    
    for i, topic in enumerate(key_topics):
        task = asyncio.create_task(get_key_context(topic, travel_country, passport, visatype, library))
        tasks.add(task)

    context = []
    for i, task in enumerate(tasks):
        addContext = await task
        if addContext:
            context.append(addContext)

    context.append(f"For more information please refer to the {travel_country} visa website at {travel_country_url}.")

    ## Get Client response
    print("Generating response...")  
    prompter = Prompt().load_model(os.environ['RAG_MODEL_P0'],api_key=os.environ['OPENAI_API_KEY'], max_output=500)
    prompter.clear_source_materials()
    final_context = f"You are a travel agent responding to a client on WhatsApp."
    final_context = final_context + f" The client has a {passport} passport."
    final_context = final_context + f" Number the steps for how to apply for a {visa_type} visa to travel to {travel_country}, keep it under 5 steps."
    final_context = final_context + f" Add additional information relating to the client's question '{q}'."
    final_context = final_context + f" Try provide a link for the client's next step, otherwise you must include the visa website url {travel_country_url}."
    output = prompter.prompt_main(final_context, context="\n".join(context), prompt_name="default_with_context")
    return output["llm_response"]

if __name__ == "__main__":
    print("Starting Visa Info Flask server...")
    app.run(host=os.environ["VISAINFO_AI_API_ENDPOINT"], port=os.environ["VISAINFO_AI_API_PORT"])
    