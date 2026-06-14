import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PGCONN_STRING = os.getenv("PGCONN_STRING")
PGCONN_MATRIKKEL = os.getenv("PGCONN_MATRIKKEL")


PGCONN_FLOMAKTSOMRADER = os.getenv("PGCONN_FLOMAKTSOMRADER_GIS")  
PGCONN_FLOMSONER       = os.getenv("PGCONN_FLOMSONER_GIS")        
PGCONN_BRANNSTASJONER  = os.getenv("PGCONN_BRANNSTASJONER_GIS")   

PGCONN_KVIKKLEIRE = os.getenv("PGCONN_KVIKKLEIRE_GIS")
PGCONN_SNOSKREDAKTSOMHETSKART = os.getenv("PGCONN_SNOSKREDAKTSOMHETSKART_GIS")
PGCONN_MATRIKKELENEIENDOMSKARTTEIG = os.getenv("PGCONN_MATRIKKELENEIENDOMSKARTTEIG_GIS")





PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED = os.getenv("PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED_GIS")

PGCONN_KOMMUNER = os.getenv("PGCONN_KOMMUNER_GIS")


PGCONN_FYLKER_GIS = os.getenv("PGCONN_FYLKER_GIS")
PGCONN_STATISTISKRUTENETT1KM_GIS = os.getenv("PGCONN_STATISTISKRUTENETT1KM_GIS")
PGCONN_TRAFIKKMENGDE_GIS = os.getenv("PGCONN_TRAFIKKMENGDE_GIS")
PGCONN_STEDSNAVN_GIS = os.getenv("PGCONN_STEDSNAVN_GIS")


PGCONN_AREALRESSURSKART_AR50 = os.getenv("PGCONN_AREALRESSURSKART_AR50")
PGCONN_STORMFLOHAVNIVA_GIS = os.getenv("PGCONN_STORMFLOHAVNIVA_GIS")



AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

ORS_API_KEY = os.getenv("ORS_API_KEY", "")

AZURE_FOUNDRY_KEY = os.getenv("AZURE_FOUNDRY_KEY")
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")





AZURE_MODEL_DEPLOYMENT = {
    # Azure OpenAI
    "gpt-35-turbo":          "gpt-35-turbo",
    "gpt-35-turbo-16k":      "gpt-35-turbo-16k",
    "gpt-35-turbo-instruct": "gpt-35-turbo-instruct",
    "gpt-4":                 "gpt-4",
    "gpt-4o":                "gpt-4o",
    "gpt-4o-mini":           "gpt-4o-mini",
    "gpt-4.1-nano":          "gpt-4.1-nano",
    # Azure Foundry — GPT
    "gpt-4-32k":             "gpt-4-32k",
    "gpt-4.1":               "gpt-4.1",
    "gpt-4.1-mini":          "gpt-4.1-mini",
    "gpt-4.5-preview":       "gpt-4.5-preview",
    "gpt-5-mini":            "gpt-5-mini",
    "o1":                    "o1",
    "o3":                    "o3",
    "o3-mini":               "o3-mini",
    "o3-pro":                "o3-pro",
    "o4-mini":               "o4-mini",
    # Azure Foundry — andre
    "Mistral-Large-3":       "Mistral-Large-3",
    "Mistral-small-2503":    "mistral-small-2503",
    "Llama-4-Maverick":      "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "grok-3-mini":           "grok-3-mini",
    "grok-3":                "grok-3",
    "DeepSeek-V3-0324":      "DeepSeek-V3-0324",
    "DeepSeek-V3.1":           "DeepSeek-V3.1",
    "DeepSeek-V3.2":         "DeepSeek-V3.2",
    "Cohere-command-r":      "Cohere-command-r-08-2024",
    "Phi-4":                 "Phi-4",
}

AZURE_OPENAI_MODEL_IDS = {
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-35-turbo-instruct",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-nano",
}






AZURE_FOUNDRY_MODEL_IDS = set(AZURE_MODEL_DEPLOYMENT.keys()) - AZURE_OPENAI_MODEL_IDS









# ALL_MODELS = list(AZURE_MODEL_DEPLOYMENT.keys())








ALL_MODELS = ["Mistral-Large-3", "gpt-5-mini"]











AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)

EMBEDDINGS_TOP_K = int(os.getenv("EMBEDDINGS_TOP_K", "5"))





EMBEDDINGS_MIN_SCORE = float(os.getenv("EMBEDDINGS_MIN_SCORE", "0.20"))




EMBEDDINGS_TABLE_INDEX_PATH = os.getenv(
    "EMBEDDINGS_TABLE_INDEX_PATH",
    str(Path(__file__).resolve().parent / "backend" / "embeddings" / "table_index.json"),
)



OLLAMA_MODEL_IDS = {
    "functiongemma",
}





#blob storage


AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "llm-gis-evaluation-results")

STORAGE_VERSION = os.getenv("STORAGE_VERSION", "statistical_repeated_runs")


DB_DESCRIPTIONS = {
    "matrikkelenbygning": "Building and cadastral building data, including building identifiers, municipality attributes, status fields, and building representation points.",
    "flomaktsomrader": "Flood hazard and flood susceptibility spatial data, including hazard polygons and related flood-risk layers.",
    "flomsoner": "Flood zone spatial data, including flood hazard polygons and flood zone identifiers.",
    "kvikkleire": "Quick clay hazard data and related landslide-risk spatial layers.",
    "snoskredaktsomhetskart": "Snow avalanche susceptibility data and related hazard zones.",
    "matrikkeleneiendomskartteig": "Property parcel data with parcel identifiers, municipality attributes, parcel polygons, and cadastral parcel metadata.",
    "aktsomhetskart_jord_flomskred": "Soil and flood landslide susceptibility data and related hazard layers.",
    "kommuner": "Municipality boundary and municipality reference data, including official municipality names, municipality numbers, and municipality polygons.",
    "fylker": "County boundary and county reference data, including official county names, county numbers, and county polygons.",
    "statistiskrutenett1km": "Regular 1 km statistical grid for spatial aggregation and grid-based analysis independent of municipality boundaries.",
    "stedsnavn": "Place name data with coordinates, language form, municipality affiliation, and place name type.",

    
    "arealressurskart_ar50": "AR50 land resource data with area types, agriculture classes, and forest productivity information.",





}
