import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def save_to_mongo(result: dict):
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)
    db = client["coldconnect"]  # Usamos la base de datos "coldconnect"
    collection = db["shelf_scans"]  # Almacenamos en la colección "shelf_scans"
    collection.insert_one(result)
