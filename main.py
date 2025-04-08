from fastapi import FastAPI
from app.shelf_scanner import scan_shelf_logic
from app.models import ShelfScanResponse
from app.mongo_client import save_to_mongo

app = FastAPI()

@app.post("/scan-shelf", response_model=ShelfScanResponse)
def scan_shelf():
    result = scan_shelf_logic()  # Procesamos el escaneo de la estantería
    save_to_mongo(result.dict())  # Guardamos los resultados en MongoDB
    return result
