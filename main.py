from fastapi import FastAPI, Query
from app.shelf_scanner import scan_shelf_logic
from app.models import ShelfScanResponse
from app.mongo_client import save_to_mongo
from app.mqtt_client import publish_to_mqtt

app = FastAPI(title="ColdConnect Shelf Scanner", 
              description="API para escaneo de estanterías con Azure Kinect y SpatialLM")

@app.post("/scan-shelf", response_model=ShelfScanResponse)
def scan_shelf(
    shelf_id: str = Query("shelf_A1", description="ID de la estantería a escanear"),
    use_simulation: bool = Query(False, description="Usar datos simulados en lugar de la cámara real"),
    save_mongo: bool = Query(True, description="Guardar resultados en MongoDB"),
    publish_mqtt: bool = Query(False, description="Publicar resultados vía MQTT")
):
    """
    Escanea una estantería y devuelve información sobre su ocupación y objetos detectados.
    Puede opcionalmente guardar los resultados en MongoDB y/o publicarlos por MQTT.
    """
    result = scan_shelf_logic(shelf_id=shelf_id, use_simulation=use_simulation)
    
    # Convertir a diccionario
    result_dict = result.dict()
    
    # Guardar en MongoDB si se solicita
    if save_mongo:
        save_to_mongo(result_dict)
    
    # Publicar vía MQTT si se solicita
    if publish_mqtt:
        publish_to_mqtt(result_dict)
    
    return result

@app.get("/health")
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return {"status": "ok"}