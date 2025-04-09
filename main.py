from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from app.shelf_scanner import scan_shelf_logic
from app.models import ShelfScanResponse
from app.mongo_client import save_to_mongo
from app.mqtt_client import publish_to_mqtt
from app.realtime_processor import RealtimeProcessor
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="ColdConnect Shelf Scanner", 
              description="API para escaneo de estanterías con Azure Kinect y SpatialLM en tiempo real")

# Almacenamiento global para el procesador en tiempo real
realtime_processors = {}

@app.post("/scan-shelf", response_model=ShelfScanResponse)
def scan_shelf(
    shelf_id: str = Query("shelf_A1", description="ID de la estantería a escanear"),
    use_simulation: bool = Query(False, description="Usar datos simulados en lugar de la cámara real"),
    save_mongo: bool = Query(True, description="Guardar resultados en MongoDB"),
    publish_mqtt: bool = Query(False, description="Publicar resultados vía MQTT")
):
    """
    Escanea una estantería una sola vez y devuelve información sobre su ocupación y objetos detectados.
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

@app.post("/start-realtime/{shelf_id}")
def start_realtime_processing(
    shelf_id: str,
    interval: float = Query(5.0, description="Intervalo en segundos entre escaneos"),
    save_mongo: bool = Query(True, description="Guardar resultados en MongoDB"),
    publish_mqtt: bool = Query(True, description="Publicar resultados por MQTT"),
    background_tasks: BackgroundTasks = None
):
    """
    Inicia el procesamiento en tiempo real para una estantería específica.
    
    Args:
        shelf_id: ID de la estantería a monitorear
        interval: Intervalo en segundos entre cada escaneo
        save_mongo: Si es True, guarda resultados en MongoDB
        publish_mqtt: Si es True, publica resultados por MQTT
    """
    # Verificar si ya hay un procesador para esta estantería
    if shelf_id in realtime_processors:
        if realtime_processors[shelf_id].running:
            return {"status": "already_running", "message": f"El procesamiento en tiempo real ya está activo para {shelf_id}"}
        else:
            # Si existe pero no está corriendo, eliminar y crear uno nuevo
            del realtime_processors[shelf_id]
    
    try:
        # Crear un nuevo procesador para esta estantería
        processor = RealtimeProcessor(
            shelf_id=shelf_id,
            interval=interval,
            save_mongo=save_mongo,
            publish_mqtt=publish_mqtt
        )
        
        # Iniciar el procesamiento
        processor.start()
        
        # Almacenar el procesador
        realtime_processors[shelf_id] = processor
        
        return {
            "status": "started", 
            "message": f"Procesamiento en tiempo real iniciado para {shelf_id} con intervalo de {interval} segundos",
            "saving_to_mongo": save_mongo,
            "publishing_to_mqtt": publish_mqtt
        }
    
    except Exception as e:
        logger.error(f"Error al iniciar procesamiento en tiempo real: {e}")
        raise HTTPException(status_code=500, detail=f"Error al iniciar procesamiento: {str(e)}")

@app.post("/stop-realtime/{shelf_id}")
def stop_realtime_processing(shelf_id: str):
    """
    Detiene el procesamiento en tiempo real para una estantería específica
    
    Args:
        shelf_id: ID de la estantería a detener
    """
    if shelf_id not in realtime_processors:
        return {"status": "not_found", "message": f"No hay procesamiento activo para {shelf_id}"}
    
    try:
        # Detener el procesamiento
        realtime_processors[shelf_id].stop()
        
        # Opcional: eliminar el procesador después de detenerlo
        # del realtime_processors[shelf_id]
        
        return {"status": "stopped", "message": f"Procesamiento en tiempo real detenido para {shelf_id}"}
    
    except Exception as e:
        logger.error(f"Error al detener procesamiento en tiempo real: {e}")
        raise HTTPException(status_code=500, detail=f"Error al detener procesamiento: {str(e)}")

@app.get("/realtime-status")
def realtime_status():
    """Devuelve el estado de todos los procesadores en tiempo real"""
    status = {}
    for shelf_id, processor in realtime_processors.items():
        status[shelf_id] = {
            "running": processor.running,
            "last_update": processor.last_result.timestamp.isoformat() if processor.last_result else None
        }
    
    return {"processors": status, "count": len(realtime_processors)}

@app.get("/latest-scan/{shelf_id}")
def get_latest_scan(shelf_id: str):
    """
    Devuelve el último escaneo disponible para una estantería específica
    
    Args:
        shelf_id: ID de la estantería
    """
    if shelf_id not in realtime_processors:
        raise HTTPException(status_code=404, detail=f"No hay procesador activo para {shelf_id}")
    
    result = realtime_processors[shelf_id].get_last_result()
    if not result:
        raise HTTPException(status_code=404, detail=f"No hay datos disponibles para {shelf_id}")
    
    return result

@app.get("/health")
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return {
        "status": "ok",
        "active_processors": len(realtime_processors),
        "timestamp": datetime.now().isoformat()
    }

# Para manejo limpio al apagar
@app.on_event("shutdown")
def shutdown_event():
    """Detiene todos los procesadores al apagar la aplicación"""
    for shelf_id, processor in list(realtime_processors.items()):
        try:
            logger.info(f"Deteniendo procesador para {shelf_id}")
            processor.stop()
        except Exception as e:
            logger.error(f"Error al detener procesador {shelf_id}: {e}")