import argparse
import logging
import time
import signal
import sys
import uvicorn
from app.realtime_processor import RealtimeProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coldconnect.log')
    ]
)
logger = logging.getLogger("runner")

# Variables globales para manejo de señales
processor = None

def signal_handler(sig, frame):
    """Manejador de señales para detener graciosamente"""
    logger.info("Señal de terminación recibida. Deteniendo...")
    if processor:
        processor.stop()
    sys.exit(0)

def run_realtime_standalone(shelf_id, interval, save_mongo, publish_mqtt, visualization):
    """Ejecuta el procesamiento en tiempo real en modo independiente"""
    global processor
    
    logger.info(f"Iniciando procesamiento en tiempo real para estantería {shelf_id}")
    logger.info(f"  - Intervalo: {interval} segundos")
    logger.info(f"  - Guardar en MongoDB: {save_mongo}")
    logger.info(f"  - Publicar por MQTT: {publish_mqtt}")
    logger.info(f"  - Visualización: {visualization}")
    
    # Registrar handler para señales de terminación
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Inicializar y comenzar el procesador
        processor = RealtimeProcessor(
            shelf_id=shelf_id,
            interval=interval,
            save_mongo=save_mongo,
            publish_mqtt=publish_mqtt,
            enable_visualization=visualization
        )
        processor.start()
        
        # Mantener el programa en ejecución
        logger.info("Procesamiento en tiempo real iniciado. Presione Ctrl+C para detener.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario. Deteniendo...")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
    finally:
        if processor:
            processor.stop()
        logger.info("Procesamiento en tiempo real finalizado")

def run_api_server(host, port, reload):
    """Ejecuta el servidor API FastAPI"""
    logger.info(f"Iniciando servidor API en {host}:{port}")
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error al iniciar servidor API: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColdConnect Shelf Scanner")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Subparser para el servidor API
    api_parser = subparsers.add_parser("api", help="Iniciar servidor API")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host para el servidor API")
    api_parser.add_argument("--port", type=int, default=8000, help="Puerto para el servidor API")
    api_parser.add_argument("--reload", action="store_true", help="Habilitar recarga automática")
    
    # Subparser para procesamiento en tiempo real
    rt_parser = subparsers.add_parser("realtime", help="Iniciar procesamiento en tiempo real")
    rt_parser.add_argument("--shelf-id", default="shelf_A1", help="ID de la estantería a escanear")
    rt_parser.add_argument("--interval", type=float, default=5.0, help="Intervalo en segundos entre escaneos")
    rt_parser.add_argument("--no-mongo", action="store_true", help="No guardar en MongoDB")
    rt_parser.add_argument("--mqtt", action="store_true", help="Publicar por MQTT")
    rt_parser.add_argument("--no-visualization", action="store_true", help="Deshabilitar visualización")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api_server(args.host, args.port, args.reload)
    
    elif args.command == "realtime":
        run_realtime_standalone(
            shelf_id=args.shelf_id,
            interval=args.interval,
            save_mongo=not args.no_mongo,
            publish_mqtt=args.mqtt,
            visualization=not args.no_visualization
        )
    
    else:
        parser.print_help()