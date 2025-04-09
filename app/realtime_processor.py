import threading
import time
import logging
from datetime import datetime

from app.azure_kinect import KinectCamera
from app.spatial_lm import SpatialLM
from app.models import ShelfScanResponse, DetectedObject
from app.mongo_client import save_to_mongo
from app.mqtt_client import publish_to_mqtt
from app.visualization import RealtimeVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("realtime_processor")

class RealtimeProcessor:
    def __init__(self, shelf_id="shelf_A1", interval=5.0, save_mongo=True, publish_mqtt=True, enable_visualization=True):
        """
        Procesador en tiempo real para el escaneo de estanterías
        
        Args:
            shelf_id: ID de la estantería a escanear
            interval: Intervalo en segundos entre escaneos
            save_mongo: Si es True, guarda los resultados en MongoDB
            publish_mqtt: Si es True, publica los resultados por MQTT
            enable_visualization: Si es True, muestra visualización en tiempo real
        """
        self.shelf_id = shelf_id
        self.interval = interval
        self.save_mongo = save_mongo
        self.publish_mqtt = publish_mqtt
        self.enable_visualization = enable_visualization
        
        self.running = False
        self.processing_thread = None
        self.last_result = None
        
        # Inicializar cámara y modelo
        try:
            self.camera = KinectCamera()
            self.model = SpatialLM()
            logger.info("Cámara y modelo inicializados correctamente")
            
            # Iniciar captura continua
            self.camera.start_continuous_capture()
            
            # Inicializar visualizador si está habilitado
            self.visualizer = None
            if self.enable_visualization:
                self.visualizer = RealtimeVisualizer(window_name=f"ColdConnect - {shelf_id}")
                self.visualizer.start()
                logger.info("Visualizador iniciado")
                
        except Exception as e:
            logger.error(f"Error al inicializar cámara o modelo: {e}")
            raise
    
    def start(self):
        """Inicia el procesamiento en tiempo real"""
        if self.running:
            logger.warning("El procesador ya está en ejecución")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info(f"Iniciado procesamiento en tiempo real para estantería {self.shelf_id}")
    
    def stop(self):
        """Detiene el procesamiento en tiempo real"""
        if not self.running:
            logger.warning("El procesador no está en ejecución")
            return
        
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=10.0)
        
        # Detener visualizador si está habilitado
        if self.visualizer:
            self.visualizer.stop()
            logger.info("Visualizador detenido")
        
        # Detener captura continua
        try:
            self.camera.stop_continuous_capture()
            logger.info("Captura continua detenida")
        except Exception as e:
            logger.error(f"Error al detener captura continua: {e}")
        
        # Cerrar la cámara
        try:
            self.camera.close()
            logger.info("Cámara cerrada correctamente")
        except Exception as e:
            logger.error(f"Error al cerrar la cámara: {e}")
        
        logger.info("Procesamiento en tiempo real detenido")
    
    def get_last_result(self):
        """Devuelve el último resultado del procesamiento"""
        return self.last_result
    
    def _processing_loop(self):
        """Bucle principal de procesamiento en tiempo real"""
        while self.running:
            try:
                start_time = time.time()
                
                # Realizar un escaneo
                result = self._perform_scan()
                self.last_result = result
                
                # Guardar y publicar resultados
                if self.save_mongo:
                    save_to_mongo(result.dict())
                    logger.info("Resultados guardados en MongoDB")
                
                if self.publish_mqtt:
                    publish_to_mqtt(result.dict())
                    logger.info("Resultados publicados por MQTT")
                
                # Calcular tiempo de procesamiento y esperar si es necesario
                processing_time = time.time() - start_time
                logger.info(f"Escaneo completado en {processing_time:.2f} segundos")
                
                # Esperar el tiempo restante para mantener el intervalo deseado
                sleep_time = max(0, self.interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error en el bucle de procesamiento: {e}")
                # Esperar un poco antes de reintentar para evitar bucles de error
                time.sleep(1.0)
    
    def _perform_scan(self):
        """Realiza un escaneo individual de la estantería"""
        logger.info("Iniciando escaneo...")
        
        # Capturar datos de la cámara
        frame_data = self.camera.capture_frame()
        color_image = frame_data["color_image"]
        depth_image = frame_data["depth_image"]
        
        # Obtener nube de puntos
        point_cloud = self.camera.get_point_cloud()
        
        # Obtener dimensiones de la estantería
        shelf_dims = self.camera.get_shelf_dimensions(depth_image)
        total_volume = shelf_dims["width"] * shelf_dims["height"] * shelf_dims["depth"]
        
        # Analizar la escena con SpatialLM
        logger.info("Procesando escena con SpatialLM...")
        detected_objects_data = self.model.analyze_scene(point_cloud, color_image)
        
        # Convertir los datos detectados al formato de nuestro modelo
        objects = []
        occupied_volume = 0.0
        
        for obj_data in detected_objects_data:
            obj_volume = obj_data["width"] * obj_data["height"] * obj_data["depth"]
            occupied_volume += obj_volume
            
            obj = DetectedObject(
                id=obj_data["id"],
                x=obj_data["x"],
                y=obj_data["y"],
                z=obj_data["z"],
                width=obj_data["width"],
                height=obj_data["height"],
                depth=obj_data["depth"],
                confidence=obj_data.get("confidence", 0.9),
                object_type=obj_data.get("object_type", "box")
            )
            objects.append(obj)
        
        # Calcular volumen libre
        free_volume = total_volume - occupied_volume
        
        # Crear respuesta
        result = ShelfScanResponse(
            shelf_id=self.shelf_id,
            timestamp=datetime.now(),
            total_volume=round(total_volume, 2),
            occupied_volume=round(occupied_volume, 2),
            free_volume=round(free_volume, 2),
            objects=objects
        )
        
        # Actualizar visualizador si está habilitado
        if self.visualizer:
            # Información de la estantería para el visualizador
            shelf_info = {
                "shelf_id": self.shelf_id,
                "total_volume": result.total_volume,
                "occupied_volume": result.occupied_volume,
                "free_volume": result.free_volume,
                "usage_percentage": result.usage_percentage
            }
            
            # Actualizar datos del visualizador
            self.visualizer.update_data(
                color_image=color_image,
                depth_image=depth_image,
                detected_objects=detected_objects_data,
                shelf_info=shelf_info
            )
        
        return result