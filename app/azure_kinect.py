import numpy as np
import cv2
import time
import logging
import threading
import pyk4a
from pyk4a import Config, PyK4A

logger = logging.getLogger("kinect_camera")

class KinectCamera:
    def __init__(self):
        """Inicializa la cámara Azure Kinect"""
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
                camera_fps=pyk4a.FPS.FPS_15,  # 15 FPS para un buen equilibrio
            )
        )
        self.k4a.start()
        logger.info("Cámara Azure Kinect inicializada")
        
        # Esperar a que la cámara se caliente
        time.sleep(1.0)
        
        # Buffer para el último frame capturado
        self.last_frame = None
        self.last_point_cloud = None
        self.frame_lock = threading.Lock()
        
        # Crear un hilo para capturar frames continuamente
        self.continuous_capture = False
        self.capture_thread = None
    
    def start_continuous_capture(self):
        """Inicia la captura continua de frames en un hilo separado"""
        if self.continuous_capture:
            logger.warning("La captura continua ya está activa")
            return
        
        self.continuous_capture = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Captura continua iniciada")
    
    def stop_continuous_capture(self):
        """Detiene la captura continua de frames"""
        if not self.continuous_capture:
            logger.warning("La captura continua no está activa")
            return
        
        self.continuous_capture = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        logger.info("Captura continua detenida")
    
    def _capture_loop(self):
        """Bucle para capturar frames continuamente"""
        while self.continuous_capture:
            try:
                capture = self.k4a.get_capture()
                if capture.color is not None and capture.depth is not None:
                    # Convertir de BGRA a RGB
                    color_image = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB)
                    depth_image = capture.transformed_depth
                    
                    # Obtener nube de puntos
                    points3d = self.k4a.transformation.depth_image_to_point_cloud(
                        capture.depth,
                        pyk4a.CalibrationType.DEPTH
                    )
                    
                    # Actualizar el buffer de frame
                    with self.frame_lock:
                        self.last_frame = {
                            "color_image": color_image,
                            "depth_image": depth_image,
                            "timestamp": time.time()
                        }
                        self.last_point_cloud = points3d
                
                # Pequeña pausa para no saturar el CPU
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error en captura continua: {e}")
                time.sleep(0.1)  # Esperar un poco antes de reintentar
    
    def capture_frame(self):
        """
        Captura un frame de la cámara Kinect y devuelve imágenes RGB y de profundidad
        
        Si la captura continua está activa, devuelve el último frame
        Si no, captura un nuevo frame
        """
        if self.continuous_capture and self.last_frame is not None:
            with self.frame_lock:
                return self.last_frame.copy()
        
        # Si no hay captura continua o no hay frame disponible, capturar uno nuevo
        try:
            capture = self.k4a.get_capture()
            
            if capture.color is not None and capture.depth is not None:
                # Convertir de BGRA a RGB
                color_image = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB)
                depth_image = capture.transformed_depth
                
                return {
                    "color_image": color_image,
                    "depth_image": depth_image,
                    "timestamp": time.time()
                }
            else:
                raise Exception("No se pudo capturar la imagen de la cámara Kinect")
        except Exception as e:
            logger.error(f"Error al capturar frame: {e}")
            raise
    
    def get_point_cloud(self):
        """
        Genera una nube de puntos 3D a partir de los datos de profundidad
        
        Si la captura continua está activa, devuelve la última nube de puntos
        Si no, genera una nueva
        """
        if self.continuous_capture and self.last_point_cloud is not None:
            with self.frame_lock:
                return self.last_point_cloud.copy()
        
        # Si no hay captura continua o no hay nube de puntos disponible, generar una nueva
        try:
            capture = self.k4a.get_capture()
            
            if capture.depth is not None:
                # Convertir el mapa de profundidad a una nube de puntos 3D
                points3d = self.k4a.transformation.depth_image_to_point_cloud(
                    capture.depth,
                    pyk4a.CalibrationType.DEPTH
                )
                
                return points3d
            else:
                raise Exception("No se pudo obtener datos de profundidad")
        except Exception as e:
            logger.error(f"Error al generar nube de puntos: {e}")
            raise
    
    def get_shelf_dimensions(self, depth_image):
        """Estima las dimensiones de la estantería basado en el mapa de profundidad"""
        # Esta es una implementación simplificada y necesitaría adaptarse
        # al entorno específico y la colocación de la cámara
        
        # Detectar bordes para identificar los límites de la estantería
        edges = cv2.Canny(depth_image, 100, 200)
        
        # Encontrar contornos en la imagen de bordes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno más grande (asumiendo que es la estantería)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Estimar la profundidad promedio en el área del contorno
            roi = depth_image[y:y+h, x:x+w]
            valid_depths = roi[roi > 0]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths) / 1000.0  # convertir a metros
            else:
                avg_depth = 0.5  # valor predeterminado
            
            # Convertir píxeles a metros (aproximación)
            # Esto requeriría calibración real basada en la posición de la cámara
            pixel_to_meter = 0.001  # Ejemplo, necesita calibración
            width = w * pixel_to_meter
            height = h * pixel_to_meter
            
            return {
                "width": width,
                "height": height,
                "depth": avg_depth
            }
        else:
            # Valores predeterminados si no se detecta
            return {"width": 1.0, "height": 2.0, "depth": 0.5}
    
    def close(self):
        """Cierra la conexión con la cámara"""
        if self.continuous_capture:
            self.stop_continuous_capture()
        
        self.k4a.stop()
        logger.info("Cámara Azure Kinect cerrada")