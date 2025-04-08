import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A

class KinectCamera:
    def __init__(self):
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()
        
    def capture_frame(self):
        """Captura un frame de la cámara Kinect y devuelve imágenes RGB y de profundidad"""
        capture = self.k4a.get_capture()
        
        if capture.color is not None and capture.depth is not None:
            # Convertir de BGRA a RGB
            color_image = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB)
            
            # Obtener mapa de profundidad alineado con la imagen de color
            depth_image = capture.transformed_depth
            
            return {
                "color_image": color_image,
                "depth_image": depth_image
            }
        else:
            raise Exception("No se pudo capturar la imagen de la cámara Kinect")
    
    def get_point_cloud(self):
        """Genera una nube de puntos 3D a partir de los datos de profundidad"""
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
            avg_depth = np.mean(roi[roi > 0]) / 1000.0  # convertir a metros
            
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
        self.k4a.stop()
