import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List

logger = logging.getLogger("visualization")

class RealtimeVisualizer:
    def __init__(self, window_name="ColdConnect Shelf Scanner", width=1280, height=720):
        """
        Inicializa el visualizador en tiempo real
        
        Args:
            window_name: Nombre de la ventana de visualización
            width: Ancho de la ventana de visualización
            height: Alto de la ventana de visualización
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        self.running = False
        self.visualization_thread = None
        
        # Variables para los datos a visualizar
        self.color_image = None
        self.depth_image = None
        self.detected_objects = []
        self.shelf_info = {}
        
        # Bloqueo para sincronizar acceso a los datos
        self.data_lock = threading.Lock()
    
    def start(self):
        """Inicia la visualización en tiempo real"""
        if self.running:
            logger.warning("El visualizador ya está en ejecución")
            return
        
        self.running = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        logger.info("Visualización en tiempo real iniciada")
    
    def stop(self):
        """Detiene la visualización en tiempo real"""
        if not self.running:
            logger.warning("El visualizador no está en ejecución")
            return
        
        self.running = False
        if self.visualization_thread:
            self.visualization_thread.join(timeout=5.0)
        
        # Cerrar la ventana
        cv2.destroyWindow(self.window_name)
        logger.info("Visualización en tiempo real detenida")
    
    def update_data(self, color_image=None, depth_image=None, detected_objects=None, shelf_info=None):
        """
        Actualiza los datos a visualizar
        
        Args:
            color_image: Imagen RGB de la cámara
            depth_image: Imagen de profundidad de la cámara
            detected_objects: Lista de objetos detectados
            shelf_info: Información sobre la estantería
        """
        with self.data_lock:
            if color_image is not None:
                # Redimensionar si es necesario
                if color_image.shape[1] > self.width:
                    scale = self.width / color_image.shape[1]
                    new_height = int(color_image.shape[0] * scale)
                    color_image = cv2.resize(color_image, (self.width, new_height))
                self.color_image = color_image
            
            if depth_image is not None:
                # Normalizar la imagen de profundidad para visualización
                normalized_depth = self._normalize_depth_image(depth_image)
                
                # Redimensionar si es necesario
                if normalized_depth.shape[1] > self.width:
                    scale = self.width / normalized_depth.shape[1]
                    new_height = int(normalized_depth.shape[0] * scale)
                    normalized_depth = cv2.resize(normalized_depth, (self.width, new_height))
                
                self.depth_image = normalized_depth
            
            if detected_objects is not None:
                self.detected_objects = detected_objects
            
            if shelf_info is not None:
                self.shelf_info = shelf_info
    
    def _normalize_depth_image(self, depth_image):
        """Normaliza la imagen de profundidad para visualización"""
        # Crear una máscara para filtrar píxeles sin datos
        mask = depth_image > 0
        
        if np.any(mask):
            # Normalizar solo los píxeles con datos
            normalized = np.zeros_like(depth_image, dtype=np.uint8)
            min_val = np.min(depth_image[mask])
            max_val = np.max(depth_image[mask])
            
            # Evitar división por cero
            if max_val > min_val:
                # Normalizar a rango 0-255
                normalized[mask] = 255 * (depth_image[mask] - min_val) / (max_val - min_val)
            
            # Aplicar un mapa de colores para mejor visualización
            colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            return colormap
        else:
            # Si no hay datos válidos, devolver una imagen negra
            return np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
    
    def _visualization_loop(self):
        """Bucle principal para visualización en tiempo real"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        while self.running:
            try:
                # Crear una imagen combinada para visualización
                with self.data_lock:
                    # Verificar si tenemos imágenes para mostrar
                    if self.color_image is None and self.depth_image is None:
                        # Si no hay imágenes, mostrar una imagen negra con mensaje
                        display_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_image, "Esperando datos...", (50, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        # Si tenemos imágenes, mostrarlas
                        if self.color_image is not None and self.depth_image is not None:
                            # Asegurar que ambas imágenes tienen el mismo tamaño para la visualización
                            if self.color_image.shape[:2] != self.depth_image.shape[:2]:
                                self.depth_image = cv2.resize(self.depth_image, 
                                                            (self.color_image.shape[1], self.color_image.shape[0]))
                            
                            # Combinar imagen de color y profundidad lado a lado
                            display_image = np.hstack((self.color_image, self.depth_image))
                        elif self.color_image is not None:
                            display_image = self.color_image.copy()
                        else:
                            display_image = self.depth_image.copy()
                        
                        # Dibujar información de los objetos detectados
                        if self.color_image is not None:
                            annotated_image = self._draw_object_annotations(self.color_image.copy())
                            
                            # Si tenemos ambas imágenes, reemplazar la parte de color
                            if self.depth_image is not None:
                                display_image = np.hstack((annotated_image, self.depth_image))
                            else:
                                display_image = annotated_image
                        
                        # Añadir información de la estantería
                        display_image = self._add_shelf_info(display_image)
                
                # Mostrar la imagen
                cv2.imshow(self.window_name, display_image)
                key = cv2.waitKey(1) & 0xFF
                
                # Salir si se presiona 'q'
                if key == ord('q'):
                    self.running = False
                    break
                
                # Pequeña pausa para no saturar el CPU
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error en el bucle de visualización: {e}")
                time.sleep(0.1)  # Esperar un poco antes de reintentar
    
    def _draw_object_annotations(self, image):
        """Dibuja anotaciones de los objetos detectados en la imagen"""
        for obj in self.detected_objects:
            try:
                # Calcular posición en la imagen (esto requeriría calibración real)
                # Esta es una simplificación basada en suposiciones
                
                # Convertir coordenadas 3D a coordenadas de imagen 2D
                # Esto es una aproximación y requeriría calibración real
                img_height, img_width = image.shape[:2]
                
                # Estimar el centro del objeto en la imagen
                # Simplificación: mapear rango [0, 1] a [0, img_width] o [0, img_height]
                center_x = int(obj["x"] * img_width)
                center_y = int(obj["y"] * img_height)
                
                # Estimar el tamaño del objeto en la imagen
                # Simplificación: usar el ancho y alto como porcentaje de la imagen
                width_px = int(obj["width"] * img_width / 2)  # Dividir por 2 para escalar
                height_px = int(obj["height"] * img_height / 2)  # Dividir por 2 para escalar
                
                # Calcular las esquinas del rectángulo
                x1 = max(0, center_x - width_px // 2)
                y1 = max(0, center_y - height_px // 2)
                x2 = min(img_width - 1, center_x + width_px // 2)
                y2 = min(img_height - 1, center_y + height_px // 2)
                
                # Color basado en la confianza (verde = alta confianza, rojo = baja confianza)
                confidence = obj.get("confidence", 0.5)
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                
                # Dibujar el rectángulo
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Añadir etiqueta
                label = f"{obj.get('object_type', 'Objeto')} ({confidence:.2f})"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except Exception as e:
                logger.error(f"Error al dibujar objeto: {e}")
        
        return image
    
    def _add_shelf_info(self, image):
        """Añade información de la estantería a la imagen"""
        if not self.shelf_info:
            return image
        
        # Crear una franja en la parte superior para información
        info_height = 80
        info_background = np.zeros((info_height, image.shape[1], 3), dtype=np.uint8)
        
        # Añadir textos con información
        shelf_id = self.shelf_info.get("shelf_id", "N/A")
        total_volume = self.shelf_info.get("total_volume", 0)
        occupied_volume = self.shelf_info.get("occupied_volume", 0)
        free_volume = self.shelf_info.get("free_volume", 0)
        usage_percentage = self.shelf_info.get("usage_percentage", 0)
        
        cv2.putText(info_background, f"Estantería: {shelf_id}", (10, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_background, f"Vol. Total: {total_volume:.2f} m³  |  Ocupado: {occupied_volume:.2f} m³  |  Libre: {free_volume:.2f} m³", 
                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Dibujar barra de progreso para el uso
        bar_width = int(image.shape[1] * 0.6)
        bar_height = 20
        bar_x = 10
        bar_y = 60
        
        # Fondo de la barra
        cv2.rectangle(info_background, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Progreso de la barra
        filled_width = int(bar_width * usage_percentage / 100)
        if usage_percentage > 80:
            bar_color = (0, 0, 255)  # Rojo - casi lleno
        elif usage_percentage > 60:
            bar_color = (0, 165, 255)  # Naranja - medio lleno
        else:
            bar_color = (0, 255, 0)  # Verde - bastante espacio
        
        cv2.rectangle(info_background, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
        
        # Texto de porcentaje
        cv2.putText(info_background, f"{usage_percentage:.1f}%", (bar_x + bar_width + 10, bar_y + 15), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Combinar con la imagen original
        return np.vstack((info_background, image))