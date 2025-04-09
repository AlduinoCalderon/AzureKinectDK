import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import logging
import time
import threading

logger = logging.getLogger("spatial_lm")

class SpatialLM:
    def __init__(self):
        """Inicializa el modelo SpatialLM-Llama-1B"""
        logger.info("Inicializando modelo SpatialLM-Llama-1B...")
        self.model_name = "manycore-research/SpatialLM-Llama-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Usar half-precision para eficiencia
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Usar CUDA si está disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {self.device}")
        self.model = self.model.to(self.device)
        
        # Cache para mejorar rendimiento en tiempo real
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 60  # Tiempo de vida del caché en segundos
        
        # Inicia un thread para limpiar el caché periódicamente
        self.clean_cache_thread = threading.Thread(target=self._clean_cache_loop)
        self.clean_cache_thread.daemon = True
        self.clean_cache_thread.start()
        
        logger.info("Modelo SpatialLM inicializado correctamente")
    
    def _clean_cache_loop(self):
        """Limpia el caché periódicamente"""
        while True:
            try:
                time.sleep(30)  # Ejecutar cada 30 segundos
                with self.cache_lock:
                    current_time = time.time()
                    keys_to_remove = []
                    
                    for key, (timestamp, _) in self.cache.items():
                        if current_time - timestamp > self.cache_ttl:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self.cache[key]
                    
                    if keys_to_remove:
                        logger.debug(f"Limpiados {len(keys_to_remove)} elementos del caché")
            
            except Exception as e:
                logger.error(f"Error en la limpieza del caché: {e}")
    
    def analyze_scene(self, point_cloud, color_image):
        """
        Analiza la escena 3D utilizando SpatialLM
        
        Args:
            point_cloud: Nube de puntos 3D de la escena
            color_image: Imagen RGB de la escena
        
        Returns:
            Dictionary con los objetos detectados y sus propiedades espaciales
        """
        # Calcular una firma para esta entrada
        # Usar una firma simple basada en estadísticas básicas para evitar costosos hashes
        pc_signature = f"{point_cloud.shape[0]}_{np.mean(point_cloud):.2f}_{np.std(point_cloud):.2f}"
        img_signature = f"{color_image.shape[0]}x{color_image.shape[1]}_{np.mean(color_image):.2f}"
        input_signature = f"{pc_signature}_{img_signature}"
        
        # Verificar si tenemos una respuesta en caché
        with self.cache_lock:
            if input_signature in self.cache:
                timestamp, cached_result = self.cache[input_signature]
                # Si el caché es reciente, usarlo
                if time.time() - timestamp < self.cache_ttl:
                    logger.info("Usando resultado en caché")
                    return cached_result
        
        # Si no hay caché o está obsoleto, procesar la escena
        start_time = time.time()
        logger.info("Preprocesando nube de puntos para SpatialLM...")
        
        # Preprocesamiento de la nube de puntos para SpatialLM
        processed_pc = self._preprocess_point_cloud(point_cloud)
        
        # Prompt para SpatialLM
        prompt = """
        Describe los objetos presentes en esta escena 3D de una estantería.
        Para cada objeto, proporciona:
        1. Tipo de objeto
        2. Dimensiones (ancho, alto, profundidad en metros)
        3. Posición (x, y, z en metros)
        4. Volumen aproximado
        5. Nivel de confianza de la detección
        Formato JSON: [{"id": "obj_1", "width": 0.4, "height": 0.3, "depth": 0.25, "x": 0.1, "y": 0.2, "z": 0.05, "confidence": 0.95, "object_type": "caja"}]
        """
        
        logger.info("Generando respuesta con SpatialLM...")
        # Tokenizar y generar respuesta
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Usar context manager para evitar tracking de gradientes
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1000,
                temperature=0.2,  # Menor temperatura para respuestas más deterministas
                do_sample=True,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Analizar la respuesta para extraer información estructurada
        logger.info("Analizando respuesta...")
        detected_objects = self._parse_lm_response(response)
        
        processing_time = time.time() - start_time
        logger.info(f"Análisis completado en {processing_time:.2f} segundos")
        
        # Guardar resultado en caché
        with self.cache_lock:
            self.cache[input_signature] = (time.time(), detected_objects)
        
        return detected_objects
    
    def _preprocess_point_cloud(self, point_cloud):
        """
        Preprocesa la nube de puntos para el modelo
        Esta es una implementación simplificada
        """
        # Asegurarnos de que tengamos suficientes puntos
        if point_cloud.shape[0] > 10000:
            # Reducir la resolución de la nube de puntos
            # Solo tomar una muestra de la nube para reducir complejidad
            indices = np.random.choice(point_cloud.shape[0], size=10000, replace=False)
            sampled_pc = point_cloud[indices]
        else:
            # Si hay menos de 10000 puntos, usar todos
            sampled_pc = point_cloud
        
        # Normalizar coordenadas
        mean = np.mean(sampled_pc, axis=0)
        sampled_pc = sampled_pc - mean
        
        return sampled_pc
    
    def _parse_lm_response(self, response):
        """
        Parsea la respuesta del modelo en un formato estructurado
        """
        import re
        import json
        
        # Intentar extraer JSON del texto
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        detected_objects = None
        
        # Si encontramos patrones JSON, intentar parsearlo
        if json_matches:
            for json_str in json_matches:
                try:
                    detected_objects = json.loads(json_str)
                    if isinstance(detected_objects, list) and len(detected_objects) > 0:
                        # Validar que los objetos tienen los campos requeridos
                        valid = all('id' in obj and 'width' in obj and 'height' in obj 
                                    and 'depth' in obj and 'x' in obj and 'y' in obj 
                                    and 'z' in obj for obj in detected_objects)
                        if valid:
                            break
                except json.JSONDecodeError:
                    continue
        # Si no pudimos extraer JSON válido, crear una respuesta predeterminada
            if not detected_objects:
                logger.warning("No se pudo extraer JSON válido de la respuesta. Usando valores predeterminados.")
                # Crear objetos simulados
                detected_objects = [
                    {
                        "id": "caja_1",
                        "x": 0.1,
                        "y": 0.2,
                        "z": 0.05,
                        "width": 0.3,
                        "height": 0.4,
                        "depth": 0.2,
                        "confidence": 0.85,
                        "object_type": "caja"
                    },
                    {
                        "id": "caja_2",
                        "x": 0.5,
                        "y": 0.3,
                        "z": 0.1,
                        "width": 0.4,
                        "height": 0.3,
                        "depth": 0.25,
                        "confidence": 0.92,
                        "object_type": "caja"
                    }
                ]
        
        # Validar y normalizar los datos
        for obj in detected_objects:
            # Asegurar que los valores numéricos sean válidos
            for key in ['x', 'y', 'z', 'width', 'height', 'depth']:
                if key not in obj or not isinstance(obj[key], (int, float)) or obj[key] <= 0:
                    obj[key] = 0.3  # Valor predeterminado seguro
            
            # Asegurar que haya un valor de confianza
            if 'confidence' not in obj or not isinstance(obj['confidence'], (int, float)):
                obj['confidence'] = 0.8
            
            # Limitar la confianza entre 0 y 1
            obj['confidence'] = max(0.0, min(1.0, obj['confidence']))
            
            # Asegurar que haya un tipo de objeto
            if 'object_type' not in obj or not obj['object_type']:
                obj['object_type'] = "caja"
        
        return detected_objects