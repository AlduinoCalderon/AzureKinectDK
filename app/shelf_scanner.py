from app.models import ShelfScanResponse, DetectedObject
from app.azure_kinect import KinectCamera
from app.spatial_lm import SpatialLM
import numpy as np

def scan_shelf_logic(shelf_id="shelf_A1", use_simulation=False):
    """
    Escanea una estantería usando la cámara Azure Kinect y el modelo SpatialLM
    
    Args:
        shelf_id: ID de la estantería a escanear
        use_simulation: Si es True, usa datos simulados en lugar de la cámara real
    
    Returns:
        ShelfScanResponse con la información del escaneo
    """
    if use_simulation:
        return _simulate_scan(shelf_id)
    
    try:
        # Inicializar cámara y modelo
        camera = KinectCamera()
        model = SpatialLM()
        
        # Capturar datos de la cámara
        frame_data = camera.capture_frame()
        color_image = frame_data["color_image"]
        depth_image = frame_data["depth_image"]
        
        # Obtener nube de puntos
        point_cloud = camera.get_point_cloud()
        
        # Obtener dimensiones de la estantería
        shelf_dims = camera.get_shelf_dimensions(depth_image)
        total_volume = shelf_dims["width"] * shelf_dims["height"] * shelf_dims["depth"]
        
        # Analizar la escena con SpatialLM
        detected_objects_data = model.analyze_scene(point_cloud, color_image)
        
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
        
        # Cerrar la cámara
        camera.close()
        
        # Calcular volumen libre
        free_volume = total_volume - occupied_volume
        
        return ShelfScanResponse(
            shelf_id=shelf_id,
            total_volume=round(total_volume, 2),
            occupied_volume=round(occupied_volume, 2),
            free_volume=round(free_volume, 2),
            objects=objects
        )
    
    except Exception as e:
        # En caso de error, usar simulación
        print(f"Error al escanear estantería: {e}")
        print("Usando datos simulados como respaldo")
        return _simulate_scan(shelf_id)

def _simulate_scan(shelf_id):
    """Genera datos simulados para pruebas"""
    import random
    
    shelf_dims = (2.0, 1.0, 0.5)  # Dimensiones de la estantería (alto, ancho, profundidad)
    total_volume = shelf_dims[0] * shelf_dims[1] * shelf_dims[2]  # Volumen total

    objects = []
    occupied_volume = 0.0

    for i in range(3):  # Simulamos 3 objetos en la estantería
        width = round(random.uniform(0.3, 0.5), 2)
        height = round(random.uniform(0.3, 0.5), 2)
        depth = round(random.uniform(0.2, 0.4), 2)
        volume = width * height * depth
        occupied_volume += volume

        obj = DetectedObject(
            id=f"box_{i+1}",
            x=round(random.uniform(0.0, 1.0), 2),
            y=round(random.uniform(0.0, 2.0), 2),
            z=round(random.uniform(0.0, 0.5), 2),
            width=width,
            height=height,
            depth=depth,
            confidence=round(random.uniform(0.85, 0.99), 2),
            object_type="caja"
        )
        objects.append(obj)

    free_volume = total_volume - occupied_volume

    return ShelfScanResponse(
        shelf_id=shelf_id,
        total_volume=round(total_volume, 2),
        occupied_volume=round(occupied_volume, 2),
        free_volume=round(free_volume, 2),
        objects=objects
    )