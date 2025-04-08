from app.models import ShelfScanResponse, DetectedObject
import random

def scan_shelf_logic():
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
            depth=depth
        )
        objects.append(obj)

    free_volume = total_volume - occupied_volume

    return ShelfScanResponse(
        shelf_id="shelf_A1",
        total_volume=round(total_volume, 2),
        occupied_volume=round(occupied_volume, 2),
        free_volume=round(free_volume, 2),
        objects=objects
    )
