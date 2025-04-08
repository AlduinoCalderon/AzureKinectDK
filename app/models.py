from pydantic import BaseModel
from typing import List, Optional
import datetime

class DetectedObject(BaseModel):
    id: str
    x: float  # Posición X en metros
    y: float  # Posición Y en metros
    z: float  # Posición Z en metros
    width: float  # Ancho en metros
    height: float  # Alto en metros
    depth: float  # Profundidad en metros
    confidence: Optional[float] = None  # Nivel de confianza de la detección
    object_type: Optional[str] = None  # Tipo de objeto detectado

class ShelfScanResponse(BaseModel):
    shelf_id: str
    timestamp: datetime.datetime = datetime.datetime.now()
    total_volume: float  # Volumen total en metros cúbicos
    occupied_volume: float  # Volumen ocupado en metros cúbicos
    free_volume: float  # Volumen libre en metros cúbicos
    usage_percentage: Optional[float] = None  # Porcentaje de uso
    objects: List[DetectedObject]  # Lista de objetos detectados
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total_volume > 0:
            self.usage_percentage = round((self.occupied_volume / self.total_volume) * 100, 2)
        else:
            self.usage_percentage = 0.0
    
    def dict(self, *args, **kwargs):
        # Convertir datetime a string para MongoDB
        result = super().dict(*args, **kwargs)
        result["timestamp"] = self.timestamp.isoformat()
        return result