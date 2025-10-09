from typing import Dict, Optional
from app.data_model.base_model import BaseModel

class ModelRegistry:
    _instance = None
    _models: Dict[str, BaseModel] = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, name: str, model: BaseModel):
        self._models[name] = model
    
    def get(self, name: str) -> Optional[BaseModel]:
        return self._models.get(name)