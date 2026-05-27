from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict, Optional
from dataclasses import dataclass
import logging


@dataclass
class EmbedderConfig:
    """Configuration for embedder initialization."""
    model_name: str
    device: Optional[str] = None
    batch_size: int = 32
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseEmbedder(ABC):
    """Abstract base class for all embedders with automatic LangSmith tracing."""
    
    def __init__(self, config: Union[EmbedderConfig, str]):
        """
        Initialize embedder with config or model name.
        
        Args:
            config: EmbedderConfig object or string model name
        """
        if isinstance(config, str):
            self.config = EmbedderConfig(model_name=config)
        else:
            self.config = config
        
        self.model_name = self.config.model_name
        self.device = self.config.device
        self.batch_size = self.config.batch_size
        
        self._initialize_model()
    
    @abstractmethod 
    def _initialize_model(self):
        pass
        
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_document(self, document: str) -> List[float]:   
        pass


class EmbedderFactory:
    """Factory class for creating embedder instances."""
    
    _embedders: Dict[str, type] = {}
    
    @classmethod
    def register_embedder(cls, name: str, embedder_class: type):
        """Register an embedder class with the factory."""
        cls._embedders[name] = embedder_class
        logging.info(f"Registered embedder: {name}")
    
    @classmethod
    def create_embedder(cls, name: str, config: Union[EmbedderConfig, str, Dict[str, Any]]) -> BaseEmbedder:
        """
        Create an embedder instance.
        
        Args:
            name: Name of the registered embedder
            config: Configuration for the embedder
            
        Returns:
            Embedder instance
        """
        if name not in cls._embedders:
            available = ", ".join(cls._embedders.keys())
            raise ValueError(f"Unknown embedder: {name}. Available: {available}")
        
        embedder_class = cls._embedders[name]
        
        if isinstance(config, dict):
            config = EmbedderConfig(**config)
        
        return embedder_class(config)
    
    @classmethod
    def list_embedders(cls) -> List[str]:
        """List all registered embedders."""
        return list(cls._embedders.keys())
    
    @classmethod
    def get_embedder_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a registered embedder."""
        if name not in cls._embedders:
            raise ValueError(f"Unknown embedder: {name}")
        
        embedder_class = cls._embedders[name]
        return {
            "name": name,
            "class": embedder_class.__name__,
            "module": embedder_class.__module__,
        }

