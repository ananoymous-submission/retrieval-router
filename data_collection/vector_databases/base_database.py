from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Optional, Literal, Union
from dataclasses import dataclass
import uuid

@dataclass
class DatabaseConfig:
    """Generic configuration for vector database connections."""
    url: str
    api_key: str
    collection_name: str
    vector_size: int
    vector_type: Literal["single", "multi"] = "single"
    distance_metric: str = "cosine"
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

@dataclass 
class Document:
    """Generic document structure for indexing."""
    id: str
    embedding: Union[List[float], List[List[float]]]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Generic search result structure."""
    document_id: str
    score: float
    metadata: Dict[str, Any]


class SearchResults:
    """Container for search results."""
    
    def __init__(self, results: List[SearchResult]):
        self.results = results
    
    def __iter__(self):
        return iter(self.results)
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, index):
        return self.results[index]


class BaseVectorDatabase(ABC):
    """Abstract base class for vector database implementations with automatic tracing."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.collection_name = config.collection_name
        self.vector_type = config.vector_type
        self.vector_size = config.vector_size
    
    @abstractmethod
    async def initialize_collection(self) -> None:
        pass
    
    @abstractmethod
    async def get_indexed_files(self) -> Set[str]:
        pass
    
    @abstractmethod
    async def index_document(self, embedding: Union[List[float], List[List[float]]], metadata: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: Union[List[float], List[List[float]]],
        limit: int = 5,
        filter_conditions: Optional[Any] = None,
        filter_by_filenames: Optional[List[str]] = None
    ) -> SearchResults:
        pass
    
    def generate_document_id(self, metadata: Dict[str, Any]) -> str:
        """Generate a unique ID for a document based on metadata."""
        file_key = metadata.get("filename", "")
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, file_key))

class VectorDatabaseFactory:
    """Factory for creating vector database instances."""
    
    _databases = {}
    
    @classmethod
    def register_database(cls, name: str, database_class):
        """Register a new database implementation."""
        cls._databases[name] = database_class
    
    @classmethod
    def create_database(cls, database_type: str, config: DatabaseConfig) -> BaseVectorDatabase:
        """Create a database instance of the specified type."""
        if database_type not in cls._databases:
            raise ValueError(f"Unknown database type: {database_type}. Available: {list(cls._databases.keys())}")
        
        return cls._databases[database_type](config)
    
    @classmethod
    def list_available_databases(cls) -> List[str]:
        """List all available database types."""
        return list(cls._databases.keys())
