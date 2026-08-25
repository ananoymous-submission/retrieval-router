"""
Vector Database Agnostic Framework

This package provides a generalized interface for working with vector databases,
allowing easy switching between different implementations 
"""

from .base_database import (
    BaseVectorDatabase,
    DatabaseConfig,
    Document,
    SearchResult,
    SearchResults,
    VectorDatabaseFactory
)
# Lazy import vector database implementations to avoid heavy dependencies

def _ensure_databases_registered():
    """Register database implementations on first access."""
    if not hasattr(_ensure_databases_registered, '_registered'):
        # Always register qdrant since it's most commonly used
        from .qdrant_manager import QdrantManager
        VectorDatabaseFactory.register_database("qdrant", QdrantManager)
        _ensure_databases_registered._registered = True

# Monkey patch the factory to ensure registration on first use  
_original_create_database = VectorDatabaseFactory.create_database

@classmethod  
def _lazy_create_database(cls, database_type: str, config):
    _ensure_databases_registered()
    return _original_create_database(database_type, config)

VectorDatabaseFactory.create_database = _lazy_create_database


__all__ = [
    'BaseVectorDatabase',
    'DatabaseConfig', 
    'Document',
    'SearchResult',
    'SearchResults',
    'VectorDatabaseFactory'
] 