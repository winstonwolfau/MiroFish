"""
Graph backend entity reader abstraction.

This is Phase 1 of the Graphiti migration and keeps existing Zep behavior by
default while introducing a backend selector.
"""

from typing import Optional, List

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import ZepEntityReader, EntityNode, FilteredEntities

logger = get_logger('mirofish.graph_entity_reader')


class GraphitiEntityReader:
    """
    Placeholder Graphiti entity reader.

    Note: A full Graphiti implementation will be added in subsequent phases.
    """

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        raise NotImplementedError(
            "GRAPH_BACKEND=graphiti is enabled, but Graphiti entity reads "
            "are not implemented yet. Set GRAPH_BACKEND=zep for now."
        )

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        raise NotImplementedError(
            "GRAPH_BACKEND=graphiti is enabled, but Graphiti entity reads "
            "are not implemented yet. Set GRAPH_BACKEND=zep for now."
        )

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        raise NotImplementedError(
            "GRAPH_BACKEND=graphiti is enabled, but Graphiti entity reads "
            "are not implemented yet. Set GRAPH_BACKEND=zep for now."
        )


def get_entity_reader():
    """Return an entity reader for the configured graph backend."""
    backend = (Config.GRAPH_BACKEND or 'zep').strip().lower()

    if backend == 'zep':
        if not Config.ZEP_API_KEY:
            raise ValueError('ZEP_API_KEY is required when GRAPH_BACKEND=zep')
        return ZepEntityReader()

    if backend == 'graphiti':
        logger.info('Using Graphiti backend entity reader')
        return GraphitiEntityReader()

    raise ValueError(f'Unsupported GRAPH_BACKEND: {backend}')
