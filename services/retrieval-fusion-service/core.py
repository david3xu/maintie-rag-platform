# Retrieval Fusion Core Logic - Team Gamma
# 🎯 Team Gamma: Retrieval Fusion Service - Core Multi-Modal Search & Fusion Logic
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: core.py - Multi-Modal Search & Fusion Logic
#
# Class: SimpleFusion - Main Fusion Engine
# ✅ __init__() method - Initialize vector, entity, graph searchers, fusion config
# ✅ search_and_fuse() method - Execute parallel searches, apply adaptive fusion, rank results
#
# Class: VectorSearchEngine - Semantic Search
# ✅ __init__() method - Load embeddings, initialize similarity engine, configure search
# ✅ search() method - Encode query, calculate similarity, return top-K results
# ✅ load_embeddings() method - Load document embeddings, initialize vector index
#
# Class: EntitySearchEngine - Entity Matching
# ✅ search() method - Match entities against indices, calculate overlap scores
# ✅ build_entity_index() method - Process documents, create inverted index
#
# Class: GraphSearchEngine - Knowledge Graph Search
# ✅ search() method - Query graph for related concepts, find concept-containing documents
#
# Function: intelligent_fusion() - Adaptive Fusion Algorithm
# ✅ Multi-modal result combination with query-adaptive weighting
# === END TODO REFERENCE ===

import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import asyncio
import time
from sentence_transformers import SentenceTransformer
import httpx

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data structure for individual search results"""
    document_id: str
    content: str
    score: float
    source: str  # "vector", "entity", "graph"
    metadata: Dict[str, Any]

@dataclass
class FusionResult:
    """Data structure for fused search results"""
    document_id: str
    content: str
    fusion_score: float
    component_scores: Dict[str, float]
    rank: int
    confidence: float

class VectorSearchEngine:
    """
    Implementation of: Semantic Search with Vector Embeddings

    This class implements semantic vector search using pre-computed document embeddings
    and sentence transformers for query encoding, with similarity calculation and
    top-K retrieval for maintenance domain documents.
    """

    def __init__(self):
        """
        Implementation of: Load embeddings, initialize similarity engine

        Load pre-computed document embeddings, initialize sentence transformer model,
        set up similarity calculation engine, and configure search result limits.
        """
        self.model = None
        self.document_embeddings = None
        self.document_index = {}
        self.embedding_dimension = 384
        self.similarity_threshold = 0.3
        self.ready = False

        # Document store for test implementation
        self.documents = {
            "doc_1": "Hydraulic pump seal replacement procedure requires proper tools and safety equipment",
            "doc_2": "Motor bearing maintenance involves regular lubrication and vibration monitoring",
            "doc_3": "Pump failure diagnosis starts with checking inlet pressure and flow rate",
            "doc_4": "Seal leak detection using visual inspection and pressure testing methods",
            "doc_5": "Engine cooling system maintenance includes radiator cleaning and coolant replacement",
            "doc_6": "Bearing wear patterns indicate misalignment or inadequate lubrication issues",
            "doc_7": "Valve actuator troubleshooting procedures for pneumatic and hydraulic systems",
            "doc_8": "Preventive maintenance schedule for industrial pumps and motor assemblies"
        }

        logger.info("VectorSearchEngine initialized")

    async def initialize(self):
        """
        Implementation of: Load embeddings and initialize vector index

        Load pre-computed document embeddings from storage, initialize vector index
        for fast search, validate embedding dimensions, and set up similarity engine.
        """
        try:
            logger.info("Initializing vector search engine...")

            # Load sentence transformer model
            model_name = "all-MiniLM-L6-v2"  # Small, fast model for development
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")

            # Try to load pre-computed embeddings
            embeddings_loaded = await self._load_precomputed_embeddings()

            if not embeddings_loaded:
                # Generate embeddings for test documents
                await self._generate_embeddings()

            # Build document index
            self._build_document_index()

            self.ready = True
            logger.info(f"Vector search engine ready with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize vector search engine: {str(e)}")
            # Create minimal functionality for testing
            await self._create_minimal_setup()

    async def _load_precomputed_embeddings(self) -> bool:
        """Try to load existing pre-computed embeddings"""
        try:
            embeddings_path = Path(__file__).parent / "data" / "embeddings" / "document_embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    embedding_data = pickle.load(f)
                    self.document_embeddings = embedding_data['embeddings']
                    self.documents = embedding_data['documents']
                logger.info("Loaded pre-computed embeddings")
                return True
        except Exception as e:
            logger.warning(f"Failed to load pre-computed embeddings: {str(e)}")
        return False

    async def _generate_embeddings(self):
        """Generate embeddings for documents"""
        logger.info("Generating document embeddings...")

        documents_list = list(self.documents.values())
        self.document_embeddings = self.model.encode(
            documents_list,
            convert_to_tensor=False,
            show_progress_bar=True
        )

        # Save embeddings for future use
        await self._save_embeddings()

    async def _save_embeddings(self):
        """Save generated embeddings"""
        try:
            embeddings_dir = Path(__file__).parent / "data" / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            embeddings_path = embeddings_dir / "document_embeddings.pkl"
            with open(embeddings_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.document_embeddings,
                    'documents': self.documents
                }, f)
            logger.info(f"Embeddings saved to {embeddings_path}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {str(e)}")

    def _build_document_index(self):
        """Build document ID to index mapping"""
        self.document_index = {doc_id: idx for idx, doc_id in enumerate(self.documents.keys())}

    async def _create_minimal_setup(self):
        """Create minimal setup for testing when full initialization fails"""
        logger.warning("Creating minimal vector search setup")
        self.ready = False
        # Set flag to indicate degraded functionality

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Implementation of: Vector similarity search with query encoding

        Encode query into vector representation, calculate cosine similarity with
        documents, return top-K most similar documents with similarity scores
        and metadata, handle query encoding failures gracefully.

        Args:
            query (str): Text query for semantic search
            max_results (int): Maximum number of results to return

        Returns:
            Dict: Search results with similarity scores and metadata
        """
        try:
            if not self.ready or self.model is None:
                logger.warning("Vector search engine not ready, returning empty results")
                return {
                    "results": [],
                    "similarity_threshold": self.similarity_threshold,
                    "embedding_model": "unavailable"
                }

            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=False)[0]

            # Calculate similarities
            similarities = []
            for doc_id, doc_content in self.documents.items():
                doc_idx = self.document_index[doc_id]
                doc_embedding = self.document_embeddings[doc_idx]

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )

                if similarity >= self.similarity_threshold:
                    similarities.append({
                        "document_id": doc_id,
                        "content": doc_content,
                        "similarity_score": float(similarity),
                        "source": "vector",
                        "metadata": {
                            "embedding_model": "all-MiniLM-L6-v2",
                            "similarity_threshold": self.similarity_threshold
                        }
                    })

            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = similarities[:max_results]

            logger.info(f"Vector search found {len(top_results)} results for query: '{query[:50]}...'")

            return {
                "results": top_results,
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": "all-MiniLM-L6-v2",
                "total_documents": len(self.documents)
            }

        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return {
                "results": [],
                "error": str(e),
                "similarity_threshold": self.similarity_threshold
            }

    def is_ready(self) -> bool:
        """Check if vector search engine is ready"""
        return self.ready and self.model is not None

class EntitySearchEngine:
    """
    Implementation of: Entity-Based Retrieval with Inverted Index

    This class implements entity-based document retrieval using inverted indices,
    entity overlap scoring, and support for entity variations and synonyms.
    """

    def __init__(self):
        """
        Implementation of: Load document entity indices and matching algorithms

        Load document entity indices, initialize entity matching algorithms,
        set up entity normalization rules, and configure match scoring parameters.
        """
        self.entity_index = defaultdict(set)  # entity -> set of document_ids
        self.document_entities = defaultdict(set)  # document_id -> set of entities
        self.entity_aliases = {}
        self.ready = False

        # Test documents with known entities
        self.documents = {
            "doc_1": "Hydraulic pump seal replacement procedure requires proper tools and safety equipment",
            "doc_2": "Motor bearing maintenance involves regular lubrication and vibration monitoring",
            "doc_3": "Pump failure diagnosis starts with checking inlet pressure and flow rate",
            "doc_4": "Seal leak detection using visual inspection and pressure testing methods",
            "doc_5": "Engine cooling system maintenance includes radiator cleaning and coolant replacement",
            "doc_6": "Bearing wear patterns indicate misalignment or inadequate lubrication issues",
            "doc_7": "Valve actuator troubleshooting procedures for pneumatic and hydraulic systems",
            "doc_8": "Preventive maintenance schedule for industrial pumps and motor assemblies"
        }

        # Entity extraction patterns
        self.entity_patterns = {
            "equipment": ["pump", "motor", "engine", "valve", "actuator", "radiator"],
            "components": ["seal", "bearing", "impeller", "rotor", "gasket"],
            "conditions": ["failure", "leak", "wear", "maintenance", "lubrication"],
            "procedures": ["replacement", "diagnosis", "inspection", "troubleshooting", "cleaning"]
        }

        logger.info("EntitySearchEngine initialized")

    async def initialize(self):
        """Initialize entity search engine and build indices"""
        try:
            logger.info("Initializing entity search engine...")

            # Build entity index from documents
            await self._build_entity_index()

            # Load entity aliases
            self._load_entity_aliases()

            self.ready = True
            logger.info(f"Entity search engine ready with {len(self.entity_index)} unique entities")

        except Exception as e:
            logger.error(f"Failed to initialize entity search engine: {str(e)}")
            self.ready = False

    async def _build_entity_index(self):
        """
        Implementation of: Process documents for entity extraction and create inverted index

        Process documents for entity extraction, create inverted index of entities to documents,
        apply entity normalization and deduplication, and save index for fast lookup.
        """
        logger.info("Building entity index from documents...")

        for doc_id, content in self.documents.items():
            content_lower = content.lower()
            doc_entities = set()

            # Extract entities using patterns
            for entity_type, entities in self.entity_patterns.items():
                for entity in entities:
                    if entity in content_lower:
                        # Normalize entity
                        normalized_entity = self._normalize_entity(entity)
                        doc_entities.add(normalized_entity)

                        # Add to inverted index
                        self.entity_index[normalized_entity].add(doc_id)

            # Store document entities
            self.document_entities[doc_id] = doc_entities

        logger.info(f"Entity index built: {len(self.entity_index)} entities across {len(self.documents)} documents")

    def _load_entity_aliases(self):
        """Load entity aliases and synonyms"""
        self.entity_aliases = {
            "pump": ["pumps", "pumping"],
            "motor": ["motors", "engine"],
            "bearing": ["bearings"],
            "seal": ["seals", "sealing", "gasket"],
            "failure": ["fault", "malfunction", "breakdown"],
            "maintenance": ["service", "servicing", "repair"]
        }

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity text"""
        return entity.lower().strip()

    def search(self, entities: List[str], max_results: int = 10) -> Dict[str, Any]:
        """
        Implementation of: Entity-based retrieval with overlap scoring

        Match entities against document indices, calculate entity overlap scores,
        return documents with highest entity matches, handle variations and synonyms,
        and support partial entity matching with confidence scoring.

        Args:
            entities (List[str]): List of entities to search for
            max_results (int): Maximum number of results to return

        Returns:
            Dict: Entity-matched documents with match confidence and metadata
        """
        try:
            if not self.ready:
                logger.warning("Entity search engine not ready")
                return {"results": [], "match_strategy": "unavailable"}

            # Normalize and expand entities
            expanded_entities = set()
            for entity in entities:
                normalized = self._normalize_entity(entity)
                expanded_entities.add(normalized)

                # Add aliases
                if normalized in self.entity_aliases:
                    expanded_entities.update(self.entity_aliases[normalized])

            # Find matching documents
            document_scores = defaultdict(float)
            entities_found = set()

            for entity in expanded_entities:
                if entity in self.entity_index:
                    entities_found.add(entity)
                    matching_docs = self.entity_index[entity]

                    # Score documents based on entity match
                    for doc_id in matching_docs:
                        # Calculate match score based on entity overlap
                        doc_entities = self.document_entities[doc_id]
                        overlap_score = len(expanded_entities & doc_entities) / len(expanded_entities)
                        document_scores[doc_id] += overlap_score

            # Build results
            results = []
            for doc_id, score in document_scores.items():
                if doc_id in self.documents:
                    results.append({
                        "document_id": doc_id,
                        "content": self.documents[doc_id],
                        "entity_match_score": score,
                        "source": "entity",
                        "metadata": {
                            "matched_entities": list(entities_found & self.document_entities[doc_id]),
                            "total_entities_in_doc": len(self.document_entities[doc_id]),
                            "match_strategy": "entity_overlap"
                        }
                    })

            # Sort by match score and limit results
            results.sort(key=lambda x: x["entity_match_score"], reverse=True)
            top_results = results[:max_results]

            logger.info(f"Entity search found {len(top_results)} results for entities: {entities}")

            return {
                "results": top_results,
                "match_strategy": "entity_overlap",
                "entities_found": list(entities_found),
                "entities_missing": list(set(self._normalize_entity(e) for e in entities) - entities_found),
                "total_unique_entities": len(self.entity_index)
            }

        except Exception as e:
            logger.error(f"Entity search error: {str(e)}")
            return {
                "results": [],
                "error": str(e),
                "match_strategy": "error"
            }

    def is_ready(self) -> bool:
        """Check if entity search engine is ready"""
        return self.ready

class GraphSearchEngine:
    """
    Implementation of: Knowledge Graph Search with Concept Expansion

    This class implements knowledge graph-based search using concept expansion
    and relationship traversal to find relevant documents through graph reasoning.
    """

    def __init__(self):
        """
        Implementation of: Initialize knowledge graph client and search strategies

        Initialize knowledge graph client, set up graph traversal parameters,
        configure relationship scoring, and load graph search strategies.
        """
        self.kg_service_url = "http://knowledge-graph:8000"  # Service URL
        self.http_client = None
        self.ready = False
        self.expansion_cache = {}

        # Fallback concept expansion for when KG service unavailable
        self.fallback_expansions = {
            "pump": ["impeller", "seal", "bearing", "motor", "hydraulic"],
            "seal": ["gasket", "o-ring", "leak", "replacement"],
            "motor": ["bearing", "rotor", "stator", "vibration"],
            "bearing": ["lubrication", "wear", "noise", "maintenance"],
            "failure": ["fault", "malfunction", "diagnosis", "repair"],
            "maintenance": ["inspection", "service", "preventive", "schedule"]
        }

        logger.info("GraphSearchEngine initialized")

    async def initialize(self):
        """Initialize graph search engine and validate KG service connectivity"""
        try:
            logger.info("Initializing graph search engine...")

            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=10.0)

            # Test knowledge graph service connectivity
            kg_available = await self._test_kg_service()

            if kg_available:
                logger.info("Knowledge graph service connected successfully")
            else:
                logger.warning("Knowledge graph service unavailable, using fallback expansions")

            self.ready = True

        except Exception as e:
            logger.error(f"Failed to initialize graph search engine: {str(e)}")
            self.ready = False

    async def _test_kg_service(self) -> bool:
        """Test knowledge graph service connectivity"""
        try:
            response = await self.http_client.get(f"{self.kg_service_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def search(self, concepts: List[str], depth: int = 2, max_results: int = 10) -> Dict[str, Any]:
        """
        Implementation of: Graph-based retrieval with concept expansion

        Query knowledge graph for related concepts, find documents containing related
        concepts, score documents by concept relevance, handle graph service
        communication, and return graph-informed results.

        Args:
            concepts (List[str]): Concepts to expand via knowledge graph
            depth (int): Graph traversal depth
            max_results (int): Maximum number of results to return

        Returns:
            Dict: Graph-informed search results with concept expansion metadata
        """
        try:
            if not self.ready:
                logger.warning("Graph search engine not ready")
                return {"results": [], "expanded_concepts": []}

            # Get expanded concepts
            expanded_concepts = asyncio.create_task(self._expand_concepts(concepts, depth))
            expanded_concepts = asyncio.run(expanded_concepts) if asyncio.iscoroutine(expanded_concepts) else expanded_concepts

            # Find documents containing expanded concepts
            results = self._find_documents_with_concepts(concepts + expanded_concepts, max_results)

            logger.info(f"Graph search found {len(results)} results for concepts: {concepts}")

            return {
                "results": results,
                "expanded_concepts": expanded_concepts,
                "graph_statistics": {
                    "original_concepts": len(concepts),
                    "expanded_concepts": len(expanded_concepts),
                    "total_concepts_searched": len(concepts) + len(expanded_concepts)
                }
            }

        except Exception as e:
            logger.error(f"Graph search error: {str(e)}")
            return {
                "results": [],
                "error": str(e),
                "expanded_concepts": []
            }

    async def _expand_concepts(self, concepts: List[str], depth: int) -> List[str]:
        """Expand concepts using knowledge graph or fallback"""
        expanded = []

        # Try knowledge graph service first
        if self.http_client:
            try:
                response = await self.http_client.post(
                    f"{self.kg_service_url}/expand",
                    json={"concepts": concepts, "depth": depth, "max_results": 10}
                )
                if response.status_code == 200:
                    data = response.json()
                    expanded_concepts = data.get("expanded_concepts", [])
                    expanded = [concept["concept"] for concept in expanded_concepts if isinstance(concept, dict)]
                    return expanded
            except Exception as e:
                logger.warning(f"Knowledge graph expansion failed: {str(e)}")

        # Fallback to local expansion
        for concept in concepts:
            if concept.lower() in self.fallback_expansions:
                expanded.extend(self.fallback_expansions[concept.lower()][:3])

        return list(set(expanded))

    def _find_documents_with_concepts(self, concepts: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Find documents containing the specified concepts"""
        # Simple document store for testing
        test_documents = {
            "doc_1": "Hydraulic pump seal replacement procedure requires proper tools and safety equipment",
            "doc_2": "Motor bearing maintenance involves regular lubrication and vibration monitoring",
            "doc_3": "Pump failure diagnosis starts with checking inlet pressure and flow rate",
            "doc_4": "Seal leak detection using visual inspection and pressure testing methods",
            "doc_5": "Engine cooling system maintenance includes radiator cleaning and coolant replacement"
        }

        results = []
        concept_set = set(concept.lower() for concept in concepts)

        for doc_id, content in test_documents.items():
            content_lower = content.lower()
            matched_concepts = [concept for concept in concept_set if concept in content_lower]

            if matched_concepts:
                relevance_score = len(matched_concepts) / len(concept_set)
                results.append({
                    "document_id": doc_id,
                    "content": content,
                    "graph_relevance_score": relevance_score,
                    "source": "graph",
                    "metadata": {
                        "matched_concepts": matched_concepts,
                        "total_concepts_searched": len(concept_set),
                        "concept_match_ratio": relevance_score
                    }
                })

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["graph_relevance_score"], reverse=True)
        return results[:max_results]

    def is_ready(self) -> bool:
        """Check if graph search engine is ready"""
        return self.ready

class SimpleFusion:
    """
    Implementation of: Main Fusion Engine with Adaptive Multi-Modal Combination

    This class implements the core fusion algorithm that combines vector, entity, and
    graph search results using adaptive weighting strategies, query-specific optimization,
    and intelligent result ranking for superior retrieval performance.
    """

    def __init__(self, vector_engine: VectorSearchEngine, entity_engine: EntitySearchEngine,
                 graph_engine: GraphSearchEngine):
        """
        Implementation of: Initialize fusion engine with search components

        Initialize vector, entity, and graph searchers, load fusion configuration
        parameters, set up result scoring mechanisms, and configure search timeouts.
        """
        self.vector_engine = vector_engine
        self.entity_engine = entity_engine
        self.graph_engine = graph_engine

        # Fusion configuration
        self.fusion_strategies = {
            "troubleshooting": {"vector": 0.25, "entity": 0.35, "graph": 0.40},
            "procedural": {"vector": 0.30, "entity": 0.25, "graph": 0.45},
            "informational": {"vector": 0.40, "entity": 0.30, "graph": 0.30},
            "default": {"vector": 0.33, "entity": 0.33, "graph": 0.34}
        }

        self.max_results_per_engine = 20
        self.fusion_threshold = 0.1

        logger.info("SimpleFusion engine initialized with adaptive weighting")

    def search_and_fuse(self, enhanced_query: Dict[str, Any], max_results: int = 5) -> Dict[str, Any]:
        """
        Implementation of: Execute parallel multi-modal searches and apply intelligent fusion

        Execute parallel vector, entity, and graph searches, apply adaptive fusion weights
        based on query characteristics, score and rank combined results, remove duplicates
        intelligently, and return top-K fused results with comprehensive metadata.

        Args:
            enhanced_query (Dict): Enhanced query structure with classification and entities
            max_results (int): Maximum number of results to return

        Returns:
            Dict: Fused search results with rankings and metadata
        """
        start_time = time.time()

        try:
            # Extract query components
            original_query = enhanced_query.get("original", "")
            entities = enhanced_query.get("entities", [])
            query_type = enhanced_query.get("classification", {}).get("type", "default")
            expanded_concepts = enhanced_query.get("expanded_concepts", [])

            logger.info(f"Executing fusion search for query type: {query_type}")

            # Execute parallel searches
            search_results = self._execute_parallel_searches(
                original_query, entities, expanded_concepts
            )

            # Determine fusion strategy based on query type
            fusion_weights = self.fusion_strategies.get(query_type, self.fusion_strategies["default"])

            # Apply intelligent fusion
            fused_results = self._apply_intelligent_fusion(
                search_results, fusion_weights, max_results
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(fused_results, search_results)

            # Build comprehensive response
            response = {
                "results": fused_results,
                "total_found": len(fused_results),
                "fusion_strategy": query_type,
                "search_modes_used": list(search_results.keys()),
                "confidence_scores": confidence_scores,
                "search_statistics": {
                    "vector_results": len(search_results.get("vector", [])),
                    "entity_results": len(search_results.get("entity", [])),
                    "graph_results": len(search_results.get("graph", [])),
                    "fusion_processing_time": time.time() - start_time
                }
            }

            logger.info(f"Fusion completed: {len(fused_results)} results in {response['search_statistics']['fusion_processing_time']:.3f}s")

            return response

        except Exception as e:
            logger.error(f"Fusion search error: {str(e)}")
            return {
                "results": [],
                "error": str(e),
                "total_found": 0,
                "fusion_strategy": "error"
            }

    def _execute_parallel_searches(self, query: str, entities: List[str],
                                 concepts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute searches across all engines"""
        search_results = {}

        # Vector search
        try:
            vector_results = self.vector_engine.search(query, max_results=self.max_results_per_engine)
            search_results["vector"] = vector_results.get("results", [])
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}")
            search_results["vector"] = []

        # Entity search
        try:
            if entities:
                entity_list = [e["text"] if isinstance(e, dict) else e for e in entities]
                entity_results = self.entity_engine.search(entity_list, max_results=self.max_results_per_engine)
                search_results["entity"] = entity_results.get("results", [])
            else:
                search_results["entity"] = []
        except Exception as e:
            logger.warning(f"Entity search failed: {str(e)}")
            search_results["entity"] = []

        # Graph search
        try:
            if concepts:
                graph_results = self.graph_engine.search(concepts, max_results=self.max_results_per_engine)
                search_results["graph"] = graph_results.get("results", [])
            else:
                search_results["graph"] = []
        except Exception as e:
            logger.warning(f"Graph search failed: {str(e)}")
            search_results["graph"] = []

        return search_results

    def _apply_intelligent_fusion(self, search_results: Dict[str, List[Dict[str, Any]]],
                                fusion_weights: Dict[str, float], max_results: int) -> List[Dict[str, Any]]:
        """
        Implementation of: Adaptive fusion algorithm with query-specific weighting

        Analyze query characteristics for weighting, apply adaptive fusion weights by
        query type, score combined results using multiple signals, remove duplicates
        with intelligent merging, and rank final results by composite score.
        """
        # Collect all unique documents
        all_documents = {}

        for search_type, results in search_results.items():
            weight = fusion_weights.get(search_type, 0.0)

            for result in results:
                doc_id = result.get("document_id", "")
                if not doc_id:
                    continue

                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        "document_id": doc_id,
                        "content": result.get("content", ""),
                        "scores": {},
                        "sources": [],
                        "metadata": {}
                    }

                # Add weighted score
                score_key = f"{search_type}_score"
                original_score = self._extract_score(result, search_type)
                weighted_score = original_score * weight

                all_documents[doc_id]["scores"][score_key] = weighted_score
                all_documents[doc_id]["sources"].append(search_type)
                all_documents[doc_id]["metadata"][search_type] = result.get("metadata", {})

        # Calculate fusion scores
        fused_results = []
        for doc_id, doc_data in all_documents.items():
            # Calculate composite fusion score
            fusion_score = sum(doc_data["scores"].values())

            # Boost for multiple source agreement
            source_bonus = (len(doc_data["sources"]) - 1) * 0.1
            fusion_score += source_bonus

            # Only include results above threshold
            if fusion_score >= self.fusion_threshold:
                fused_results.append({
                    "document_id": doc_id,
                    "content": doc_data["content"],
                    "fusion_score": fusion_score,
                    "component_scores": doc_data["scores"],
                    "sources": doc_data["sources"],
                    "metadata": doc_data["metadata"]
                })

        # Sort by fusion score and limit results
        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)

        # Add ranking information
        for rank, result in enumerate(fused_results[:max_results], 1):
            result["rank"] = rank

        return fused_results[:max_results]

    def _extract_score(self, result: Dict[str, Any], search_type: str) -> float:
        """Extract normalized score from search result"""
        if search_type == "vector":
            return result.get("similarity_score", 0.0)
        elif search_type == "entity":
            return result.get("entity_match_score", 0.0)
        elif search_type == "graph":
            return result.get("graph_relevance_score", 0.0)
        return 0.0

    def _calculate_confidence_scores(self, fused_results: List[Dict[str, Any]],
                                   search_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate confidence metrics for fusion results"""
        if not fused_results:
            return {"overall_confidence": 0.0}

        # Calculate various confidence metrics
        avg_fusion_score = sum(r["fusion_score"] for r in fused_results) / len(fused_results)

        # Source agreement confidence
        multi_source_count = sum(1 for r in fused_results if len(r["sources"]) > 1)
        source_agreement = multi_source_count / len(fused_results) if fused_results else 0

        # Result distribution confidence
        score_variance = np.var([r["fusion_score"] for r in fused_results]) if len(fused_results) > 1 else 0
        distribution_confidence = 1.0 / (1.0 + score_variance)

        return {
            "overall_confidence": min(avg_fusion_score, 1.0),
            "source_agreement": source_agreement,
            "distribution_confidence": distribution_confidence,
            "result_count_confidence": min(len(fused_results) / 5.0, 1.0)
        }

    def get_available_strategies(self) -> List[str]:
        """Get list of available fusion strategies"""
        return list(self.fusion_strategies.keys())

    def get_current_config(self) -> Dict[str, Any]:
        """Get current fusion configuration"""
        return {
            "strategies": self.fusion_strategies,
            "max_results_per_engine": self.max_results_per_engine,
            "fusion_threshold": self.fusion_threshold,
            "engines_ready": {
                "vector": self.vector_engine.is_ready(),
                "entity": self.entity_engine.is_ready(),
                "graph": self.graph_engine.is_ready()
            }
        }
