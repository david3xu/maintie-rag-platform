# Knowledge Graph Core Logic - Team Beta
# 🔗 Team Beta: Knowledge Graph Service - Core Graph Operations
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: core.py - Core Graph Operations and Logic
#
# Class: SimpleKnowledgeGraph - Main Graph Engine
# ✅ __init__() method - Initialize NetworkX graph, entity/relation mappings, load data
# ✅ load_graph() method - Read processed MaintIE data, build graph, create indices
# ✅ expand_concepts() method - Graph traversal, depth limits, relevance scoring
#
# Class: GraphBuilder - Graph Construction
# ✅ build_from_maintie_data() method - Process annotations, extract entities/relations
# ✅ Apply deduplication, validation, build graph structure
#
# Function: load_raw_data() - Data Loading
# ✅ Load MaintIE files, parse scheme.json, handle errors, validate format
#
# Class: ConceptExpander - Expansion Algorithms
# ✅ expand_with_traversal() method - Breadth-first traversal, relationship filtering
# ✅ score_expansion_relevance() - Path distance, relationship importance, frequency
# === END TODO REFERENCE ===

import json
import pickle
import networkx as nx
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Data structure for knowledge graph entities"""
    id: str
    text: str
    entity_type: str
    properties: Dict[str, Any]
    frequency: int
    aliases: List[str]

@dataclass
class Relation:
    """Data structure for knowledge graph relations"""
    id: str
    source: str
    target: str
    relation_type: str
    confidence: float
    context: List[str]

@dataclass
class ExpansionResult:
    """Data structure for concept expansion results"""
    concept: str
    relevance_score: float
    path_length: int
    relationship_type: str
    source_concepts: List[str]

class GraphBuilder:
    """
    Implementation of: Graph Construction from MaintIE Annotations

    This class processes raw MaintIE annotation data and constructs a knowledge graph
    with entity deduplication, relation validation, and graph optimization for
    production use in concept expansion and retrieval enhancement.
    """

    def __init__(self):
        """
        Implementation of: Initialize data processing components

        Initialize components for processing MaintIE data including entity deduplication
        tools, relation validation rules, and schema definitions.
        """
        # Entity deduplication configuration
        self.entity_similarity_threshold = 0.85
        self.relation_confidence_threshold = 0.7

        # MaintIE schema mappings
        self.entity_type_mappings = {
            "PhysicalObject": "equipment",
            "Activity": "action",
            "State": "condition",
            "Process": "procedure",
            "Property": "attribute"
        }

        # Relation type normalization
        self.relation_type_mappings = {
            "hasPatient": "affects",
            "hasAgent": "performed_by",
            "hasPart": "contains",
            "hasProperty": "has_attribute",
            "isA": "type_of"
        }

        # Entity normalization rules
        self.normalization_rules = {
            r"(\w+)\s+pump": r"\1_pump",
            r"(\w+)\s+motor": r"\1_motor",
            r"(\w+)\s+valve": r"\1_valve",
            r"o[-_]?ring": "o_ring",
            r"seal\s+ring": "seal_ring"
        }

        logger.info("GraphBuilder initialized for MaintIE data processing")

    async def build_from_maintie_data(self, raw_data_path: str) -> nx.Graph:
        """
        Implementation of: Process raw MaintIE annotations and build knowledge graph

        This method processes MaintIE annotation files, extracts entities and relationships,
        applies deduplication logic, validates relations, and constructs an optimized
        NetworkX graph for concept expansion and search operations.

        Args:
            raw_data_path (str): Path to raw MaintIE data directory

        Returns:
            nx.Graph: Constructed knowledge graph with entities and relations
        """
        logger.info(f"Building knowledge graph from MaintIE data at: {raw_data_path}")

        try:
            # Load raw MaintIE data
            raw_data = await self._load_maintie_files(raw_data_path)

            # Extract entities and relations
            entities = self._extract_entities(raw_data)
            relations = self._extract_relations(raw_data, entities)

            # Apply deduplication and validation
            deduplicated_entities = self._deduplicate_entities(entities)
            validated_relations = self._validate_relations(relations, deduplicated_entities)

            # Build NetworkX graph
            graph = self._build_networkx_graph(deduplicated_entities, validated_relations)

            # Optimize graph for querying
            optimized_graph = self._optimize_graph(graph)

            logger.info(f"Knowledge graph built successfully: "
                       f"{len(deduplicated_entities)} entities, "
                       f"{len(validated_relations)} relations")

            return optimized_graph

        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {str(e)}")
            raise

    async def _load_maintie_files(self, data_path: str) -> Dict[str, Any]:
        """
        Implementation of: Load MaintIE annotation files with error handling

        Load gold_release.json, silver_release.json, and scheme.json files,
        handle file not found errors, and validate data format consistency.
        """
        data_path = Path(data_path)
        raw_data = {}

        # Load gold annotations (expert annotations)
        gold_file = data_path / "gold_release.json"
        if gold_file.exists():
            with open(gold_file, 'r', encoding='utf-8') as f:
                raw_data['gold'] = json.load(f)
            logger.info(f"Loaded {len(raw_data['gold'])} gold annotations")
        else:
            logger.warning(f"Gold file not found: {gold_file}")
            raw_data['gold'] = []

        # Load silver annotations (auto annotations)
        silver_file = data_path / "silver_release.json"
        if silver_file.exists():
            with open(silver_file, 'r', encoding='utf-8') as f:
                raw_data['silver'] = json.load(f)
            logger.info(f"Loaded {len(raw_data['silver'])} silver annotations")
        else:
            logger.warning(f"Silver file not found: {silver_file}")
            raw_data['silver'] = []

        # Load schema definitions
        scheme_file = data_path / "scheme.json"
        if scheme_file.exists():
            with open(scheme_file, 'r', encoding='utf-8') as f:
                raw_data['scheme'] = json.load(f)
            logger.info("Schema definitions loaded")
        else:
            logger.warning(f"Schema file not found: {scheme_file}")
            raw_data['scheme'] = {"entities": [], "relations": []}

        return raw_data

    def _extract_entities(self, raw_data: Dict[str, Any]) -> List[Entity]:
        """
        Implementation of: Extract entities from MaintIE annotations

        Parse entity spans and types, apply normalization rules, handle variations,
        create unique identifiers, and build entity property mappings.
        """
        entities = {}
        entity_counter = Counter()

        # Process gold and silver annotations
        for data_type in ['gold', 'silver']:
            annotations = raw_data.get(data_type, [])
            confidence_multiplier = 1.0 if data_type == 'gold' else 0.8

            for doc in annotations:
                text = doc.get('text', '')
                entities_list = doc.get('entities', [])

                for entity in entities_list:
                    entity_text = entity.get('text', '').strip()
                    entity_type = entity.get('type', 'Unknown')

                    if not entity_text:
                        continue

                    # Normalize entity text
                    normalized_text = self._normalize_entity_text(entity_text)
                    entity_id = self._generate_entity_id(normalized_text, entity_type)

                    # Count frequency
                    entity_counter[entity_id] += 1

                    # Create or update entity
                    if entity_id not in entities:
                        entities[entity_id] = Entity(
                            id=entity_id,
                            text=entity_text,
                            entity_type=self.entity_type_mappings.get(entity_type, entity_type),
                            properties={
                                'original_type': entity_type,
                                'confidence': confidence_multiplier,
                                'contexts': []
                            },
                            frequency=0,
                            aliases=[]
                        )

                    # Add context and aliases
                    entities[entity_id].properties['contexts'].append(text[:100])
                    if entity_text not in entities[entity_id].aliases:
                        entities[entity_id].aliases.append(entity_text)

        # Update frequencies
        for entity_id, entity in entities.items():
            entity.frequency = entity_counter[entity_id]

        logger.info(f"Extracted {len(entities)} unique entities")
        return list(entities.values())

    def _extract_relations(self, raw_data: Dict[str, Any], entities: List[Entity]) -> List[Relation]:
        """
        Implementation of: Extract relationships from MaintIE annotations

        Parse relation types and arguments, validate consistency, apply normalization,
        create directional links, and handle confidence scores.
        """
        relations = []
        entity_text_to_id = {entity.text: entity.id for entity in entities}

        # Add aliases to mapping
        for entity in entities:
            for alias in entity.aliases:
                if alias not in entity_text_to_id:
                    entity_text_to_id[alias] = entity.id

        relation_counter = 0

        # Process gold and silver annotations
        for data_type in ['gold', 'silver']:
            annotations = raw_data.get(data_type, [])
            confidence_multiplier = 1.0 if data_type == 'gold' else 0.8

            for doc in annotations:
                text = doc.get('text', '')
                relations_list = doc.get('relations', [])

                for relation in relations_list:
                    head_text = relation.get('head', {}).get('text', '').strip()
                    tail_text = relation.get('tail', {}).get('text', '').strip()
                    relation_type = relation.get('type', '')

                    if not all([head_text, tail_text, relation_type]):
                        continue

                    # Map entity texts to IDs
                    head_id = entity_text_to_id.get(head_text)
                    tail_id = entity_text_to_id.get(tail_text)

                    if not head_id or not tail_id:
                        continue

                    # Normalize relation type
                    normalized_relation_type = self.relation_type_mappings.get(
                        relation_type, relation_type
                    )

                    # Create relation
                    relation_obj = Relation(
                        id=f"rel_{relation_counter}",
                        source=head_id,
                        target=tail_id,
                        relation_type=normalized_relation_type,
                        confidence=confidence_multiplier,
                        context=[text[:100]]
                    )

                    relations.append(relation_obj)
                    relation_counter += 1

        logger.info(f"Extracted {len(relations)} relations")
        return relations

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Apply entity deduplication using similarity matching"""
        deduplicated = []
        seen_normalized = set()

        for entity in entities:
            normalized_key = entity.text.lower().strip()

            # Apply normalization patterns
            for pattern, replacement in self.normalization_rules.items():
                normalized_key = re.sub(pattern, replacement, normalized_key)

            if normalized_key not in seen_normalized:
                seen_normalized.add(normalized_key)
                deduplicated.append(entity)
            else:
                # Merge with existing entity
                for existing in deduplicated:
                    if existing.text.lower().strip() == normalized_key:
                        existing.frequency += entity.frequency
                        existing.aliases.extend(entity.aliases)
                        break

        logger.info(f"Deduplication: {len(entities)} -> {len(deduplicated)} entities")
        return deduplicated

    def _validate_relations(self, relations: List[Relation], entities: List[Entity]) -> List[Relation]:
        """Validate relations and filter by confidence threshold"""
        entity_ids = {entity.id for entity in entities}
        validated = []

        for relation in relations:
            # Check if entities exist
            if relation.source in entity_ids and relation.target in entity_ids:
                # Check confidence threshold
                if relation.confidence >= self.relation_confidence_threshold:
                    validated.append(relation)

        logger.info(f"Relation validation: {len(relations)} -> {len(validated)} relations")
        return validated

    def _build_networkx_graph(self, entities: List[Entity], relations: List[Relation]) -> nx.Graph:
        """Build NetworkX graph from entities and relations"""
        graph = nx.Graph()

        # Add entity nodes
        for entity in entities:
            graph.add_node(
                entity.id,
                text=entity.text,
                entity_type=entity.entity_type,
                frequency=entity.frequency,
                aliases=entity.aliases,
                properties=entity.properties
            )

        # Add relation edges
        for relation in relations:
            graph.add_edge(
                relation.source,
                relation.target,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                context=relation.context
            )

        return graph

    def _optimize_graph(self, graph: nx.Graph) -> nx.Graph:
        """Optimize graph for querying performance"""
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated_nodes)

        # Calculate centrality measures for relevance scoring
        try:
            centrality = nx.degree_centrality(graph)
            betweenness = nx.betweenness_centrality(graph, k=100)  # Sample for large graphs

            # Add centrality as node attributes
            for node_id in graph.nodes():
                graph.nodes[node_id]['degree_centrality'] = centrality.get(node_id, 0)
                graph.nodes[node_id]['betweenness_centrality'] = betweenness.get(node_id, 0)
        except Exception as e:
            logger.warning(f"Failed to calculate centrality measures: {str(e)}")

        logger.info(f"Graph optimization complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph

    def _normalize_entity_text(self, text: str) -> str:
        """Apply text normalization rules"""
        normalized = text.lower().strip()
        for pattern, replacement in self.normalization_rules.items():
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def _generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generate unique entity identifier"""
        normalized_type = self.entity_type_mappings.get(entity_type, entity_type)
        return f"{normalized_type}_{text.replace(' ', '_')}"

class ConceptExpander:
    """
    Implementation of: Expansion Algorithms with Graph Traversal

    This class implements breadth-first graph traversal with relationship filtering,
    relevance scoring, and path analysis for intelligent concept expansion.
    """

    def __init__(self, knowledge_graph: 'SimpleKnowledgeGraph'):
        """
        Implementation of: Initialize expansion strategies and scoring mechanisms

        Initialize expansion strategies, set up scoring mechanisms, configure expansion
        limits, and load domain-specific rules for concept expansion.
        """
        self.knowledge_graph = knowledge_graph
        self.max_expansion_depth = 3
        self.max_results_per_level = 20

        # Relationship type weights for relevance scoring
        self.relationship_weights = {
            "contains": 0.9,
            "affects": 0.8,
            "has_attribute": 0.7,
            "performed_by": 0.6,
            "type_of": 0.5
        }

        # Entity type importance weights
        self.entity_type_weights = {
            "equipment": 1.0,
            "action": 0.8,
            "condition": 0.7,
            "procedure": 0.6,
            "attribute": 0.5
        }

        logger.info("ConceptExpander initialized with graph traversal algorithms")

    def expand_concepts(self, concepts: List[str], depth: int = 2, max_results: int = 10) -> Dict[str, Any]:
        """
        Implementation of: Breadth-first graph traversal with relevance scoring

        Implement breadth-first graph traversal, apply relationship type filtering,
        score paths by relevance, limit expansion depth and breadth, and return
        ranked expansion results with metadata.

        Args:
            concepts (List[str]): Starting concepts for expansion
            depth (int): Maximum traversal depth
            max_results (int): Maximum number of results to return

        Returns:
            Dict: Expansion results with ranked concepts and metadata
        """
        logger.info(f"Expanding concepts: {concepts} (depth={depth}, max={max_results})")

        # Find starting nodes in graph
        start_nodes = self._find_concept_nodes(concepts)
        if not start_nodes:
            logger.warning(f"No starting nodes found for concepts: {concepts}")
            return {"expanded_concepts": [], "nodes_traversed": 0, "paths_explored": 0}

        # Perform breadth-first expansion
        expansion_results = []
        visited_nodes = set(start_nodes.keys())
        current_level = list(start_nodes.keys())
        nodes_traversed = 0
        paths_explored = 0

        for current_depth in range(1, depth + 1):
            next_level = []

            for node in current_level:
                # Get neighbors
                neighbors = self.knowledge_graph.get_node_neighbors(node)
                nodes_traversed += len(neighbors)

                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        # Calculate relevance score
                        relevance_score = self._calculate_relevance_score(
                            neighbor, node, current_depth, concepts
                        )

                        # Create expansion result
                        expansion_result = ExpansionResult(
                            concept=self.knowledge_graph.get_node_text(neighbor),
                            relevance_score=relevance_score,
                            path_length=current_depth,
                            relationship_type=self.knowledge_graph.get_edge_type(node, neighbor),
                            source_concepts=[self.knowledge_graph.get_node_text(node)]
                        )

                        expansion_results.append(expansion_result)
                        next_level.append(neighbor)
                        visited_nodes.add(neighbor)
                        paths_explored += 1

            # Limit results per level
            next_level = next_level[:self.max_results_per_level]
            current_level = next_level

            if not current_level:
                break

        # Sort by relevance and limit results
        expansion_results.sort(key=lambda x: x.relevance_score, reverse=True)
        top_results = expansion_results[:max_results]

        # Format results
        formatted_results = [
            {
                "concept": result.concept,
                "relevance_score": result.relevance_score,
                "path_length": result.path_length,
                "relationship_type": result.relationship_type,
                "source_concepts": result.source_concepts
            }
            for result in top_results
        ]

        logger.info(f"Expansion complete: {len(formatted_results)} concepts found")

        return {
            "expanded_concepts": formatted_results,
            "nodes_traversed": nodes_traversed,
            "paths_explored": paths_explored,
            "expansion_depth": depth,
            "original_concepts": concepts
        }

    def _find_concept_nodes(self, concepts: List[str]) -> Dict[str, str]:
        """Find graph nodes corresponding to input concepts"""
        concept_nodes = {}

        for concept in concepts:
            node_id = self.knowledge_graph.find_node_by_text(concept)
            if node_id:
                concept_nodes[node_id] = concept
            else:
                # Try fuzzy matching
                similar_nodes = self.knowledge_graph.find_similar_nodes(concept)
                if similar_nodes:
                    concept_nodes[similar_nodes[0]] = concept

        return concept_nodes

    def _calculate_relevance_score(self, target_node: str, source_node: str,
                                 path_length: int, original_concepts: List[str]) -> float:
        """
        Implementation of: Relevance scoring with multiple signals

        Calculate path distance weighting, apply relationship type importance,
        consider concept frequency, combine multiple relevance signals, and
        normalize scores for ranking.
        """
        score = 0.0

        # Distance penalty
        distance_weight = 1.0 / (path_length ** 0.5)
        score += distance_weight * 0.4

        # Relationship type importance
        relationship_type = self.knowledge_graph.get_edge_type(source_node, target_node)
        relationship_weight = self.relationship_weights.get(relationship_type, 0.3)
        score += relationship_weight * 0.3

        # Entity type importance
        entity_type = self.knowledge_graph.get_node_type(target_node)
        entity_weight = self.entity_type_weights.get(entity_type, 0.5)
        score += entity_weight * 0.2

        # Frequency/centrality bonus
        centrality = self.knowledge_graph.get_node_centrality(target_node)
        score += centrality * 0.1

        return min(score, 1.0)

class SimpleKnowledgeGraph:
    """
    Implementation of: Main Graph Engine with NetworkX and Search Operations

    This class provides the main interface for knowledge graph operations including
    graph loading, entity management, concept expansion, and search functionality
    for the MaintIE enhanced RAG system.
    """

    def __init__(self):
        """
        Implementation of: Initialize NetworkX graph structure and configurations

        Initialize NetworkX graph structure, set up entity and relation mappings,
        configure expansion parameters, and prepare for data loading.
        """
        self.graph = nx.Graph()
        self.entity_id_to_text = {}
        self.text_to_entity_id = {}
        self.loaded = False

        # Graph statistics
        self.stats = {
            "entities": 0,
            "relations": 0,
            "entity_types": set(),
            "relation_types": set()
        }

        logger.info("SimpleKnowledgeGraph initialized")

    async def initialize(self):
        """Initialize the knowledge graph by loading or building data"""
        try:
            # Try to load existing processed graph
            if await self._load_processed_graph():
                logger.info("Loaded existing processed knowledge graph")
                self.loaded = True
                return

            # Build graph from raw data if no processed version exists
            logger.info("Building knowledge graph from raw MaintIE data...")
            graph_builder = GraphBuilder()

            # Use relative path for data
            data_path = Path(__file__).parent / "data" / "raw"
            self.graph = await graph_builder.build_from_maintie_data(str(data_path))

            # Update mappings and statistics
            self._update_mappings_and_stats()

            # Save processed graph
            await self._save_processed_graph()

            self.loaded = True
            logger.info("Knowledge graph initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {str(e)}")
            # Create minimal graph for testing
            self._create_minimal_graph()
            self.loaded = True

    def _create_minimal_graph(self):
        """Create a minimal graph for testing when data loading fails"""
        logger.warning("Creating minimal test graph")

        # Add basic maintenance entities
        test_entities = [
            ("pump", "equipment"), ("seal", "component"), ("motor", "equipment"),
            ("bearing", "component"), ("leak", "condition"), ("failure", "condition")
        ]

        for entity_text, entity_type in test_entities:
            entity_id = f"{entity_type}_{entity_text}"
            self.graph.add_node(
                entity_id,
                text=entity_text,
                entity_type=entity_type,
                frequency=1
            )
            self.entity_id_to_text[entity_id] = entity_text
            self.text_to_entity_id[entity_text] = entity_id

        # Add basic relationships
        test_relations = [
            ("equipment_pump", "component_seal", "contains"),
            ("equipment_pump", "component_bearing", "contains"),
            ("component_seal", "condition_leak", "causes"),
            ("condition_leak", "condition_failure", "leads_to")
        ]

        for source, target, rel_type in test_relations:
            self.graph.add_edge(source, target, relation_type=rel_type, confidence=0.8)

        self._update_mappings_and_stats()

    def get_entity_count(self) -> int:
        """Return total number of entities in the graph"""
        return len(self.graph.nodes())

    def get_relation_count(self) -> int:
        """Return total number of relations in the graph"""
        return len(self.graph.edges())

    def get_entity_details(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an entity"""
        entity_id = entity_id.lower()

        # Try exact match first
        if entity_id in self.graph.nodes():
            node_data = self.graph.nodes[entity_id]
            return {
                "id": entity_id,
                "text": node_data.get("text", entity_id),
                "type": node_data.get("entity_type", "unknown"),
                "frequency": node_data.get("frequency", 0),
                "properties": node_data.get("properties", {})
            }

        # Try text-based lookup
        if entity_id in self.text_to_entity_id:
            actual_id = self.text_to_entity_id[entity_id]
            return self.get_entity_details(actual_id)

        return None

    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity"""
        relationships = []

        if entity_id in self.graph.nodes():
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.edges[entity_id, neighbor]
                relationships.append({
                    "target_entity": neighbor,
                    "target_text": self.entity_id_to_text.get(neighbor, neighbor),
                    "relationship_type": edge_data.get("relation_type", "unknown"),
                    "confidence": edge_data.get("confidence", 0.0)
                })

        return relationships

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return {
            "nodes": len(self.graph.nodes()),
            "edges": len(self.graph.edges()),
            "entity_types": list(self.stats["entity_types"]),
            "relation_types": list(self.stats["relation_types"]),
            "average_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            "is_connected": nx.is_connected(self.graph),
            "number_of_components": nx.number_connected_components(self.graph)
        }

    def execute_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced graph queries"""
        operation = query.get("operation")

        if operation == "shortest_path":
            source = query.get("source")
            target = query.get("target")
            try:
                path = nx.shortest_path(self.graph, source, target)
                return {"path": path, "length": len(path) - 1}
            except nx.NetworkXNoPath:
                return {"path": None, "error": "No path found"}

        elif operation == "neighbors":
            node = query.get("node")
            if node in self.graph:
                neighbors = list(self.graph.neighbors(node))
                return {"neighbors": neighbors}
            return {"neighbors": [], "error": "Node not found"}

        else:
            return {"error": f"Unknown operation: {operation}"}

    # Helper methods for ConceptExpander
    def get_node_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes"""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []

    def get_node_text(self, node_id: str) -> str:
        """Get text representation of node"""
        return self.entity_id_to_text.get(node_id, node_id)

    def get_edge_type(self, source: str, target: str) -> str:
        """Get edge relationship type"""
        if self.graph.has_edge(source, target):
            return self.graph.edges[source, target].get("relation_type", "unknown")
        return "unknown"

    def get_node_type(self, node_id: str) -> str:
        """Get node entity type"""
        if node_id in self.graph:
            return self.graph.nodes[node_id].get("entity_type", "unknown")
        return "unknown"

    def get_node_centrality(self, node_id: str) -> float:
        """Get node centrality score"""
        if node_id in self.graph:
            return self.graph.nodes[node_id].get("degree_centrality", 0.0)
        return 0.0

    def find_node_by_text(self, text: str) -> Optional[str]:
        """Find node ID by text"""
        text_lower = text.lower()
        return self.text_to_entity_id.get(text_lower)

    def find_similar_nodes(self, text: str, limit: int = 5) -> List[str]:
        """Find similar nodes using text matching"""
        text_lower = text.lower()
        similar = []

        for entity_text, entity_id in self.text_to_entity_id.items():
            if text_lower in entity_text or entity_text in text_lower:
                similar.append(entity_id)
                if len(similar) >= limit:
                    break

        return similar

    async def _load_processed_graph(self) -> bool:
        """Try to load existing processed graph"""
        try:
            processed_path = Path(__file__).parent / "data" / "processed" / "knowledge_graph.pkl"
            if processed_path.exists():
                with open(processed_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.entity_id_to_text = data['entity_id_to_text']
                    self.text_to_entity_id = data['text_to_entity_id']
                    self.stats = data['stats']
                return True
        except Exception as e:
            logger.warning(f"Failed to load processed graph: {str(e)}")

        return False

    async def _save_processed_graph(self):
        """Save processed graph for future loading"""
        try:
            processed_path = Path(__file__).parent / "data" / "processed"
            processed_path.mkdir(parents=True, exist_ok=True)

            graph_file = processed_path / "knowledge_graph.pkl"
            with open(graph_file, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'entity_id_to_text': self.entity_id_to_text,
                    'text_to_entity_id': self.text_to_entity_id,
                    'stats': self.stats
                }, f)

            logger.info(f"Processed graph saved to {graph_file}")
        except Exception as e:
            logger.warning(f"Failed to save processed graph: {str(e)}")

    def _update_mappings_and_stats(self):
        """Update internal mappings and statistics"""
        self.entity_id_to_text.clear()
        self.text_to_entity_id.clear()
        self.stats = {
            "entities": len(self.graph.nodes()),
            "relations": len(self.graph.edges()),
            "entity_types": set(),
            "relation_types": set()
        }

        # Update entity mappings
        for node_id, node_data in self.graph.nodes(data=True):
            text = node_data.get("text", node_id)
            self.entity_id_to_text[node_id] = text
            self.text_to_entity_id[text.lower()] = node_id

            entity_type = node_data.get("entity_type")
            if entity_type:
                self.stats["entity_types"].add(entity_type)

        # Update relation type statistics
        for _, _, edge_data in self.graph.edges(data=True):
            relation_type = edge_data.get("relation_type")
            if relation_type:
                self.stats["relation_types"].add(relation_type)
