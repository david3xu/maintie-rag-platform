# Query Enhancement Core Logic - Team Alpha
# 🧠 Team Alpha: Query Enhancement Service - Core Business Logic
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: core.py - Core Business Logic Implementation
#
# Class: QueryEnhancer - Main Enhancement Engine
# ✅ __init__() method - Initialize enhancement components, load models, configure parameters
# ✅ enhance() method - Primary enhancement logic coordinating all steps
#
# Function: classify_query() - Query Type Classification
# ✅ Maintenance domain classification logic with confidence scoring
# ✅ Troubleshooting, procedural, informational query identification
#
# Function: extract_entities() - Entity Extraction
# ✅ Simple maintenance entity extraction with keyword-based detection
# ✅ Maintenance-specific patterns, confidence scoring, normalization
#
# Class: EntityExtractor - Entity Processing
# ✅ Load maintenance vocabularies, pattern matching, entity type mappings
#
# Class: QueryClassifier - Classification Logic
# ✅ Load classification rules, keyword patterns, confidence calculation
# === END TODO REFERENCE ===

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enumeration of maintenance query types for consistent classification"""
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURAL = "procedural"
    INFORMATIONAL = "informational"
    PREVENTIVE = "preventive"
    SAFETY = "safety"

@dataclass
class EntityMatch:
    """Data structure for entity extraction results"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    normalized_form: str

@dataclass
class ClassificationResult:
    """Data structure for query classification results"""
    query_type: QueryType
    confidence: float
    secondary_types: List[Tuple[QueryType, float]]
    reasoning: str

@dataclass
class EnhancementResult:
    """Comprehensive enhancement result structure"""
    original_query: str
    classification: ClassificationResult
    entities: List[EntityMatch]
    expanded_concepts: List[str]
    enhanced: bool
    confidence: float
    metadata: Dict[str, Any]

class QueryClassifier:
    """
    Implementation of: Classification Logic with Maintenance Domain Expertise

    This class implements rule-based classification for maintenance queries,
    identifying troubleshooting, procedural, informational, and other query types
    with confidence scoring and reasoning.
    """

    def __init__(self):
        """
        Implementation of: Load classification rules, keyword patterns, weights

        Initialize classification rules, keyword patterns, set up classification weights,
        and configure threshold parameters for accurate maintenance query classification.
        """
        # Troubleshooting indicators
        self.troubleshooting_keywords = {
            "primary": ["failure", "broken", "fault", "malfunction", "problem", "issue",
                       "not working", "stopped", "error", "alarm", "leak", "noise"],
            "secondary": ["why", "cause", "reason", "fix", "repair", "diagnose",
                         "troubleshoot", "wrong", "abnormal"]
        }

        # Procedural query indicators
        self.procedural_keywords = {
            "primary": ["how to", "procedure", "steps", "instructions", "method",
                       "process", "guide", "manual"],
            "secondary": ["install", "replace", "maintain", "calibrate", "adjust",
                         "service", "operate", "start", "stop"]
        }

        # Informational query indicators
        self.informational_keywords = {
            "primary": ["what is", "information", "specification", "detail", "describe",
                       "explain", "definition"],
            "secondary": ["type", "model", "capacity", "rating", "dimension",
                         "material", "component"]
        }

        # Preventive maintenance indicators
        self.preventive_keywords = {
            "primary": ["schedule", "frequency", "interval", "preventive", "routine",
                       "regular", "maintenance"],
            "secondary": ["when", "how often", "periodic", "inspection", "check"]
        }

        # Safety-related indicators
        self.safety_keywords = {
            "primary": ["safety", "hazard", "dangerous", "risk", "lockout", "tagout",
                       "PPE", "protection"],
            "secondary": ["safe", "precaution", "warning", "caution", "emergency"]
        }

        # Classification weights
        self.weights = {
            "primary_keyword": 0.6,
            "secondary_keyword": 0.3,
            "pattern_match": 0.4,
            "context_bonus": 0.2
        }

        # Pattern definitions
        self.patterns = {
            QueryType.TROUBLESHOOTING: [
                r"why\s+(?:is|does|won\'t|can\'t)",
                r"(?:pump|motor|valve|engine)\s+(?:not|won\'t|doesn\'t)",
                r"(?:leak|noise|vibration|overheating)"
            ],
            QueryType.PROCEDURAL: [
                r"how\s+to\s+\w+",
                r"steps?\s+(?:to|for)",
                r"procedure\s+for"
            ],
            QueryType.INFORMATIONAL: [
                r"what\s+(?:is|are)\s+(?:the|a)",
                r"(?:specification|spec)s?\s+(?:of|for)",
                r"tell\s+me\s+about"
            ]
        }

        logger.info("QueryClassifier initialized with maintenance domain rules")

    def classify(self, query: str) -> ClassificationResult:
        """
        Implementation of: Analyze query characteristics and apply classification rules

        This method analyzes query text, applies classification rules, calculates
        confidence scores, and returns classification results with reasoning.

        Args:
            query (str): Input query to classify

        Returns:
            ClassificationResult: Classification with confidence and reasoning
        """
        query_lower = query.lower().strip()

        # Calculate scores for each query type
        type_scores = {}

        for query_type in QueryType:
            score = self._calculate_type_score(query_lower, query_type)
            type_scores[query_type] = score

        # Find primary classification
        primary_type = max(type_scores.items(), key=lambda x: x[1])
        primary_confidence = primary_type[1]

        # Find secondary classifications (above threshold)
        secondary_threshold = 0.3
        secondary_types = [
            (qtype, score) for qtype, score in type_scores.items()
            if score >= secondary_threshold and qtype != primary_type[0]
        ]
        secondary_types.sort(key=lambda x: x[1], reverse=True)

        # Generate reasoning
        reasoning = self._generate_reasoning(query_lower, primary_type[0], primary_confidence)

        return ClassificationResult(
            query_type=primary_type[0],
            confidence=primary_confidence,
            secondary_types=secondary_types,
            reasoning=reasoning
        )

    def _calculate_type_score(self, query: str, query_type: QueryType) -> float:
        """Calculate classification score for a specific query type"""
        score = 0.0

        # Get keywords for this type
        if query_type == QueryType.TROUBLESHOOTING:
            keywords = self.troubleshooting_keywords
        elif query_type == QueryType.PROCEDURAL:
            keywords = self.procedural_keywords
        elif query_type == QueryType.INFORMATIONAL:
            keywords = self.informational_keywords
        elif query_type == QueryType.PREVENTIVE:
            keywords = self.preventive_keywords
        elif query_type == QueryType.SAFETY:
            keywords = self.safety_keywords
        else:
            return 0.0

        # Primary keyword matching
        primary_matches = sum(1 for keyword in keywords["primary"] if keyword in query)
        score += primary_matches * self.weights["primary_keyword"]

        # Secondary keyword matching
        secondary_matches = sum(1 for keyword in keywords["secondary"] if keyword in query)
        score += secondary_matches * self.weights["secondary_keyword"]

        # Pattern matching
        if query_type in self.patterns:
            pattern_matches = sum(1 for pattern in self.patterns[query_type]
                                if re.search(pattern, query))
            score += pattern_matches * self.weights["pattern_match"]

        # Normalize score (0-1 range)
        max_possible_score = (
            len(keywords["primary"]) * self.weights["primary_keyword"] +
            len(keywords["secondary"]) * self.weights["secondary_keyword"] +
            len(self.patterns.get(query_type, [])) * self.weights["pattern_match"]
        )

        if max_possible_score > 0:
            score = min(score / max_possible_score, 1.0)

        return score

    def _generate_reasoning(self, query: str, query_type: QueryType, confidence: float) -> str:
        """Generate human-readable reasoning for classification decision"""
        reasoning_parts = []

        if confidence > 0.7:
            reasoning_parts.append(f"High confidence classification as {query_type.value}")
        elif confidence > 0.4:
            reasoning_parts.append(f"Moderate confidence classification as {query_type.value}")
        else:
            reasoning_parts.append(f"Low confidence classification as {query_type.value}")

        # Add specific indicators found
        if "failure" in query or "problem" in query:
            reasoning_parts.append("contains problem/failure indicators")
        if "how to" in query:
            reasoning_parts.append("contains procedural request pattern")
        if "what is" in query:
            reasoning_parts.append("contains informational request pattern")

        return "; ".join(reasoning_parts)

class EntityExtractor:
    """
    Implementation of: Entity Processing with Maintenance Domain Vocabulary

    This class extracts maintenance-specific entities from queries using vocabulary
    matching, pattern recognition, and confidence scoring.
    """

    def __init__(self):
        """
        Implementation of: Load maintenance vocabularies, initialize patterns

        Load maintenance entity vocabularies, initialize pattern matching rules,
        set up entity type mappings, and configure extraction parameters.
        """
        # Maintenance entity vocabulary
        self.entity_vocabulary = {
            "equipment": [
                "pump", "motor", "engine", "compressor", "generator", "turbine",
                "boiler", "heat exchanger", "valve", "pipe", "tank", "vessel",
                "bearing", "seal", "gasket", "belt", "gear", "coupling"
            ],
            "components": [
                "impeller", "rotor", "stator", "shaft", "housing", "casing",
                "inlet", "outlet", "filter", "screen", "sensor", "gauge"
            ],
            "materials": [
                "steel", "aluminum", "brass", "rubber", "plastic", "ceramic",
                "oil", "grease", "coolant", "hydraulic fluid", "lubricant"
            ],
            "conditions": [
                "temperature", "pressure", "flow", "level", "vibration", "noise",
                "wear", "corrosion", "contamination", "leak", "blockage"
            ],
            "actions": [
                "replace", "repair", "inspect", "clean", "lubricate", "calibrate",
                "adjust", "tighten", "loosen", "install", "remove"
            ]
        }

        # Entity patterns for multi-word entities
        self.entity_patterns = [
            (r"hydraulic\s+pump", "equipment"),
            (r"centrifugal\s+pump", "equipment"),
            (r"ball\s+bearing", "components"),
            (r"O[_-]?ring", "components"),
            (r"pressure\s+gauge", "components"),
            (r"temperature\s+sensor", "components"),
            (r"motor\s+oil", "materials"),
            (r"cooling\s+system", "equipment")
        ]

        # Entity normalization rules
        self.normalization_rules = {
            "o-ring": "O-ring",
            "o_ring": "O-ring",
            "oring": "O-ring",
            "motor oil": "motor_oil",
            "hydraulic pump": "hydraulic_pump"
        }

        logger.info("EntityExtractor initialized with maintenance vocabulary")

    def extract(self, query: str) -> List[EntityMatch]:
        """
        Implementation of: Process query text for entities with confidence scoring

        Extract maintenance entities from query text, apply patterns, score matches,
        and return structured entity results with confidence and normalization.

        Args:
            query (str): Input query text

        Returns:
            List[EntityMatch]: Extracted entities with metadata
        """
        entities = []
        query_lower = query.lower()

        # Pattern-based extraction for multi-word entities
        for pattern, entity_type in self.entity_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                entities.append(EntityMatch(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=0.9,  # High confidence for pattern matches
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_form=self._normalize_entity(match.group())
                ))

        # Vocabulary-based extraction
        words = re.findall(r'\b\w+\b', query_lower)
        for i, word in enumerate(words):
            # Skip if already found by pattern matching
            if any(entity.start_pos <= query_lower.find(word, sum(len(w) + 1 for w in words[:i])) <= entity.end_pos
                   for entity in entities):
                continue

            # Check vocabulary
            for entity_type, vocabulary in self.entity_vocabulary.items():
                if word in vocabulary:
                    word_start = query_lower.find(word, sum(len(w) + 1 for w in words[:i]))
                    entities.append(EntityMatch(
                        text=word,
                        entity_type=entity_type,
                        confidence=0.8,  # Good confidence for vocabulary matches
                        start_pos=word_start,
                        end_pos=word_start + len(word),
                        normalized_form=self._normalize_entity(word)
                    ))

        # Remove duplicates and sort by position
        entities = self._remove_duplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)

        return entities

    def _normalize_entity(self, entity_text: str) -> str:
        """Apply entity normalization rules"""
        normalized = entity_text.lower().strip()
        return self.normalization_rules.get(normalized, normalized)

    def _remove_duplicate_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove duplicate entities, keeping highest confidence"""
        seen_entities = {}

        for entity in entities:
            key = entity.normalized_form
            if key not in seen_entities or entity.confidence > seen_entities[key].confidence:
                seen_entities[key] = entity

        return list(seen_entities.values())

class QueryEnhancer:
    """
    Implementation of: Main Enhancement Engine coordinating all components

    This class coordinates query classification, entity extraction, and concept
    expansion to transform raw maintenance queries into enhanced, structured queries
    that enable superior retrieval and response generation.
    """

    def __init__(self):
        """
        Implementation of: Initialize enhancement components and configuration

        Initialize all enhancement components, load models and data, configure
        expansion parameters, and set up the enhancement pipeline.
        """
        # Initialize components
        self.classifier = QueryClassifier()
        self.entity_extractor = EntityExtractor()

        # Basic concept expansion rules (simple implementation)
        self.concept_expansion_rules = {
            "pump": ["impeller", "seal", "bearing", "motor", "flow", "pressure"],
            "motor": ["bearing", "rotor", "stator", "winding", "vibration", "temperature"],
            "bearing": ["lubrication", "wear", "noise", "vibration", "seal"],
            "seal": ["leak", "wear", "replacement", "gasket", "O-ring"],
            "failure": ["fault", "malfunction", "problem", "broken", "repair"],
            "maintenance": ["service", "inspection", "repair", "replacement", "preventive"]
        }

        # Enhancement configuration
        self.config = {
            "max_expanded_concepts": 5,
            "min_entity_confidence": 0.5,
            "min_classification_confidence": 0.3
        }

        logger.info("QueryEnhancer initialized with all components")

    def enhance(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Implementation of: Primary enhancement logic coordinating all steps

        This method coordinates all enhancement steps including classification,
        entity extraction, and concept expansion, combining results into a
        structured enhancement output with metadata and confidence scoring.

        Args:
            query (str): Raw maintenance query
            context (Dict, optional): Additional context information

        Returns:
            Dict: Enhanced query structure with all enhancement results
        """
        start_time = datetime.now()

        try:
            # Input validation
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            query = query.strip()
            logger.info(f"Enhancing query: '{query[:100]}...'")

            # Step 1: Query Classification
            classification = self.classifier.classify(query)
            logger.debug(f"Classification: {classification.query_type.value} (confidence: {classification.confidence:.2f})")

            # Step 2: Entity Extraction
            entities = self.entity_extractor.extract(query)
            # Filter by confidence threshold
            entities = [e for e in entities if e.confidence >= self.config["min_entity_confidence"]]
            logger.debug(f"Extracted {len(entities)} entities")

            # Step 3: Concept Expansion
            expanded_concepts = self._expand_concepts(query, entities, classification)
            logger.debug(f"Expanded to {len(expanded_concepts)} concepts")

            # Step 4: Calculate overall enhancement confidence
            overall_confidence = self._calculate_enhancement_confidence(
                classification, entities, expanded_concepts
            )

            # Build enhancement result
            enhancement_result = {
                "original": query,
                "classification": {
                    "type": classification.query_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "secondary_types": [
                        {"type": qtype.value, "confidence": conf}
                        for qtype, conf in classification.secondary_types
                    ]
                },
                "entities": [
                    {
                        "text": entity.text,
                        "type": entity.entity_type,
                        "confidence": entity.confidence,
                        "normalized": entity.normalized_form,
                        "position": [entity.start_pos, entity.end_pos]
                    }
                    for entity in entities
                ],
                "expanded_concepts": expanded_concepts,
                "enhanced": True,
                "confidence": overall_confidence,
                "metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "entity_count": len(entities),
                    "concept_count": len(expanded_concepts),
                    "context_provided": context is not None,
                    "enhancement_version": "1.0.0"
                }
            }

            # Add context information if provided
            if context:
                enhancement_result["context"] = context

            logger.info(f"Enhancement completed successfully (confidence: {overall_confidence:.2f})")
            return enhancement_result

        except Exception as e:
            logger.error(f"Enhancement failed for query '{query}': {str(e)}")
            # Return basic structure even on failure
            return {
                "original": query,
                "enhanced": False,
                "error": str(e),
                "confidence": 0.0,
                "metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "enhancement_version": "1.0.0"
                }
            }

    def _expand_concepts(self, query: str, entities: List[EntityMatch],
                        classification: ClassificationResult) -> List[str]:
        """
        Implementation of: Basic concept expansion logic with domain rules

        Expand concepts using rule-based expansion, maintenance domain synonyms,
        and classification-specific expansion strategies.
        """
        expanded_concepts = set()

        # Expand based on extracted entities
        for entity in entities:
            entity_key = entity.normalized_form
            if entity_key in self.concept_expansion_rules:
                expanded_concepts.update(self.concept_expansion_rules[entity_key][:3])

        # Add query type specific concepts
        if classification.query_type == QueryType.TROUBLESHOOTING:
            expanded_concepts.update(["diagnostic", "symptoms", "root_cause", "solution"])
        elif classification.query_type == QueryType.PROCEDURAL:
            expanded_concepts.update(["procedure", "steps", "tools", "safety"])
        elif classification.query_type == QueryType.PREVENTIVE:
            expanded_concepts.update(["schedule", "inspection", "maintenance", "frequency"])

        # Limit results
        expanded_list = list(expanded_concepts)[:self.config["max_expanded_concepts"]]

        return expanded_list

    def _calculate_enhancement_confidence(self, classification: ClassificationResult,
                                        entities: List[EntityMatch],
                                        expanded_concepts: List[str]) -> float:
        """Calculate overall enhancement confidence based on all components"""
        # Base confidence from classification
        confidence = classification.confidence * 0.4

        # Add entity extraction confidence
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            confidence += avg_entity_confidence * 0.3

        # Add concept expansion confidence
        if expanded_concepts:
            expansion_confidence = min(len(expanded_concepts) / self.config["max_expanded_concepts"], 1.0)
            confidence += expansion_confidence * 0.3

        return min(confidence, 1.0)
