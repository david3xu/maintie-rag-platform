# Knowledge Graph Service - Team Beta
# 🔗 Team Beta: Knowledge Graph Service - FastAPI Application
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: main.py - FastAPI Application and Endpoints
#
# Class: FastAPI App Configuration
# ✅ Configure FastAPI application with title, version, CORS, logging middleware
#
# Function: expand_concepts() - Primary Expansion Endpoint
# ✅ POST /expand endpoint - Accept concepts list, call graph expansion, return ranked results
# ✅ Handle empty/invalid concept lists, add confidence scores
#
# Function: get_entity() - Entity Information Endpoint
# ✅ GET /entities/{entity_id} - Retrieve entity details, relationships, handle not found
#
# Function: health() - Health Check Endpoint
# ✅ Return service status, graph statistics, readiness indicator, performance metrics
#
# Function: startup_event() - Service Initialization
# ✅ Initialize knowledge graph, load processed data, validate completeness, setup logging
# === END TODO REFERENCE ===

from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn

from .core import SimpleKnowledgeGraph, GraphBuilder, ConceptExpander

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Configuration
# Implementation of: Configure FastAPI application with proper setup for graph service
app = FastAPI(
    title="Knowledge Graph Service",
    version="1.0.0",
    description="MaintIE Enhanced RAG - Domain knowledge representation and intelligent concept expansion",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for cross-service communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
knowledge_graph: Optional[SimpleKnowledgeGraph] = None
concept_expander: Optional[ConceptExpander] = None
service_metrics = {
    "startup_time": None,
    "total_requests": 0,
    "successful_requests": 0,
    "average_response_time": 0.0,
    "graph_stats": {
        "entities_loaded": 0,
        "relations_loaded": 0,
        "expansion_cache_size": 0
    }
}

# Middleware for request/response logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Implementation of: Add request/response logging and graph service monitoring

    This middleware tracks request timing, logs all graph operations, and updates
    service metrics for monitoring graph performance and usage patterns.
    """
    start_time = time.time()

    # Log incoming graph request
    logger.info(f"Graph Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate response time
    process_time = time.time() - start_time

    # Update service metrics
    service_metrics["total_requests"] += 1
    if response.status_code == 200:
        service_metrics["successful_requests"] += 1

    # Update average response time
    current_avg = service_metrics["average_response_time"]
    total_requests = service_metrics["total_requests"]
    service_metrics["average_response_time"] = (
        (current_avg * (total_requests - 1) + process_time) / total_requests
    )

    # Log response with graph-specific metrics
    logger.info(f"Graph Response: {response.status_code} - {process_time:.3f}s")

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response

@app.on_event("startup")
async def startup_event():
    """
    Implementation of: Service startup configuration and graph initialization

    This function initializes the knowledge graph instance, loads processed graph data,
    validates graph completeness, and sets up logging and monitoring for graph operations.
    """
    global knowledge_graph, concept_expander, service_metrics

    try:
        logger.info("🔗 Initializing Knowledge Graph Service...")

        # Initialize knowledge graph and load data
        logger.info("Loading MaintIE knowledge graph...")
        knowledge_graph = SimpleKnowledgeGraph()
        await knowledge_graph.initialize()

        # Initialize concept expander
        concept_expander = ConceptExpander(knowledge_graph)

        # Record startup time
        service_metrics["startup_time"] = datetime.now().isoformat()

        # Update graph statistics
        service_metrics["graph_stats"]["entities_loaded"] = knowledge_graph.get_entity_count()
        service_metrics["graph_stats"]["relations_loaded"] = knowledge_graph.get_relation_count()

        # Validate graph readiness with test expansion
        test_concepts = ["pump", "seal"]
        test_expansion = concept_expander.expand_concepts(test_concepts)

        if not test_expansion:
            logger.warning("Graph expansion test returned empty results")

        logger.info("✅ Knowledge Graph Service ready!")
        logger.info(f"Graph loaded: {service_metrics['graph_stats']['entities_loaded']} entities, "
                   f"{service_metrics['graph_stats']['relations_loaded']} relations")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Knowledge Graph Service: {str(e)}")
        raise

@app.post("/expand")
async def expand_concepts(concepts: List[str], depth: Optional[int] = 2, max_results: Optional[int] = 10):
    """
    Implementation of: POST /expand endpoint - Primary Expansion Logic

    This endpoint accepts a list of maintenance concepts and returns intelligently
    expanded related concepts using knowledge graph traversal. Handles empty or
    invalid concept lists and provides confidence scores for expansion results.

    Args:
        concepts (List[str]): List of concepts to expand
        depth (int, optional): Expansion depth (1-3 hops). Defaults to 2.
        max_results (int, optional): Maximum results to return. Defaults to 10.

    Returns:
        Dict: Expanded concepts with relevance scores and metadata
    """
    start_time = time.time()

    try:
        # Input validation
        if not concepts or not isinstance(concepts, list):
            raise HTTPException(
                status_code=400,
                detail="Concepts parameter must be a non-empty list"
            )

        # Filter empty or invalid concepts
        valid_concepts = [c.strip() for c in concepts if c and isinstance(c, str) and c.strip()]
        if not valid_concepts:
            raise HTTPException(
                status_code=400,
                detail="No valid concepts provided"
            )

        # Validate service readiness
        if not concept_expander or not knowledge_graph:
            raise HTTPException(
                status_code=503,
                detail="Knowledge graph service not ready. Please try again later."
            )

        # Validate parameters
        depth = max(1, min(depth or 2, 3))  # Limit depth between 1-3
        max_results = max(1, min(max_results or 10, 50))  # Limit results between 1-50

        logger.info(f"Expanding concepts: {valid_concepts} (depth={depth}, max_results={max_results})")

        # Call core expansion logic
        expansion_result = concept_expander.expand_concepts(
            valid_concepts,
            depth=depth,
            max_results=max_results
        )

        # Build response with metadata
        response = {
            "original_concepts": valid_concepts,
            "expanded_concepts": expansion_result["expanded_concepts"],
            "expansion_metadata": {
                "total_found": len(expansion_result["expanded_concepts"]),
                "expansion_depth": depth,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "graph_stats": {
                    "nodes_traversed": expansion_result.get("nodes_traversed", 0),
                    "paths_explored": expansion_result.get("paths_explored", 0)
                }
            }
        }

        logger.info(f"Expansion completed: {len(expansion_result['expanded_concepts'])} concepts found "
                   f"in {response['expansion_metadata']['processing_time']:.3f}s")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Expansion error for concepts {concepts}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal expansion error: {str(e)}"
        )

@app.get("/entities/{entity_id}")
async def get_entity(entity_id: str = Path(..., description="Entity identifier to retrieve")):
    """
    Implementation of: GET /entities/{entity_id} - Entity Information Endpoint

    This endpoint retrieves detailed information about a specific entity including
    its relationships, properties, and metadata. Handles entity not found cases
    and provides comprehensive entity details.

    Args:
        entity_id (str): Unique identifier for the entity

    Returns:
        Dict: Entity details with relationships and properties
    """
    try:
        # Input validation
        if not entity_id or not entity_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Entity ID is required"
            )

        entity_id = entity_id.strip().lower()

        # Validate service readiness
        if not knowledge_graph:
            raise HTTPException(
                status_code=503,
                detail="Knowledge graph service not ready"
            )

        logger.info(f"Retrieving entity information for: {entity_id}")

        # Get entity details from knowledge graph
        entity_info = knowledge_graph.get_entity_details(entity_id)

        if not entity_info:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity_id}' not found in knowledge graph"
            )

        # Enrich with relationship information
        relationships = knowledge_graph.get_entity_relationships(entity_id)

        # Build comprehensive response
        response = {
            "entity_id": entity_id,
            "entity_info": entity_info,
            "relationships": relationships,
            "metadata": {
                "relationship_count": len(relationships),
                "entity_type": entity_info.get("type", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"Entity '{entity_id}' retrieved with {len(relationships)} relationships")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity retrieval error for '{entity_id}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving entity: {str(e)}"
        )

@app.get("/health")
async def health():
    """
    Implementation of: GET /health endpoint with comprehensive graph status

    This endpoint provides detailed health information including service status,
    graph statistics, readiness indicators, and performance metrics for monitoring
    and load balancing decisions.

    Returns:
        Dict: Comprehensive service health status and graph metrics
    """
    try:
        # Basic health status
        health_status = {
            "status": "healthy",
            "service": "knowledge-graph",
            "timestamp": datetime.now().isoformat(),
            "uptime_since": service_metrics["startup_time"]
        }

        # Add service metrics
        health_status.update({
            "metrics": {
                "total_requests": service_metrics["total_requests"],
                "successful_requests": service_metrics["successful_requests"],
                "success_rate": (
                    service_metrics["successful_requests"] / service_metrics["total_requests"]
                    if service_metrics["total_requests"] > 0 else 0.0
                ),
                "average_response_time": service_metrics["average_response_time"]
            }
        })

        # Add graph-specific statistics
        health_status.update({
            "graph_status": {
                "entities_loaded": service_metrics["graph_stats"]["entities_loaded"],
                "relations_loaded": service_metrics["graph_stats"]["relations_loaded"],
                "graph_ready": knowledge_graph is not None,
                "expander_ready": concept_expander is not None
            }
        })

        # Validate graph functionality
        if knowledge_graph and concept_expander:
            try:
                # Test basic graph operations
                test_expansion = concept_expander.expand_concepts(["pump"], depth=1, max_results=3)
                health_status["functionality_test"] = "passed"
                health_status["graph_status"]["test_expansion_count"] = len(test_expansion.get("expanded_concepts", []))
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["functionality_test"] = f"failed: {str(e)}"
        else:
            health_status["status"] = "degraded"
            health_status["issues"] = ["Knowledge graph or expander not initialized"]

        return health_status

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "service": "knowledge-graph",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/graph/stats")
async def get_graph_statistics():
    """
    Additional endpoint for detailed graph statistics and analysis

    Returns:
        Dict: Detailed graph statistics and analysis
    """
    try:
        if not knowledge_graph:
            raise HTTPException(status_code=503, detail="Knowledge graph not ready")

        stats = knowledge_graph.get_detailed_statistics()

        return {
            "graph_statistics": stats,
            "timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/query")
async def query_graph(query: Dict[str, Any]):
    """
    Advanced graph querying endpoint for complex graph operations

    Args:
        query (Dict): Graph query specification

    Returns:
        Dict: Query results
    """
    try:
        if not knowledge_graph:
            raise HTTPException(status_code=503, detail="Knowledge graph not ready")

        # Validate query structure
        if not query or "operation" not in query:
            raise HTTPException(status_code=400, detail="Query must specify operation")

        # Execute graph query
        results = knowledge_graph.execute_query(query)

        return {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Knowledge Graph Service",
        "version": "1.0.0",
        "description": "MaintIE Enhanced RAG - Domain knowledge representation and concept expansion",
        "endpoints": {
            "expand": "POST /expand - Expand concepts using knowledge graph",
            "entities": "GET /entities/{id} - Get entity details",
            "stats": "GET /graph/stats - Get graph statistics",
            "query": "POST /graph/query - Advanced graph queries",
            "health": "GET /health - Service health status",
            "docs": "GET /docs - API documentation"
        },
        "graph_status": {
            "ready": knowledge_graph is not None,
            "entities": service_metrics["graph_stats"]["entities_loaded"],
            "relations": service_metrics["graph_stats"]["relations_loaded"]
        }
    }

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
