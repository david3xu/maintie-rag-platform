# Retrieval Fusion Service - Team Gamma
# 🎯 Team Gamma: Retrieval Fusion Service - FastAPI Application
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: main.py - FastAPI Application and Search Endpoints
#
# Class: FastAPI App Configuration
# ✅ Configure FastAPI application with performance monitoring middleware
#
# Function: search_and_fuse() - Primary Multi-Modal Search
# ✅ POST /search endpoint - Accept enhanced queries, coordinate searches, apply fusion
# ✅ Return ranked results with confidence scores, handle timeouts gracefully
#
# Function: vector_search() - Vector Search Endpoint
# ✅ POST /search/vector - Accept text queries, return similarity results
#
# Function: entity_search() - Entity-Based Search
# ✅ POST /search/entity - Accept entity lists, return entity-matched documents
#
# Function: health() - Health Check Endpoint
# ✅ Return service status, search readiness, performance metrics, fusion algorithm status
#
# Function: startup_event() - Service Initialization
# ✅ Initialize search components, load embeddings and indices, validate readiness
# === END TODO REFERENCE ===

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn

from .core import SimpleFusion, VectorSearchEngine, EntitySearchEngine, GraphSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Configuration
# Implementation of: Configure FastAPI application with performance monitoring
app = FastAPI(
    title="Retrieval Fusion Service",
    version="1.0.0",
    description="MaintIE Enhanced RAG - Multi-modal retrieval and intelligent result fusion",
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
fusion_engine: Optional[SimpleFusion] = None
vector_engine: Optional[VectorSearchEngine] = None
entity_engine: Optional[EntitySearchEngine] = None
graph_engine: Optional[GraphSearchEngine] = None

service_metrics = {
    "startup_time": None,
    "total_requests": 0,
    "successful_requests": 0,
    "average_response_time": 0.0,
    "search_stats": {
        "vector_searches": 0,
        "entity_searches": 0,
        "graph_searches": 0,
        "fusion_operations": 0,
        "cache_hits": 0
    }
}

# Middleware for request/response logging and performance monitoring
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Implementation of: Add performance monitoring middleware for search operations

    This middleware tracks search request timing, logs all search operations, and updates
    service metrics specifically for retrieval and fusion performance monitoring.
    """
    start_time = time.time()

    # Log incoming search request
    endpoint = request.url.path
    logger.info(f"Search Request: {request.method} {endpoint}")

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

    # Log search-specific performance
    if "search" in endpoint:
        logger.info(f"Search Response: {response.status_code} - {process_time:.3f}s")

        # Update search-specific metrics
        if "vector" in endpoint:
            service_metrics["search_stats"]["vector_searches"] += 1
        elif "entity" in endpoint:
            service_metrics["search_stats"]["entity_searches"] += 1
        elif "graph" in endpoint:
            service_metrics["search_stats"]["graph_searches"] += 1
        elif endpoint == "/search":
            service_metrics["search_stats"]["fusion_operations"] += 1

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response

@app.on_event("startup")
async def startup_event():
    """
    Implementation of: Service startup configuration and search engine initialization

    This function initializes all search components (vector, entity, graph), loads
    embeddings and indices, validates search engine readiness, and sets up
    performance monitoring systems for retrieval operations.
    """
    global fusion_engine, vector_engine, entity_engine, graph_engine, service_metrics

    try:
        logger.info("🎯 Initializing Retrieval Fusion Service...")

        # Initialize vector search engine
        logger.info("Loading vector search engine...")
        vector_engine = VectorSearchEngine()
        await vector_engine.initialize()

        # Initialize entity search engine
        logger.info("Loading entity search engine...")
        entity_engine = EntitySearchEngine()
        await entity_engine.initialize()

        # Initialize graph search engine
        logger.info("Loading graph search engine...")
        graph_engine = GraphSearchEngine()
        await graph_engine.initialize()

        # Initialize fusion engine
        logger.info("Initializing fusion engine...")
        fusion_engine = SimpleFusion(vector_engine, entity_engine, graph_engine)

        # Record startup time
        service_metrics["startup_time"] = datetime.now().isoformat()

        # Validate search functionality with test queries
        test_query = {"original": "test pump failure", "entities": ["pump", "failure"]}
        test_results = fusion_engine.search_and_fuse(test_query)

        if not test_results or "results" not in test_results:
            raise Exception("Fusion engine validation failed")

        logger.info("✅ Retrieval Fusion Service ready!")
        logger.info("All search engines initialized and validated")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Retrieval Fusion Service: {str(e)}")
        raise

@app.post("/search")
async def search_and_fuse(enhanced_query: Dict[str, Any], max_results: Optional[int] = 5):
    """
    Implementation of: POST /search endpoint - Primary Multi-Modal Search

    This endpoint accepts enhanced query structures from the query enhancement service,
    coordinates vector, entity, and graph search operations, applies intelligent fusion
    algorithm, and returns ranked result sets with confidence scores and metadata.
    Handles search timeouts and failures gracefully.

    Args:
        enhanced_query (Dict): Enhanced query structure with classification and entities
        max_results (int, optional): Maximum results to return. Defaults to 5.

    Returns:
        Dict: Ranked document results with relevance scores and fusion metadata
    """
    start_time = time.time()

    try:
        # Input validation
        if not enhanced_query or not isinstance(enhanced_query, dict):
            raise HTTPException(
                status_code=400,
                detail="Enhanced query structure is required"
            )

        # Validate required fields
        required_fields = ["original"]
        missing_fields = [field for field in required_fields if field not in enhanced_query]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {missing_fields}"
            )

        # Validate service readiness
        if not fusion_engine:
            raise HTTPException(
                status_code=503,
                detail="Fusion engine not ready. Please try again later."
            )

        # Validate and set parameters
        max_results = max(1, min(max_results or 5, 20))  # Limit between 1-20

        logger.info(f"Processing multi-modal search for query: '{enhanced_query.get('original', '')[:50]}...'")

        # Execute fusion search
        search_results = fusion_engine.search_and_fuse(enhanced_query, max_results=max_results)

        # Add response metadata
        response = {
            "query": enhanced_query,
            "results": search_results["results"],
            "fusion_metadata": {
                "total_found": search_results.get("total_found", 0),
                "fusion_strategy": search_results.get("fusion_strategy", "adaptive"),
                "search_modes_used": search_results.get("search_modes_used", []),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "confidence_scores": search_results.get("confidence_scores", {}),
                "search_statistics": search_results.get("search_statistics", {})
            }
        }

        logger.info(f"Multi-modal search completed: {len(search_results['results'])} results "
                   f"in {response['fusion_metadata']['processing_time']:.3f}s")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Search error for query '{enhanced_query.get('original', '')}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal search error: {str(e)}"
        )

@app.post("/search/vector")
async def vector_search(query: str, max_results: Optional[int] = 10):
    """
    Implementation of: POST /search/vector - Vector Search Endpoint

    This endpoint accepts text queries for semantic search, returns vector similarity
    results with similarity scores, handles empty result sets, and includes
    performance timing metrics for vector operations.

    Args:
        query (str): Text query for semantic similarity search
        max_results (int, optional): Maximum results to return. Defaults to 10.

    Returns:
        Dict: Vector similarity results with scores and metadata
    """
    start_time = time.time()

    try:
        # Input validation
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query parameter is required and cannot be empty"
            )

        # Validate service readiness
        if not vector_engine:
            raise HTTPException(
                status_code=503,
                detail="Vector search engine not ready"
            )

        # Validate parameters
        max_results = max(1, min(max_results or 10, 50))

        logger.info(f"Processing vector search for: '{query[:50]}...'")

        # Execute vector search
        search_results = vector_engine.search(query, max_results=max_results)

        # Build response
        response = {
            "query": query,
            "results": search_results["results"],
            "metadata": {
                "total_found": len(search_results["results"]),
                "search_type": "vector_similarity",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "similarity_threshold": search_results.get("similarity_threshold", 0.0),
                "embedding_model": search_results.get("embedding_model", "unknown")
            }
        }

        logger.info(f"Vector search completed: {len(search_results['results'])} results")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/entity")
async def entity_search(entities: List[str], max_results: Optional[int] = 10):
    """
    Implementation of: POST /search/entity - Entity-Based Search Endpoint

    This endpoint accepts entity lists for exact matching, returns documents containing
    specified entities with match confidence, handles entity variations and synonyms,
    and supports multi-entity queries with intersection logic.

    Args:
        entities (List[str]): List of entities to search for
        max_results (int, optional): Maximum results to return. Defaults to 10.

    Returns:
        Dict: Entity-matched documents with match confidence and metadata
    """
    start_time = time.time()

    try:
        # Input validation
        if not entities or not isinstance(entities, list):
            raise HTTPException(
                status_code=400,
                detail="Entities parameter must be a non-empty list"
            )

        # Filter valid entities
        valid_entities = [e.strip() for e in entities if e and isinstance(e, str) and e.strip()]
        if not valid_entities:
            raise HTTPException(
                status_code=400,
                detail="No valid entities provided"
            )

        # Validate service readiness
        if not entity_engine:
            raise HTTPException(
                status_code=503,
                detail="Entity search engine not ready"
            )

        # Validate parameters
        max_results = max(1, min(max_results or 10, 50))

        logger.info(f"Processing entity search for: {valid_entities}")

        # Execute entity search
        search_results = entity_engine.search(valid_entities, max_results=max_results)

        # Build response
        response = {
            "entities": valid_entities,
            "results": search_results["results"],
            "metadata": {
                "total_found": len(search_results["results"]),
                "search_type": "entity_matching",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "entity_match_strategy": search_results.get("match_strategy", "exact"),
                "entities_found": search_results.get("entities_found", []),
                "entities_missing": search_results.get("entities_missing", [])
            }
        }

        logger.info(f"Entity search completed: {len(search_results['results'])} results")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/graph")
async def graph_search(concepts: List[str], depth: Optional[int] = 2, max_results: Optional[int] = 10):
    """
    Implementation of: POST /search/graph - Knowledge Graph Search Endpoint

    This endpoint performs knowledge graph-based search using concept expansion
    and relationship traversal to find relevant documents through graph reasoning.

    Args:
        concepts (List[str]): Concepts to expand via knowledge graph
        depth (int, optional): Graph traversal depth. Defaults to 2.
        max_results (int, optional): Maximum results to return. Defaults to 10.

    Returns:
        Dict: Graph-informed search results with concept expansion metadata
    """
    start_time = time.time()

    try:
        # Input validation
        if not concepts or not isinstance(concepts, list):
            raise HTTPException(
                status_code=400,
                detail="Concepts parameter must be a non-empty list"
            )

        # Validate service readiness
        if not graph_engine:
            raise HTTPException(
                status_code=503,
                detail="Graph search engine not ready"
            )

        # Validate parameters
        depth = max(1, min(depth or 2, 3))
        max_results = max(1, min(max_results or 10, 50))

        logger.info(f"Processing graph search for concepts: {concepts}")

        # Execute graph search
        search_results = graph_engine.search(concepts, depth=depth, max_results=max_results)

        # Build response
        response = {
            "concepts": concepts,
            "results": search_results["results"],
            "metadata": {
                "total_found": len(search_results["results"]),
                "search_type": "graph_traversal",
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "expansion_depth": depth,
                "expanded_concepts": search_results.get("expanded_concepts", []),
                "graph_statistics": search_results.get("graph_statistics", {})
            }
        }

        logger.info(f"Graph search completed: {len(search_results['results'])} results")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """
    Implementation of: GET /health endpoint with comprehensive search service status

    This endpoint provides detailed health information including service status,
    search readiness indicators, performance metrics, and fusion algorithm status
    for monitoring and load balancing decisions.

    Returns:
        Dict: Comprehensive service health status and search engine metrics
    """
    try:
        # Basic health status
        health_status = {
            "status": "healthy",
            "service": "retrieval-fusion",
            "timestamp": datetime.now().isoformat(),
            "uptime_since": service_metrics["startup_time"]
        }

        # Add service performance metrics
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

        # Add search engine status
        search_engines_status = {
            "vector_engine": vector_engine is not None and vector_engine.is_ready(),
            "entity_engine": entity_engine is not None and entity_engine.is_ready(),
            "graph_engine": graph_engine is not None and graph_engine.is_ready(),
            "fusion_engine": fusion_engine is not None
        }

        health_status["search_engines"] = search_engines_status

        # Add search statistics
        health_status["search_statistics"] = service_metrics["search_stats"]

        # Overall readiness assessment
        all_engines_ready = all(search_engines_status.values())
        if not all_engines_ready:
            health_status["status"] = "degraded"
            health_status["issues"] = [
                f"{engine}_engine not ready" for engine, ready
                in search_engines_status.items() if not ready
            ]

        # Test search functionality if all engines ready
        if all_engines_ready:
            try:
                test_query = {"original": "test query", "entities": ["test"]}
                test_results = fusion_engine.search_and_fuse(test_query, max_results=1)
                health_status["functionality_test"] = "passed"
                health_status["test_results_count"] = len(test_results.get("results", []))
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["functionality_test"] = f"failed: {str(e)}"

        return health_status

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "service": "retrieval-fusion",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/fusion/strategies")
async def get_fusion_strategies():
    """
    Additional endpoint for fusion strategy information and configuration

    Returns:
        Dict: Available fusion strategies and current configuration
    """
    try:
        if not fusion_engine:
            raise HTTPException(status_code=503, detail="Fusion engine not ready")

        strategies = fusion_engine.get_available_strategies()

        return {
            "available_strategies": strategies,
            "current_configuration": fusion_engine.get_current_config(),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fusion strategies error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Retrieval Fusion Service",
        "version": "1.0.0",
        "description": "MaintIE Enhanced RAG - Multi-modal retrieval and intelligent result fusion",
        "endpoints": {
            "search": "POST /search - Multi-modal search with fusion",
            "vector": "POST /search/vector - Vector similarity search",
            "entity": "POST /search/entity - Entity-based search",
            "graph": "POST /search/graph - Knowledge graph search",
            "strategies": "GET /fusion/strategies - Fusion strategy info",
            "health": "GET /health - Service health status",
            "docs": "GET /docs - API documentation"
        },
        "search_engines_status": {
            "vector_engine": vector_engine is not None,
            "entity_engine": entity_engine is not None,
            "graph_engine": graph_engine is not None,
            "fusion_engine": fusion_engine is not None
        },
        "performance_metrics": {
            "total_searches": service_metrics["search_stats"]["fusion_operations"],
            "average_response_time": service_metrics["average_response_time"]
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
