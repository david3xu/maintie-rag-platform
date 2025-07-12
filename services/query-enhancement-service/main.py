# Query Enhancement Service - Team Alpha
# 🧠 Team Alpha: Query Enhancement Service - FastAPI Application
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: main.py - FastAPI Application Entry Point
#
# Class: FastAPI App Configuration
# ✅ Configure FastAPI application with title, version, CORS, middleware
#
# Function: enhance_query() - Main Enhancement Endpoint
# ✅ POST /enhance endpoint implementation
# ✅ Accept query string parameter, call core logic, return structured results
# ✅ Handle error cases gracefully, add request/response logging
#
# Function: health() - Health Check Endpoint
# ✅ GET /health endpoint with service status, timestamp, basic metrics
#
# Function: startup_event() - Service Initialization
# ✅ Initialize core enhancer, load models/data, setup logging, validate dependencies
# === END TODO REFERENCE ===

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import uvicorn

from .core import QueryEnhancer, EntityExtractor, QueryClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Configuration
# Implementation of: Configure FastAPI application with proper setup
app = FastAPI(
    title="Query Enhancement Service",
    version="1.0.0",
    description="MaintIE Enhanced RAG - Intelligent maintenance query understanding and enhancement",
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
enhancer: Optional[QueryEnhancer] = None
service_metrics = {
    "startup_time": None,
    "total_requests": 0,
    "successful_requests": 0,
    "average_response_time": 0.0
}

# Middleware for request/response logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Implementation of: Add request/response logging and performance monitoring

    This middleware tracks request timing, logs all requests, and updates service metrics
    for monitoring and debugging purposes.
    """
    start_time = time.time()

    # Log incoming request
    logger.info(f"Request: {request.method} {request.url.path}")

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

    # Log response
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response

@app.on_event("startup")
async def startup_event():
    """
    Implementation of: Service startup configuration and initialization

    This function initializes all core enhancement components, loads required models,
    sets up logging configuration, and validates service dependencies on startup.
    """
    global enhancer, service_metrics

    try:
        logger.info("🧠 Initializing Query Enhancement Service...")

        # Initialize core enhancement components
        logger.info("Loading query enhancement components...")
        enhancer = QueryEnhancer()

        # Record startup time
        service_metrics["startup_time"] = datetime.now().isoformat()

        # Validate service readiness
        test_query = "test pump failure"
        test_result = enhancer.enhance(test_query)

        if not test_result or "enhanced" not in test_result:
            raise Exception("Enhancement validation failed")

        logger.info("✅ Query Enhancement Service ready!")
        logger.info(f"Service metrics initialized: {service_metrics}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Query Enhancement Service: {str(e)}")
        raise

@app.post("/enhance")
async def enhance_query(query: str, context: Optional[Dict[str, Any]] = None):
    """
    Implementation of: POST /enhance endpoint - Main Enhancement Logic

    This endpoint accepts maintenance queries and transforms them into structured,
    enhanced queries with classification, entity extraction, and concept expansion.
    Handles error cases gracefully and provides detailed enhancement results.

    Args:
        query (str): Raw maintenance query from user
        context (Dict, optional): Additional context information

    Returns:
        Dict: Enhanced query structure with classification, entities, and expansion
    """
    start_time = time.time()

    try:
        # Input validation
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query parameter is required and cannot be empty"
            )

        # Validate enhancer is available
        if not enhancer:
            raise HTTPException(
                status_code=503,
                detail="Enhancement service not ready. Please try again later."
            )

        logger.info(f"Processing enhancement request for query: '{query[:50]}...'")

        # Call core enhancement logic
        enhancement_result = enhancer.enhance(query, context)

        # Add metadata
        enhancement_result.update({
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0"
        })

        logger.info(f"Enhancement completed successfully in {enhancement_result['processing_time']:.3f}s")

        return enhancement_result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Enhancement error for query '{query}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal enhancement error: {str(e)}"
        )

@app.get("/health")
async def health():
    """
    Implementation of: GET /health endpoint with comprehensive service status

    This endpoint provides service health information including status, timestamp,
    basic metrics, and service readiness for monitoring and load balancing.

    Returns:
        Dict: Service health status and performance metrics
    """
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "service": "query-enhancement",
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

        # Validate enhancer availability
        if not enhancer:
            health_status["status"] = "degraded"
            health_status["issues"] = ["Enhancement engine not initialized"]

        # Test basic functionality
        try:
            test_result = enhancer.enhance("test query") if enhancer else None
            health_status["functionality_test"] = "passed" if test_result else "failed"
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
                "service": "query-enhancement",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/classify")
async def classify_query(query: str):
    """
    Additional endpoint for standalone query classification

    Args:
        query (str): Query to classify

    Returns:
        Dict: Classification results with confidence
    """
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query parameter required")

        if not enhancer:
            raise HTTPException(status_code=503, detail="Service not ready")

        # Use the classifier component directly
        classification = enhancer.classifier.classify(query)

        return {
            "query": query,
            "classification": classification,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Query Enhancement Service",
        "version": "1.0.0",
        "description": "MaintIE Enhanced RAG - Intelligent maintenance query understanding",
        "endpoints": {
            "enhance": "POST /enhance - Enhance maintenance queries",
            "classify": "GET /classify - Classify query type",
            "health": "GET /health - Service health status",
            "docs": "GET /docs - API documentation"
        },
        "status": "ready" if enhancer else "initializing"
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
