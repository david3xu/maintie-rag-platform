# Response Generation Service - Team Delta
# 📝 Team Delta: Response Generation Service - FastAPI Application
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: main.py - FastAPI Application and Generation Endpoints
#
# Class: FastAPI App Configuration
# ✅ Configure FastAPI application with response timing and quality monitoring
#
# Function: generate_response() - Primary Generation Endpoint
# ✅ POST /generate endpoint - Accept context from retrieval, generate domain-aware responses
# ✅ Include response confidence and source attribution, handle LLM API failures gracefully
#
# Function: validate_response() - Response Quality Validation
# ✅ POST /validate endpoint - Apply domain validation rules, check safety considerations
#
# Function: get_templates() - Template Management
# ✅ GET /templates endpoint - Return response templates by query type
#
# Function: health() - Health Check Endpoint
# ✅ Return service status, LLM connectivity, generation performance metrics
#
# Function: startup_event() - Service Initialization
# ✅ Initialize LLM client connections, load templates, validate API keys, setup monitoring
# === END TODO REFERENCE ===

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn
import os

from .core import SimpleGenerator, PromptBuilder, LLMInterface, ResponseValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Configuration
# Implementation of: Configure FastAPI application with response monitoring
app = FastAPI(
    title="Response Generation Service",
    version="1.0.0",
    description="MaintIE Enhanced RAG - Domain-aware response generation with maintenance expertise",
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
generator: Optional[SimpleGenerator] = None
prompt_builder: Optional[PromptBuilder] = None
response_validator: Optional[ResponseValidator] = None
llm_interface: Optional[LLMInterface] = None

service_metrics = {
    "startup_time": None,
    "total_requests": 0,
    "successful_requests": 0,
    "average_response_time": 0.0,
    "generation_stats": {
        "total_generations": 0,
        "successful_generations": 0,
        "average_generation_time": 0.0,
        "total_tokens_used": 0,
        "api_errors": 0,
        "quality_scores": []
    }
}

# Middleware for request/response logging and response quality monitoring
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Implementation of: Add response timing and quality monitoring middleware

    This middleware tracks response generation timing, logs all generation operations,
    and updates service metrics specifically for LLM and response quality monitoring.
    """
    start_time = time.time()

    # Log incoming generation request
    endpoint = request.url.path
    logger.info(f"Generation Request: {request.method} {endpoint}")

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

    # Log generation-specific performance
    if endpoint == "/generate":
        logger.info(f"Generation Response: {response.status_code} - {process_time:.3f}s")
        service_metrics["generation_stats"]["total_generations"] += 1

        if response.status_code == 200:
            service_metrics["generation_stats"]["successful_generations"] += 1

            # Update average generation time
            current_gen_avg = service_metrics["generation_stats"]["average_generation_time"]
            total_gens = service_metrics["generation_stats"]["total_generations"]
            service_metrics["generation_stats"]["average_generation_time"] = (
                (current_gen_avg * (total_gens - 1) + process_time) / total_gens
            )

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response

@app.on_event("startup")
async def startup_event():
    """
    Implementation of: Service startup configuration and LLM initialization

    This function initializes LLM client connections, loads response templates and prompts,
    validates API key configuration, and sets up quality monitoring systems for
    response generation operations.
    """
    global generator, prompt_builder, response_validator, llm_interface, service_metrics

    try:
        logger.info("📝 Initializing Response Generation Service...")

        # Validate API key configuration
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. LLM functionality will be limited.")

        # Initialize LLM interface
        logger.info("Initializing LLM interface...")
        llm_interface = LLMInterface()
        await llm_interface.initialize()

        # Initialize prompt builder
        logger.info("Loading response templates and prompts...")
        prompt_builder = PromptBuilder()
        await prompt_builder.initialize()

        # Initialize response validator
        logger.info("Setting up response validation...")
        response_validator = ResponseValidator()
        await response_validator.initialize()

        # Initialize main generator
        logger.info("Initializing response generator...")
        generator = SimpleGenerator(llm_interface, prompt_builder, response_validator)

        # Record startup time
        service_metrics["startup_time"] = datetime.now().isoformat()

        # Test generation functionality
        test_context = {"results": ["Test maintenance document"]}
        test_query = "test query"
        test_response = generator.generate(test_context, test_query)

        if not test_response or "answer" not in test_response:
            logger.warning("Generation test returned unexpected results")

        logger.info("✅ Response Generation Service ready!")
        logger.info("LLM interface initialized and response templates loaded")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Response Generation Service: {str(e)}")
        raise

@app.post("/generate")
async def generate_response(context: Dict[str, Any], query: str,
                          query_type: Optional[str] = None,
                          max_tokens: Optional[int] = 500):
    """
    Implementation of: POST /generate endpoint - Primary Generation Logic

    This endpoint accepts context from retrieval fusion service and original user query,
    generates domain-aware maintenance responses using LLM integration, includes response
    confidence and source attribution, and handles LLM API failures gracefully with
    comprehensive error recovery.

    Args:
        context (Dict): Context from retrieval fusion with ranked documents
        query (str): Original user query for reference
        query_type (str, optional): Type of query for template selection
        max_tokens (int, optional): Maximum tokens for response generation

    Returns:
        Dict: Generated maintenance response with quality metrics and metadata
    """
    start_time = time.time()

    try:
        # Input validation
        if not context or not isinstance(context, dict):
            raise HTTPException(
                status_code=400,
                detail="Context parameter is required and must be a dictionary"
            )

        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query parameter is required and cannot be empty"
            )

        # Validate service readiness
        if not generator:
            raise HTTPException(
                status_code=503,
                detail="Response generation service not ready. Please try again later."
            )

        # Validate and set parameters
        query = query.strip()
        max_tokens = max(100, min(max_tokens or 500, 1500))  # Limit token range

        logger.info(f"Generating response for query: '{query[:50]}...' (type: {query_type})")

        # Generate response using core logic
        generation_result = generator.generate(
            context=context,
            query=query,
            query_type=query_type,
            max_tokens=max_tokens
        )

        # Track token usage
        tokens_used = generation_result.get("tokens_used", 0)
        service_metrics["generation_stats"]["total_tokens_used"] += tokens_used

        # Add response metadata
        response = {
            "query": query,
            "answer": generation_result["answer"],
            "confidence": generation_result.get("confidence", 0.0),
            "source_attribution": generation_result.get("source_attribution", []),
            "generation_metadata": {
                "query_type": query_type or generation_result.get("detected_query_type", "unknown"),
                "template_used": generation_result.get("template_used", "default"),
                "tokens_used": tokens_used,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "llm_model": generation_result.get("llm_model", "unknown"),
                "safety_check": generation_result.get("safety_check", "passed")
            }
        }

        # Track quality score if available
        quality_score = generation_result.get("quality_score")
        if quality_score is not None:
            service_metrics["generation_stats"]["quality_scores"].append(quality_score)
            response["generation_metadata"]["quality_score"] = quality_score

        logger.info(f"Response generated successfully in {response['generation_metadata']['processing_time']:.3f}s "
                   f"({tokens_used} tokens)")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Generation error for query '{query}': {str(e)}")
        service_metrics["generation_stats"]["api_errors"] += 1

        raise HTTPException(
            status_code=500,
            detail=f"Internal generation error: {str(e)}"
        )

@app.post("/validate")
async def validate_response(response_text: str, query: str, context: Optional[Dict[str, Any]] = None):
    """
    Implementation of: POST /validate endpoint - Response Quality Validation

    This endpoint accepts generated responses for quality checking, applies domain-specific
    validation rules, checks for safety considerations inclusion, validates technical
    accuracy against context, and returns validation results with improvement suggestions.

    Args:
        response_text (str): Generated response to validate
        query (str): Original query for context
        context (Dict, optional): Original context for accuracy checking

    Returns:
        Dict: Validation results with quality scores and improvement suggestions
    """
    start_time = time.time()

    try:
        # Input validation
        if not response_text or not response_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Response text is required and cannot be empty"
            )

        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query parameter is required for validation context"
            )

        # Validate service readiness
        if not response_validator:
            raise HTTPException(
                status_code=503,
                detail="Response validation service not ready"
            )

        logger.info(f"Validating response for query: '{query[:50]}...'")

        # Perform comprehensive validation
        validation_result = response_validator.validate_maintenance_response(
            response_text=response_text,
            query=query,
            context=context
        )

        # Build validation response
        response = {
            "query": query,
            "response_text": response_text,
            "validation_results": {
                "overall_quality": validation_result["overall_quality"],
                "technical_accuracy": validation_result["technical_accuracy"],
                "safety_considerations": validation_result["safety_considerations"],
                "completeness": validation_result["completeness"],
                "clarity": validation_result["clarity"]
            },
            "improvement_suggestions": validation_result.get("improvement_suggestions", []),
            "safety_check": validation_result.get("safety_check", {}),
            "validation_metadata": {
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "validation_version": "1.0.0"
            }
        }

        logger.info(f"Validation completed: overall quality {validation_result['overall_quality']:.2f}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}"
        )

@app.get("/templates")
async def get_templates(query_type: Optional[str] = None):
    """
    Implementation of: GET /templates endpoint - Template Management

    This endpoint returns available response templates by query type, includes template
    usage statistics, supports template customization parameters, and handles template
    not found cases with appropriate fallbacks.

    Args:
        query_type (str, optional): Specific query type to get templates for

    Returns:
        Dict: Available templates with usage statistics and customization options
    """
    try:
        # Validate service readiness
        if not prompt_builder:
            raise HTTPException(
                status_code=503,
                detail="Template service not ready"
            )

        logger.info(f"Retrieving templates for query type: {query_type or 'all'}")

        # Get templates from prompt builder
        if query_type:
            templates = prompt_builder.get_templates_for_type(query_type)
            if not templates:
                raise HTTPException(
                    status_code=404,
                    detail=f"No templates found for query type: {query_type}"
                )
        else:
            templates = prompt_builder.get_all_templates()

        # Get usage statistics
        usage_stats = prompt_builder.get_template_usage_stats()

        # Build response
        response = {
            "query_type": query_type,
            "templates": templates,
            "usage_statistics": usage_stats,
            "available_query_types": prompt_builder.get_supported_query_types(),
            "customization_options": {
                "max_tokens": {"min": 100, "max": 1500, "default": 500},
                "temperature": {"min": 0.0, "max": 1.0, "default": 0.1},
                "safety_level": {"options": ["standard", "high", "critical"], "default": "standard"}
            },
            "metadata": {
                "total_templates": len(templates),
                "timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"Templates retrieved: {len(templates)} templates available")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template retrieval error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal template error: {str(e)}"
        )

@app.get("/health")
async def health():
    """
    Implementation of: GET /health endpoint with comprehensive generation service status

    This endpoint provides detailed health information including service status,
    LLM connectivity status, generation performance metrics, and response quality
    statistics for monitoring and load balancing decisions.

    Returns:
        Dict: Comprehensive service health status and generation metrics
    """
    try:
        # Basic health status
        health_status = {
            "status": "healthy",
            "service": "response-generation",
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

        # Add generation-specific statistics
        gen_stats = service_metrics["generation_stats"]
        health_status["generation_metrics"] = {
            "total_generations": gen_stats["total_generations"],
            "successful_generations": gen_stats["successful_generations"],
            "generation_success_rate": (
                gen_stats["successful_generations"] / gen_stats["total_generations"]
                if gen_stats["total_generations"] > 0 else 0.0
            ),
            "average_generation_time": gen_stats["average_generation_time"],
            "total_tokens_used": gen_stats["total_tokens_used"],
            "api_errors": gen_stats["api_errors"],
            "average_quality_score": (
                sum(gen_stats["quality_scores"]) / len(gen_stats["quality_scores"])
                if gen_stats["quality_scores"] else 0.0
            )
        }

        # Check component status
        component_status = {
            "generator": generator is not None,
            "llm_interface": llm_interface is not None and llm_interface.is_ready(),
            "prompt_builder": prompt_builder is not None,
            "response_validator": response_validator is not None
        }

        health_status["components"] = component_status

        # Overall readiness assessment
        all_components_ready = all(component_status.values())
        if not all_components_ready:
            health_status["status"] = "degraded"
            health_status["issues"] = [
                f"{component} not ready" for component, ready
                in component_status.items() if not ready
            ]

        # Test generation functionality if all components ready
        if all_components_ready:
            try:
                test_context = {"results": ["Test maintenance document"]}
                test_response = generator.generate(test_context, "test query", max_tokens=50)
                health_status["functionality_test"] = "passed"
                health_status["test_response_length"] = len(test_response.get("answer", ""))
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["functionality_test"] = f"failed: {str(e)}"

        # API key status
        health_status["api_status"] = {
            "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "llm_service_available": llm_interface.is_ready() if llm_interface else False
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "service": "response-generation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/metrics")
async def get_detailed_metrics():
    """
    Additional endpoint for detailed generation metrics and cost tracking

    Returns:
        Dict: Detailed generation metrics including cost and quality analytics
    """
    try:
        if not generator:
            raise HTTPException(status_code=503, detail="Service not ready")

        detailed_metrics = {
            "generation_performance": service_metrics["generation_stats"],
            "cost_analytics": {
                "total_tokens": service_metrics["generation_stats"]["total_tokens_used"],
                "estimated_cost": service_metrics["generation_stats"]["total_tokens_used"] * 0.000002,  # Rough estimate
                "average_tokens_per_request": (
                    service_metrics["generation_stats"]["total_tokens_used"] /
                    service_metrics["generation_stats"]["total_generations"]
                    if service_metrics["generation_stats"]["total_generations"] > 0 else 0
                )
            },
            "quality_analytics": {
                "quality_scores": service_metrics["generation_stats"]["quality_scores"][-10:],  # Last 10
                "quality_trend": "improving" if len(service_metrics["generation_stats"]["quality_scores"]) >= 2
                                and service_metrics["generation_stats"]["quality_scores"][-1] >
                                service_metrics["generation_stats"]["quality_scores"][-2] else "stable"
            },
            "timestamp": datetime.now().isoformat()
        }

        return detailed_metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Response Generation Service",
        "version": "1.0.0",
        "description": "MaintIE Enhanced RAG - Domain-aware response generation with maintenance expertise",
        "endpoints": {
            "generate": "POST /generate - Generate maintenance responses",
            "validate": "POST /validate - Validate response quality",
            "templates": "GET /templates - Get response templates",
            "metrics": "GET /metrics - Detailed generation metrics",
            "health": "GET /health - Service health status",
            "docs": "GET /docs - API documentation"
        },
        "service_status": {
            "ready": generator is not None,
            "llm_available": llm_interface is not None and llm_interface.is_ready() if llm_interface else False,
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
        },
        "generation_stats": {
            "total_generations": service_metrics["generation_stats"]["total_generations"],
            "success_rate": (
                service_metrics["generation_stats"]["successful_generations"] /
                service_metrics["generation_stats"]["total_generations"]
                if service_metrics["generation_stats"]["total_generations"] > 0 else 0.0
            )
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
