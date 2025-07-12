# Response Generation Core Logic - Team Delta
# 📝 Team Delta: Response Generation Service - Core LLM Integration & Response Generation
#
# === TODO IMPLEMENTATION REFERENCE ===
# File: core.py - LLM Integration and Response Generation Logic
#
# Class: SimpleGenerator - Main Generation Engine
# ✅ __init__() method - Initialize OpenAI client, load templates, setup validation
# ✅ generate() method - Build prompts, call LLM API, post-process responses, validate
#
# Class: PromptBuilder - Domain-Specific Prompt Engineering
# ✅ build_maintenance_prompt() method - Create domain prompts from context and query
# ✅ build_troubleshooting_prompt() method - Structure diagnostic prompts
# ✅ build_procedural_prompt() method - Format step-by-step instruction prompts
#
# Class: LLMInterface - API Integration
# ✅ generate_with_context() method - Send prompts to LLM, handle errors, track usage
# ✅ validate_api_response() method - Check response completeness and safety
#
# Class: ResponseValidator - Quality Assurance
# ✅ validate_maintenance_response() method - Check technical accuracy, safety inclusion
# ✅ check_safety_considerations() method - Scan for safety warnings, PPE requirements
# === END TODO REFERENCE ===

import os
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query types for prompt selection"""
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURAL = "procedural"
    INFORMATIONAL = "informational"
    PREVENTIVE = "preventive"
    SAFETY = "safety"

@dataclass
class GenerationResult:
    """Data structure for generation results"""
    answer: str
    confidence: float
    tokens_used: int
    template_used: str
    quality_score: Optional[float]
    safety_check: str
    source_attribution: List[str]
    llm_model: str

@dataclass
class ValidationResult:
    """Data structure for response validation results"""
    overall_quality: float
    technical_accuracy: float
    safety_considerations: float
    completeness: float
    clarity: float
    improvement_suggestions: List[str]
    safety_check: Dict[str, Any]

class LLMInterface:
    """
    Implementation of: API Integration with OpenAI and Error Handling

    This class handles OpenAI API integration with error handling, retry logic,
    usage tracking, and response validation for maintenance domain applications.
    """

    def __init__(self):
        """
        Implementation of: Initialize OpenAI client with configuration

        Initialize OpenAI client with API configuration, set up API error handling
        and retry logic, configure model parameters for maintenance domain, and
        implement rate limiting and usage tracking.
        """
        self.client = None
        self.api_key = None
        self.model_name = "gpt-3.5-turbo"
        self.ready = False
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": []
        }

        # Model configuration for maintenance domain
        self.default_params = {
            "temperature": 0.1,  # Low temperature for factual responses
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.1
        }

        logger.info("LLMInterface initialized")

    async def initialize(self):
        """Initialize LLM interface and validate API connectivity"""
        try:
            # Get API key from environment
            self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.api_key:
                logger.warning("OpenAI API key not found. Using fallback mode.")
                self.ready = False
                return

            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=self.api_key)

            # Test API connectivity
            test_successful = await self._test_api_connectivity()

            if test_successful:
                self.ready = True
                logger.info("LLM interface ready with OpenAI API")
            else:
                logger.warning("OpenAI API test failed. Limited functionality available.")
                self.ready = False

        except Exception as e:
            logger.error(f"Failed to initialize LLM interface: {str(e)}")
            self.ready = False

    async def _test_api_connectivity(self) -> bool:
        """Test OpenAI API connectivity with minimal request"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.warning(f"API connectivity test failed: {str(e)}")
            return False

    def generate_with_context(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Implementation of: Send maintenance-optimized prompts to LLM

        Send maintenance-optimized prompts to LLM, handle API rate limiting and errors,
        implement response streaming if needed, track API usage and costs, and return
        structured response with metadata.

        Args:
            prompt (str): Formatted prompt for LLM
            **kwargs: Additional parameters for generation

        Returns:
            Dict: LLM response with metadata and usage information
        """
        start_time = time.time()

        try:
            # Update usage statistics
            self.usage_stats["total_requests"] += 1

            if not self.ready or not self.client:
                # Fallback response when API unavailable
                return self._generate_fallback_response(prompt)

            # Prepare generation parameters
            params = {**self.default_params, **kwargs}

            # Make API request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a maintenance expert assistant. Provide accurate, safety-conscious, and practical maintenance guidance."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                **params
            )

            # Extract response data
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            # Update usage statistics
            self.usage_stats["successful_requests"] += 1
            self.usage_stats["total_tokens"] += tokens_used

            # Estimate cost (rough calculation)
            estimated_cost = tokens_used * 0.000002  # Approximate cost per token
            self.usage_stats["total_cost"] += estimated_cost

            # Validate API response
            validation_result = self._validate_api_response(answer)

            result = {
                "answer": answer,
                "tokens_used": tokens_used,
                "model": self.model_name,
                "processing_time": time.time() - start_time,
                "estimated_cost": estimated_cost,
                "validation": validation_result,
                "api_success": True
            }

            logger.info(f"LLM generation completed: {tokens_used} tokens, {result['processing_time']:.2f}s")

            return result

        except Exception as e:
            # Handle API errors
            error_info = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name
            }
            self.usage_stats["errors"].append(error_info)

            logger.error(f"LLM generation error: {str(e)}")

            # Return fallback response
            return self._generate_fallback_response(prompt, error=str(e))

    def _validate_api_response(self, response: str) -> Dict[str, Any]:
        """
        Implementation of: Check LLM response for completeness and safety

        Check LLM response for completeness, validate response format and structure,
        ensure safety considerations included, check for maintenance domain relevance,
        and handle incomplete or invalid responses.
        """
        validation = {
            "is_complete": True,
            "has_safety_content": False,
            "is_relevant": True,
            "length_appropriate": True,
            "issues": []
        }

        if not response or len(response.strip()) < 10:
            validation["is_complete"] = False
            validation["issues"].append("Response too short")

        # Check for safety-related content
        safety_keywords = ["safety", "caution", "warning", "ppe", "personal protective equipment",
                          "hazard", "risk", "lockout", "tagout"]
        if any(keyword in response.lower() for keyword in safety_keywords):
            validation["has_safety_content"] = True

        # Check response length
        if len(response) > 2000:
            validation["length_appropriate"] = False
            validation["issues"].append("Response too long")

        # Check for maintenance relevance
        maintenance_keywords = ["maintenance", "repair", "service", "inspection", "component",
                               "equipment", "procedure", "diagnosis"]
        if not any(keyword in response.lower() for keyword in maintenance_keywords):
            validation["is_relevant"] = False
            validation["issues"].append("Low maintenance domain relevance")

        return validation

    def _generate_fallback_response(self, prompt: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Generate fallback response when API unavailable"""
        fallback_answer = """
        I apologize, but I'm currently unable to access the full knowledge base to provide a comprehensive answer.

        For maintenance queries, I recommend:
        1. Consulting your equipment manual or documentation
        2. Following established safety procedures
        3. Contacting qualified maintenance personnel if needed

        Please ensure proper safety precautions before performing any maintenance work.
        """

        if error:
            fallback_answer += f"\n\nTechnical note: {error}"

        return {
            "answer": fallback_answer.strip(),
            "tokens_used": 0,
            "model": "fallback",
            "processing_time": 0.1,
            "estimated_cost": 0.0,
            "api_success": False,
            "fallback_used": True
        }

    def is_ready(self) -> bool:
        """Check if LLM interface is ready"""
        return self.ready

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return self.usage_stats.copy()

class PromptBuilder:
    """
    Implementation of: Domain-Specific Prompt Engineering for Maintenance

    This class builds maintenance-specific prompts from context and queries,
    includes relevant document excerpts, adds safety considerations, and applies
    query type specific formatting for optimal LLM performance.
    """

    def __init__(self):
        """
        Implementation of: Load maintenance prompt templates

        Load maintenance prompt templates, initialize context formatting rules,
        set up query type prompt mappings, and configure safety and procedure
        emphasis rules for domain-specific prompt engineering.
        """
        self.templates = {}
        self.usage_stats = {}
        self.supported_query_types = [qt.value for qt in QueryType]

        # Load prompt templates
        self._load_prompt_templates()

        # Safety-related keywords and phrases
        self.safety_keywords = [
            "safety", "caution", "warning", "danger", "hazard", "risk",
            "ppe", "personal protective equipment", "lockout", "tagout",
            "emergency", "first aid", "ventilation", "electrical safety"
        ]

        logger.info("PromptBuilder initialized with maintenance domain templates")

    async def initialize(self):
        """Initialize prompt builder and load templates"""
        try:
            self._load_prompt_templates()
            logger.info("Prompt templates loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize prompt builder: {str(e)}")
            self._create_default_templates()

    def _load_prompt_templates(self):
        """Load prompt templates for different query types"""
        self.templates = {
            QueryType.TROUBLESHOOTING.value: {
                "system_prompt": """You are a maintenance troubleshooting expert. Provide systematic diagnostic guidance with:
1. Clear problem identification
2. Step-by-step diagnostic procedures
3. Safety considerations and warnings
4. Potential causes and solutions
5. When to escalate to specialists""",
                "user_template": """Based on the following maintenance documentation:
{context}

Help diagnose this maintenance issue: {query}

Provide a structured troubleshooting approach including:
- Problem assessment
- Safety precautions
- Diagnostic steps
- Likely causes
- Recommended solutions
- Escalation criteria"""
            },

            QueryType.PROCEDURAL.value: {
                "system_prompt": """You are a maintenance procedures expert. Provide clear, step-by-step instructions with:
1. Required tools and materials
2. Safety requirements and PPE
3. Detailed procedural steps
4. Quality checks and validation
5. Safety warnings and cautions""",
                "user_template": """Based on the following maintenance documentation:
{context}

Provide step-by-step instructions for: {query}

Include:
- Required tools and materials
- Safety requirements and PPE needed
- Detailed step-by-step procedure
- Quality checks and verification steps
- Safety warnings and precautions"""
            },

            QueryType.INFORMATIONAL.value: {
                "system_prompt": """You are a maintenance information specialist. Provide comprehensive, accurate information with:
1. Clear definitions and explanations
2. Technical specifications when relevant
3. Safety considerations
4. Best practices and recommendations
5. Related maintenance considerations""",
                "user_template": """Based on the following maintenance documentation:
{context}

Provide comprehensive information about: {query}

Include:
- Clear explanation or definition
- Technical specifications if applicable
- Safety considerations
- Best practices and recommendations
- Related maintenance information"""
            },

            QueryType.PREVENTIVE.value: {
                "system_prompt": """You are a preventive maintenance expert. Provide systematic maintenance guidance with:
1. Maintenance schedules and intervals
2. Inspection procedures
3. Safety requirements
4. Performance indicators
5. Best practices for reliability""",
                "user_template": """Based on the following maintenance documentation:
{context}

Provide preventive maintenance guidance for: {query}

Include:
- Recommended maintenance schedule
- Inspection procedures and checkpoints
- Safety requirements
- Performance indicators to monitor
- Best practices for equipment reliability"""
            },

            "default": {
                "system_prompt": """You are a maintenance expert assistant. Provide accurate, safety-conscious, and practical maintenance guidance with attention to safety procedures and best practices.""",
                "user_template": """Based on the following maintenance documentation:
{context}

Answer this maintenance question: {query}

Provide practical guidance including safety considerations and best practices."""
            }
        }

    def _create_default_templates(self):
        """Create minimal default templates if loading fails"""
        self.templates = {
            "default": {
                "system_prompt": "You are a helpful maintenance assistant.",
                "user_template": "Context: {context}\n\nQuestion: {query}\n\nPlease provide a helpful answer."
            }
        }

    def build_maintenance_prompt(self, context: Dict[str, Any], query: str,
                                query_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Implementation of: Create domain-specific prompts from context and query

        Create domain-specific prompts from context and query, include relevant document
        excerpts, add maintenance safety considerations, integrate procedural structure
        for how-to queries, and apply query type specific formatting.

        Args:
            context (Dict): Retrieval context with ranked documents
            query (str): User query
            query_type (str, optional): Type of query for template selection

        Returns:
            Tuple[str, str]: (system_prompt, user_prompt)
        """
        # Determine query type and select template
        if query_type and query_type in self.templates:
            template = self.templates[query_type]
            self.usage_stats[query_type] = self.usage_stats.get(query_type, 0) + 1
        else:
            template = self.templates["default"]
            self.usage_stats["default"] = self.usage_stats.get("default", 0) + 1

        # Extract and format context
        formatted_context = self._format_context(context)

        # Build prompts
        system_prompt = template["system_prompt"]
        user_prompt = template["user_template"].format(
            context=formatted_context,
            query=query
        )

        # Add safety emphasis if needed
        if self._query_needs_safety_emphasis(query):
            user_prompt += "\n\nIMPORTANT: Emphasize safety considerations and proper procedures in your response."

        return system_prompt, user_prompt

    def build_troubleshooting_prompt(self, context: Dict[str, Any], query: str) -> Tuple[str, str]:
        """
        Implementation of: Structure prompts for diagnostic scenarios

        Structure prompts for diagnostic scenarios, include symptom-cause-solution framework,
        add safety warnings and precautions, reference specific maintenance procedures,
        and include escalation guidance for complex issues.
        """
        template = self.templates[QueryType.TROUBLESHOOTING.value]
        formatted_context = self._format_context(context)

        # Enhanced troubleshooting prompt with diagnostic framework
        enhanced_user_prompt = template["user_template"].format(
            context=formatted_context,
            query=query
        )

        enhanced_user_prompt += """

Use this diagnostic framework:
1. SYMPTOMS: What are the observable signs?
2. SAFETY: What immediate safety concerns exist?
3. INVESTIGATION: What should be checked first?
4. CAUSES: What are the most likely root causes?
5. SOLUTIONS: What corrective actions are recommended?
6. PREVENTION: How can this be prevented in the future?
7. ESCALATION: When should specialists be contacted?"""

        return template["system_prompt"], enhanced_user_prompt

    def build_procedural_prompt(self, context: Dict[str, Any], query: str) -> Tuple[str, str]:
        """
        Implementation of: Format prompts for step-by-step instructions

        Format prompts for step-by-step instructions, include tool and safety requirements,
        structure clear procedural sequences, add quality check and validation steps,
        and include troubleshooting for common issues.
        """
        template = self.templates[QueryType.PROCEDURAL.value]
        formatted_context = self._format_context(context)

        # Enhanced procedural prompt with structured format
        enhanced_user_prompt = template["user_template"].format(
            context=formatted_context,
            query=query
        )

        enhanced_user_prompt += """

Structure your response as follows:
1. PREPARATION
   - Required tools and materials
   - Safety equipment (PPE) needed
   - Pre-work safety checks

2. PROCEDURE
   - Numbered step-by-step instructions
   - Safety notes for each step
   - Quality checkpoints

3. COMPLETION
   - Final verification steps
   - Clean-up procedures
   - Documentation requirements

4. TROUBLESHOOTING
   - Common issues and solutions
   - When to stop and seek help"""

        return template["system_prompt"], enhanced_user_prompt

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format retrieval context for prompt inclusion"""
        if not context or "results" not in context:
            return "No specific documentation available."

        results = context["results"]
        if not results:
            return "No relevant documents found."

        # Format top results for prompt
        formatted_docs = []
        for i, result in enumerate(results[:3], 1):  # Use top 3 results
            content = result.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            formatted_docs.append(f"Document {i}: {content}")

        return "\n\n".join(formatted_docs)

    def _query_needs_safety_emphasis(self, query: str) -> bool:
        """Check if query involves safety-critical operations"""
        query_lower = query.lower()
        safety_indicators = [
            "electrical", "high voltage", "pressure", "chemical", "hot", "moving parts",
            "lifting", "confined space", "welding", "cutting", "grinding"
        ]
        return any(indicator in query_lower for indicator in safety_indicators)

    def get_templates_for_type(self, query_type: str) -> Dict[str, Any]:
        """Get templates for specific query type"""
        if query_type in self.templates:
            return {query_type: self.templates[query_type]}
        return {}

    def get_all_templates(self) -> Dict[str, Any]:
        """Get all available templates"""
        return self.templates.copy()

    def get_template_usage_stats(self) -> Dict[str, int]:
        """Get template usage statistics"""
        return self.usage_stats.copy()

    def get_supported_query_types(self) -> List[str]:
        """Get list of supported query types"""
        return self.supported_query_types.copy()

class ResponseValidator:
    """
    Implementation of: Quality Assurance with Domain-Specific Validation

    This class validates maintenance responses for technical accuracy, safety
    consideration inclusion, completeness, and domain relevance with specific
    rules for maintenance scenarios.
    """

    def __init__(self):
        """
        Implementation of: Load domain validation rules

        Load domain validation rules, initialize safety keyword checking,
        set up quality scoring mechanisms, and configure improvement
        suggestion templates for comprehensive response validation.
        """
        self.safety_keywords = [
            "safety", "caution", "warning", "danger", "hazard", "risk",
            "ppe", "personal protective equipment", "lockout", "tagout",
            "emergency", "ventilation", "grounding", "isolation"
        ]

        self.quality_criteria = {
            "technical_accuracy": 0.25,
            "safety_considerations": 0.25,
            "completeness": 0.25,
            "clarity": 0.25
        }

        self.maintenance_indicators = [
            "maintenance", "repair", "service", "inspection", "component",
            "equipment", "procedure", "diagnosis", "troubleshoot", "replace"
        ]

        logger.info("ResponseValidator initialized with maintenance domain rules")

    async def initialize(self):
        """Initialize response validator"""
        # Could load additional validation rules or models here
        logger.info("Response validator ready")

    def validate_maintenance_response(self, response_text: str, query: str,
                                    context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Implementation of: Check response for technical accuracy and safety inclusion

        Check response for technical accuracy, validate safety consideration inclusion,
        ensure procedural completeness, score response quality and usefulness, and
        generate improvement recommendations for maintenance domain responses.

        Args:
            response_text (str): Generated response to validate
            query (str): Original query for context
            context (Dict, optional): Original context for accuracy checking

        Returns:
            ValidationResult: Comprehensive validation results with scores
        """
        # Perform individual validation checks
        technical_accuracy = self._check_technical_accuracy(response_text, context)
        safety_considerations = self._check_safety_considerations(response_text, query)
        completeness = self._check_completeness(response_text, query)
        clarity = self._check_clarity(response_text)

        # Calculate overall quality score
        overall_quality = (
            technical_accuracy * self.quality_criteria["technical_accuracy"] +
            safety_considerations * self.quality_criteria["safety_considerations"] +
            completeness * self.quality_criteria["completeness"] +
            clarity * self.quality_criteria["clarity"]
        )

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            response_text, query, technical_accuracy, safety_considerations,
            completeness, clarity
        )

        # Detailed safety check
        safety_check = self._detailed_safety_check(response_text, query)

        return ValidationResult(
            overall_quality=overall_quality,
            technical_accuracy=technical_accuracy,
            safety_considerations=safety_considerations,
            completeness=completeness,
            clarity=clarity,
            improvement_suggestions=suggestions,
            safety_check=safety_check
        )

    def _check_technical_accuracy(self, response: str, context: Optional[Dict[str, Any]]) -> float:
        """Check technical accuracy of response"""
        score = 0.5  # Base score

        # Check for maintenance domain terminology
        maintenance_terms = sum(1 for term in self.maintenance_indicators
                              if term in response.lower())
        if maintenance_terms >= 2:
            score += 0.3

        # Check for specific technical details
        if any(indicator in response.lower() for indicator in
               ["step", "procedure", "check", "inspect", "measure", "replace"]):
            score += 0.2

        # Penalize if response is too generic
        if len(response) < 100:
            score -= 0.2

        return min(max(score, 0.0), 1.0)

    def check_safety_considerations(self, response: str, query: str) -> float:
        """
        Implementation of: Scan for safety warning inclusion

        Scan for safety warning inclusion, validate lockout/tagout references where needed,
        check for PPE requirement mentions, ensure hazard awareness integration, and
        score safety consideration completeness for maintenance responses.
        """
        return self._check_safety_considerations(response, query)

    def _check_safety_considerations(self, response: str, query: str) -> float:
        """Internal method for safety checking"""
        score = 0.0
        response_lower = response.lower()
        query_lower = query.lower()

        # Check for safety keywords
        safety_mentions = sum(1 for keyword in self.safety_keywords
                            if keyword in response_lower)
        if safety_mentions > 0:
            score += 0.4

        # Check for specific safety procedures
        if any(proc in response_lower for proc in ["lockout", "tagout", "ppe", "personal protective"]):
            score += 0.3

        # Check if query suggests safety-critical work
        safety_critical_work = any(indicator in query_lower for indicator in
                                 ["electrical", "pressure", "chemical", "hot", "moving"])

        if safety_critical_work and safety_mentions == 0:
            score = 0.0  # Major penalty for missing safety in critical work
        elif safety_critical_work and safety_mentions > 0:
            score += 0.3  # Bonus for including safety in critical work

        return min(score, 1.0)

    def _check_completeness(self, response: str, query: str) -> float:
        """Check response completeness"""
        score = 0.0

        # Check response length appropriateness
        if 50 <= len(response) <= 1000:
            score += 0.3
        elif len(response) < 50:
            score -= 0.2

        # Check for structured content
        if any(indicator in response for indicator in ["1.", "2.", "3.", "•", "-", "Step"]):
            score += 0.3

        # Check for actionable content
        action_words = ["should", "must", "need to", "ensure", "check", "verify", "replace"]
        if any(word in response.lower() for word in action_words):
            score += 0.2

        # Check for conclusion or summary
        if any(conclusion in response.lower() for conclusion in
               ["in summary", "conclusion", "finally", "remember", "important"]):
            score += 0.2

        return min(max(score, 0.0), 1.0)

    def _check_clarity(self, response: str) -> float:
        """Check response clarity and readability"""
        score = 0.5  # Base score

        # Check sentence structure
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        if 10 <= avg_sentence_length <= 25:  # Optimal sentence length
            score += 0.2

        # Check for clear organization
        if any(org in response for org in ["First", "Next", "Then", "Finally", "1.", "2."]):
            score += 0.2

        # Penalize excessive jargon without explanation
        technical_terms = ["specification", "calibration", "tolerance", "parameter"]
        unexplained_jargon = sum(1 for term in technical_terms
                               if term in response.lower() and f"{term}:" not in response.lower())
        if unexplained_jargon > 2:
            score -= 0.1

        return min(max(score, 0.0), 1.0)

    def _generate_improvement_suggestions(self, response: str, query: str,
                                        technical_accuracy: float, safety_considerations: float,
                                        completeness: float, clarity: float) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []

        if technical_accuracy < 0.7:
            suggestions.append("Include more specific technical details and maintenance procedures")

        if safety_considerations < 0.7:
            suggestions.append("Add safety considerations, PPE requirements, and hazard warnings")

        if completeness < 0.7:
            suggestions.append("Provide more comprehensive coverage of the topic with actionable steps")

        if clarity < 0.7:
            suggestions.append("Improve clarity with better organization and simpler language")

        # Check for missing elements
        if "step" not in response.lower() and "procedure" in query.lower():
            suggestions.append("Include step-by-step procedures for better guidance")

        if len(response) < 100:
            suggestions.append("Expand response with more detailed information and examples")

        return suggestions

    def _detailed_safety_check(self, response: str, query: str) -> Dict[str, Any]:
        """Perform detailed safety analysis"""
        response_lower = response.lower()
        query_lower = query.lower()

        safety_check = {
            "has_safety_warnings": any(word in response_lower for word in
                                     ["warning", "caution", "danger", "hazard"]),
            "mentions_ppe": any(ppe in response_lower for ppe in
                              ["ppe", "personal protective", "safety glasses", "gloves", "helmet"]),
            "includes_lockout_tagout": any(loto in response_lower for loto in
                                         ["lockout", "tagout", "loto", "isolation"]),
            "addresses_electrical_safety": "electrical" in query_lower and
                                         any(elec in response_lower for elec in
                                           ["electrical", "voltage", "grounding", "circuit"]),
            "covers_emergency_procedures": any(emerg in response_lower for emerg in
                                             ["emergency", "first aid", "evacuation", "alarm"]),
            "safety_score": safety_considerations
        }

        return safety_check

class SimpleGenerator:
    """
    Implementation of: Main Generation Engine coordinating all components

    This class coordinates LLM integration, prompt building, and response validation
    to transform retrieval context into expert-quality maintenance responses with
    domain expertise, safety awareness, and quality assurance.
    """

    def __init__(self, llm_interface: LLMInterface, prompt_builder: PromptBuilder,
                 response_validator: ResponseValidator):
        """
        Implementation of: Initialize generator with all components

        Initialize LLM client connections, load response templates and prompts,
        configure generation parameters, and set up quality monitoring systems
        for comprehensive response generation.
        """
        self.llm_interface = llm_interface
        self.prompt_builder = prompt_builder
        self.response_validator = response_validator

        logger.info("SimpleGenerator initialized with all components")

    def generate(self, context: Dict[str, Any], query: str,
                query_type: Optional[str] = None, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Implementation of: Primary generation logic coordinating all steps

        This method coordinates all generation steps including prompt building,
        LLM API calls, response validation, and quality assurance, combining
        results into a comprehensive generation output with metadata and
        confidence scoring for maintenance domain responses.

        Args:
            context (Dict): Retrieval context with ranked documents
            query (str): User query
            query_type (str, optional): Type of query for template selection
            max_tokens (int): Maximum tokens for generation

        Returns:
            Dict: Generated response with quality metrics and metadata
        """
        start_time = time.time()

        try:
            # Build domain-specific prompt
            if query_type == "troubleshooting":
                system_prompt, user_prompt = self.prompt_builder.build_troubleshooting_prompt(context, query)
            elif query_type == "procedural":
                system_prompt, user_prompt = self.prompt_builder.build_procedural_prompt(context, query)
            else:
                system_prompt, user_prompt = self.prompt_builder.build_maintenance_prompt(context, query, query_type)

            # Generate response using LLM
            llm_result = self.llm_interface.generate_with_context(
                user_prompt,
                max_tokens=max_tokens,
                temperature=0.1
            )

            answer = llm_result["answer"]

            # Validate response quality
            validation_result = self.response_validator.validate_maintenance_response(
                answer, query, context
            )

            # Calculate confidence based on multiple factors
            confidence = self._calculate_response_confidence(llm_result, validation_result)

            # Extract source attribution
            source_attribution = self._extract_source_attribution(context)

            # Build comprehensive result
            result = {
                "answer": answer,
                "confidence": confidence,
                "tokens_used": llm_result["tokens_used"],
                "template_used": query_type or "default",
                "quality_score": validation_result.overall_quality,
                "safety_check": "passed" if validation_result.safety_considerations > 0.7 else "review_needed",
                "source_attribution": source_attribution,
                "llm_model": llm_result["model"],
                "detected_query_type": query_type,
                "validation_details": {
                    "technical_accuracy": validation_result.technical_accuracy,
                    "safety_considerations": validation_result.safety_considerations,
                    "completeness": validation_result.completeness,
                    "clarity": validation_result.clarity
                },
                "processing_time": time.time() - start_time
            }

            logger.info(f"Response generated successfully: confidence {confidence:.2f}, "
                       f"quality {validation_result.overall_quality:.2f}")

            return result

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error generating a response. Please try again or consult maintenance documentation.",
                "confidence": 0.0,
                "tokens_used": 0,
                "template_used": "error",
                "quality_score": 0.0,
                "safety_check": "error",
                "source_attribution": [],
                "llm_model": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def _calculate_response_confidence(self, llm_result: Dict[str, Any],
                                     validation_result: ValidationResult) -> float:
        """Calculate overall response confidence"""
        # Base confidence from LLM success
        confidence = 0.8 if llm_result["api_success"] else 0.3

        # Adjust based on validation quality
        confidence *= validation_result.overall_quality

        # Boost for high safety scores
        if validation_result.safety_considerations > 0.8:
            confidence += 0.1

        # Penalize for validation issues
        if validation_result.overall_quality < 0.5:
            confidence *= 0.5

        return min(max(confidence, 0.0), 1.0)

    def _extract_source_attribution(self, context: Dict[str, Any]) -> List[str]:
        """Extract source attribution from context"""
        if not context or "results" not in context:
            return []

        sources = []
        for result in context["results"][:3]:  # Top 3 sources
            doc_id = result.get("document_id", "unknown")
            source = result.get("source", "retrieval")
            sources.append(f"{doc_id} ({source})")

        return sources
