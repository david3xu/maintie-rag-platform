# MaintIE-Enhanced RAG Platform

A simplified microservice architecture for maintenance document retrieval and question answering.

## Quick Start

1. Copy environment file:

   ```bash
   cp env.example .env
   # Edit .env with your OPENAI_API_KEY
   ```

2. Start all services:

   ```bash
   docker-compose up --build
   ```

3. Test the platform:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "pump seal failure"}'
   ```

## Architecture

- **Query Enhancement Service** (Team Alpha): Enhances user queries
- **Knowledge Graph Service** (Team Beta): Expands concepts using graph
- **Retrieval Fusion Service** (Team Gamma): Multi-modal search and fusion
- **Response Generation Service** (Team Delta): LLM-based answer generation
- **API Gateway**: Orchestrates all services

## Development

Each service can be developed independently. See individual service directories for implementation details.
