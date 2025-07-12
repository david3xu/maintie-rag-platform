# 🚀 MaintIE-Enhanced RAG: Very Simple Platform Structure
## Simplified Version of Full Microservice Architecture

**Objective**: Keep the proper platform architecture but make it extremely simple  
**Focus**: Essential components only, minimal complexity, maximum working functionality  
**Timeline**: 2-3 days to working platform

---

## 📂 **Very Simple Platform Directory Structure**

```
maintie-rag-platform/
├── 📁 shared/                                  # Minimal shared components
│   ├── models.py                               # Common data models
│   ├── config.py                               # Shared configuration
│   └── utils.py                                # Common utilities
├── 📁 services/                                # Core microservices (simplified)
│   ├── 📁 query-enhancement-service/           # 🧠 Team Alpha
│   │   ├── main.py                             # FastAPI app
│   │   ├── core.py                             # Query enhancement logic
│   │   ├── requirements.txt                    # Dependencies
│   │   └── Dockerfile                          # Container
│   ├── 📁 knowledge-graph-service/             # 🔗 Team Beta  
│   │   ├── main.py                             # FastAPI app
│   │   ├── core.py                             # Graph operations
│   │   ├── data/                               # MaintIE datasets
│   │   │   ├── raw/                            # Original data
│   │   │   └── processed/                      # Processed graph
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── 📁 retrieval-fusion-service/            # 🎯 Team Gamma
│   │   ├── main.py                             # FastAPI app  
│   │   ├── core.py                             # Multi-modal search & fusion
│   │   ├── data/                               # Embeddings & indices
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── 📁 response-generation-service/         # 📝 Team Delta
│       ├── main.py                             # FastAPI app
│       ├── core.py                             # LLM integration
│       ├── requirements.txt
│       └── Dockerfile
├── 📁 gateway/                                 # Simple API Gateway
│   ├── main.py                                 # Gateway FastAPI app
│   ├── requirements.txt
│   └── Dockerfile
├── 📁 scripts/                                 # Essential scripts
│   ├── setup_data.py                           # Setup MaintIE data
│   ├── start_all.sh                            # Start all services
│   └── test_system.py                          # Basic system test
├── docker-compose.yml                          # All services together
├── .env.example                                # Environment variables
└── README.md                                   # Setup instructions
```

---

## 🚀 **Very Simple Service Implementation**

### **Query Enhancement Service (Team Alpha)**
```python
# services/query-enhancement-service/main.py
from fastapi import FastAPI
from .core import QueryEnhancer

app = FastAPI(title="Query Enhancement", version="1.0.0")
enhancer = QueryEnhancer()

@app.post("/enhance")
async def enhance_query(query: str):
    return enhancer.enhance(query)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# services/query-enhancement-service/core.py
class QueryEnhancer:
    def enhance(self, query: str):
        # Simple implementation
        query_type = self.classify_query(query)
        entities = self.extract_entities(query)
        return {
            "original": query,
            "type": query_type,
            "entities": entities,
            "enhanced": True
        }
    
    def classify_query(self, query: str):
        # Simple rule-based classification
        if any(word in query.lower() for word in ["failure", "broken", "problem"]):
            return "troubleshooting"
        elif any(word in query.lower() for word in ["how", "steps", "procedure"]):
            return "procedural"
        else:
            return "informational"
    
    def extract_entities(self, query: str):
        # Simple entity extraction
        # Return basic entities found in query
        return ["pump", "seal"] # simplified
```

### **Knowledge Graph Service (Team Beta)**
```python
# services/knowledge-graph-service/main.py
from fastapi import FastAPI
from .core import SimpleKnowledgeGraph

app = FastAPI(title="Knowledge Graph", version="1.0.0")
kg = SimpleKnowledgeGraph()

@app.post("/expand")
async def expand_concepts(concepts: list):
    return kg.expand_concepts(concepts)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# services/knowledge-graph-service/core.py
import json
import networkx as nx

class SimpleKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.load_graph()
    
    def load_graph(self):
        # Load from processed MaintIE data
        with open("data/processed/graph.json", "r") as f:
            graph_data = json.load(f)
        # Build NetworkX graph
        
    def expand_concepts(self, concepts: list):
        expanded = []
        for concept in concepts:
            if concept in self.graph:
                neighbors = list(self.graph.neighbors(concept))
                expanded.extend(neighbors[:3])  # Top 3 related
        return list(set(expanded))
```

### **Retrieval Fusion Service (Team Gamma)**
```python
# services/retrieval-fusion-service/main.py
from fastapi import FastAPI
from .core import SimpleFusion

app = FastAPI(title="Retrieval Fusion", version="1.0.0")
fusion = SimpleFusion()

@app.post("/search")
async def search_and_fuse(enhanced_query: dict):
    return fusion.search_and_fuse(enhanced_query)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# services/retrieval-fusion-service/core.py
class SimpleFusion:
    def __init__(self):
        self.setup_searchers()
    
    def setup_searchers(self):
        # Load embeddings and indices
        pass
    
    def search_and_fuse(self, enhanced_query: dict):
        # Vector search
        vector_results = self.vector_search(enhanced_query["original"])
        
        # Entity search  
        entity_results = self.entity_search(enhanced_query["entities"])
        
        # Simple fusion - combine and rank
        fused_results = self.simple_fusion(vector_results, entity_results)
        
        return {"results": fused_results[:5]}  # Top 5
    
    def vector_search(self, query: str):
        # Simple vector search implementation
        return ["doc1", "doc2", "doc3"]  # simplified
    
    def entity_search(self, entities: list):
        # Simple entity-based search
        return ["doc2", "doc4", "doc5"]  # simplified
    
    def simple_fusion(self, vector_results: list, entity_results: list):
        # Simple weighted combination
        combined = vector_results + entity_results
        return list(set(combined))  # Remove duplicates
```

### **Response Generation Service (Team Delta)**
```python
# services/response-generation-service/main.py
from fastapi import FastAPI
from .core import SimpleGenerator

app = FastAPI(title="Response Generation", version="1.0.0")
generator = SimpleGenerator()

@app.post("/generate")
async def generate_response(context: dict, query: str):
    return generator.generate(context, query)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# services/response-generation-service/core.py
import openai

class SimpleGenerator:
    def __init__(self):
        self.client = openai.OpenAI()
    
    def generate(self, context: dict, query: str):
        # Simple prompt building
        prompt = f"""
        Based on the following maintenance documents:
        {context.get('results', [])}
        
        Answer this maintenance question: {query}
        
        Provide a clear, helpful answer.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return {
            "answer": response.choices[0].message.content,
            "query": query
        }
```

### **Simple API Gateway**
```python
# gateway/main.py
from fastapi import FastAPI
import httpx

app = FastAPI(title="MaintIE Gateway", version="1.0.0")

SERVICE_URLS = {
    "query": "http://query-enhancement-service:8000",
    "kg": "http://knowledge-graph-service:8000", 
    "retrieval": "http://retrieval-fusion-service:8000",
    "generation": "http://response-generation-service:8000"
}

@app.post("/query")
async def process_query(query: str):
    async with httpx.AsyncClient() as client:
        # 1. Enhance query
        enhanced = await client.post(f"{SERVICE_URLS['query']}/enhance", 
                                   json={"query": query})
        enhanced_data = enhanced.json()
        
        # 2. Expand concepts
        expanded = await client.post(f"{SERVICE_URLS['kg']}/expand",
                                   json=enhanced_data["entities"])
        expanded_data = expanded.json()
        
        # 3. Search and fuse
        search_results = await client.post(f"{SERVICE_URLS['retrieval']}/search",
                                         json=enhanced_data)
        search_data = search_results.json()
        
        # 4. Generate response
        response = await client.post(f"{SERVICE_URLS['generation']}/generate",
                                   json={"context": search_data, "query": query})
        
        return response.json()

@app.get("/health")
async def health():
    return {"status": "healthy", "platform": "maintie-rag"}
```

---

## 🐳 **Simple Docker Setup**

### **docker-compose.yml**
```yaml
version: '3.8'

services:
  gateway:
    build: ./gateway
    ports: ["8000:8000"]
    depends_on: [query-enhancement, knowledge-graph, retrieval-fusion, response-generation]

  query-enhancement:
    build: ./services/query-enhancement-service
    expose: [8000]

  knowledge-graph:
    build: ./services/knowledge-graph-service
    expose: [8000]
    volumes: ["./shared:/app/shared"]

  retrieval-fusion:
    build: ./services/retrieval-fusion-service
    expose: [8000]

  response-generation:
    build: ./services/response-generation-service
    expose: [8000]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

### **Simple Dockerfile (same for all services)**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## ⚡ **Ultra-Quick Start**

### **Setup (5 minutes)**
```bash
# Clone repo
git clone <repo> && cd maintie-rag-platform

# Setup environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Setup data (if needed)
python scripts/setup_data.py
```

### **Start Platform (1 command)**
```bash
# Start all services
docker-compose up --build

# Test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "pump seal failure"}'
```

---

## 📊 **Simple Requirements Per Service**

### **Basic requirements.txt (all services)**
```
fastapi==0.104.1
uvicorn==0.24.0
httpx==0.25.2
python-dotenv==1.0.0

# Service-specific:
# query-enhancement: +spacy
# knowledge-graph: +networkx  
# retrieval-fusion: +sentence-transformers
# response-generation: +openai
```

---

## ✅ **Simple Success Criteria**

| **Service** | **Success** | **Test** |
|-------------|-------------|----------|
| **Query Enhancement** | Returns enhanced query structure | `POST /enhance` |
| **Knowledge Graph** | Expands concepts from entities | `POST /expand` |
| **Retrieval Fusion** | Returns fused search results | `POST /search` |
| **Response Generation** | Generates maintenance answer | `POST /generate` |
| **Gateway** | End-to-end query processing | `POST /query` |

### **Quick Test**
```bash
# Test entire platform
python scripts/test_system.py

# Expected: All services respond, end-to-end query works
```

---

## 🎯 **Why This Simple Structure Works**

**✅ Maintains Architecture:**
- Proper service separation (4 core services + gateway)
- Each team owns their service independently
- Clear API contracts between services

**✅ Ultra Simple Implementation:**
- Single Python file per service core logic
- Basic FastAPI apps with minimal endpoints
- Simple Docker Compose for deployment

**✅ Working in 2-3 Days:**
- Day 1: Teams implement basic service logic
- Day 2: Docker integration and gateway setup
- Day 3: End-to-end testing and deployment

**✅ Extension Ready:**
- Clear structure to add complexity later
- Each service can be enhanced independently
- Direct path to full microservice platform

This gives you the **proper microservice platform architecture** in the simplest possible form while maintaining the ability to parallel develop and extend later!