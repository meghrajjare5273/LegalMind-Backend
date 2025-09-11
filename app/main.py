from fastapi import FastAPI
from api.v1 import health, contract_analysis, rag_chat

app = FastAPI(title="LegalMind API", version="5.0.0")

# Register routers
# app.include_router(health.router, prefix="/api/v1")
# app.include_router(contract_analysis.router, prefix="/api/v1")
app.include_router(rag_chat.router, prefix="/api/v1")
