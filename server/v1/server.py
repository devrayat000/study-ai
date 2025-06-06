from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

# Local imports
from .generate_ai_response import gen_ai_response
from .models import AIRequest
from .db import ChromaDep
from .config import config
from .middleware import PerformanceMiddleware, RequestLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Study AI API",
    description="AI-powered academic assistant for physics education",
    version="1.0.0",
    debug=config.debug,
)

# Add middleware
app.add_middleware(PerformanceMiddleware)
app.add_middleware(RequestLoggingMiddleware, debug=config.debug)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint to simulate AI chat
@app.post("/chat")
async def ai_response(
    body: AIRequest,
    background_tasks: BackgroundTasks,
    pc: ChromaDep,
    request: Request,
):
    def generate_response():
        for chunk in gen_ai_response(
            prompt=body.user_prompt,
            session_id=body.conversation_id,
            vector_db=pc,
        ):
            yield chunk

    # Background task to cache the result once streaming is complete
    # background_tasks.add_task(increment_rate_limit, user_id)

    # Return streaming response
    return StreamingResponse(generate_response())
