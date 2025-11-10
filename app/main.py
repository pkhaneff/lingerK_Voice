"""Start Application."""
import os

import uvicorn
from fastapi import FastAPI
from loguru import logger as custom_logger
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message

from app.core.config import ALLOWED_HOSTS, API_PREFIX, DEBUG, PROJECT_NAME, VERSION, MAX_REQUEST_SIZE

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    custom_logger.warning("Cupy not available, using CPU instead.")

if PROJECT_NAME:
    from app.api.routers.api import app as api_router

class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.endswith("/speech_to_text"):
            request.scope["upload_max_size"] = MAX_REQUEST_SIZE
        
        response = await call_next(request)
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging All API request."""

    async def set_body(request: Request, body: bytes):

        async def receive() -> Message:
            """Receive body."""
            return {"type": "http.request", "body": body}

        request._receive = receive

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Dispatch."""
        await self.set_body(request)

        # Call the next middleware or route handler
        response = await call_next(request)

        return response
    
def get_application() -> FastAPI:
    """Get application

    Returns:
        FastAPI Chatbot application
    """

    application = FastAPI(title=PROJECT_NAME, version=VERSION)
    application.add_middleware(LargeFileMiddleware)
    application.add_middleware(LoggingMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_HOSTS or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    application.include_router(api_router, prefix=API_PREFIX)

    return application

app = get_application()

if __name__ == "__main__":
    HOST = os.getenv("APP_HOST")
    PORT = os.getenv("APP_PORT")
    uvicorn.run(
        app, 
        host=HOST, 
        port=int(PORT),
        limit_max_requests=1000,
        limit_concurrency=100,
        timeout_keep_alive=300,
        timoeout_graceful_shutdown=300,
        h11_max_incomplete_event_size=100*1024*1024,
    )