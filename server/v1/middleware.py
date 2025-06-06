"""Performance monitoring and logging middleware."""

import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor request performance and add logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log incoming request
        logger.info(f"Incoming {request.method} request to {request.url.path}")

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)

            # Log response details
            logger.info(
                f"Request processed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.3f}s"
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.3f}s"
            )
            raise


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging in debug mode."""

    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.debug:
            return await call_next(request)

        # Log request details in debug mode
        logger.debug(f"Request headers: {dict(request.headers)}")

        if request.method in ["POST", "PUT", "PATCH"]:
            # Note: In production, be careful about logging sensitive data
            body = await request.body()
            if body:
                logger.debug(f"Request body size: {len(body)} bytes")

        response = await call_next(request)

        # Log response details
        logger.debug(f"Response headers: {dict(response.headers)}")
        logger.debug(f"Response status: {response.status_code}")

        return response
