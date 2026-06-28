import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi.middleware.cors import CORSMiddleware

from server.apps import FastApiServer
from utils.logger import get_logger

logger = get_logger(__name__)

origins = [
    "http://localhost:5501",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    ]

def main():
    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    app.add_middleware(
        middleware_class  = CORSMiddleware,
        allow_origins     = origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception | %s %s | %s: %s",
            request.method, request.url.path,
            type(exc).__name__, exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {str(exc)}"},
        )

    server = FastApiServer()
    app.include_router(server.router)

    logger.info("FastAPI server initialized")
    return app

app = main()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
