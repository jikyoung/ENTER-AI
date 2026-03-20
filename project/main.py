import os
import uvicorn
from fastapi import FastAPI

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi.middleware.cors import CORSMiddleware

from server.apps import FastApiServer

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

    server = FastApiServer()

    app.include_router(server.router)
    
    return app

app = main()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
