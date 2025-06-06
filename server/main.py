from dotenv import load_dotenv

load_dotenv(override=True)

import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

# Local imports
import v1


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)

app.mount(
    "/v1", v1.app, name="Version 1"
)  # Mount the v1 app to the main app for versioning

if __name__ == "__main__":
    import uvicorn, os

    PORT = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
