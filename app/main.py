from fastapi import FastAPI
from routers import cccd_router

app = FastAPI()

app.include_router(cccd_router.router, prefix="/cccd")