from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import cccd_router
from configs import STATIC_DIR, UPLOAD_DIR
from fastapi.responses import FileResponse

app = FastAPI()

origins = [
  "http://localhost",
  "http://localhost:5173",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(cccd_router.router, prefix="/api/cccd")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/{full_path:path}")
async def read_index(full_path: str):
	file_path = STATIC_DIR / full_path
	if file_path.is_file():
		return FileResponse(file_path)
	return FileResponse(STATIC_DIR / "index.html")

# app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")