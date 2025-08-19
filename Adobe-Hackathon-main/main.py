import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.routes import router
from dotenv import load_dotenv
load_dotenv()


# Create FastAPI application
app = FastAPI(
    title="PDF Processing Server",
    description="A server for processing PDF files and extracting structured content",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

# Root route to serve the main page
@app.get("/")
async def serve_root():
    if os.path.exists("client/dist/index.html"):
        return FileResponse("client/dist/index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' in the client directory.")

# Add a catch-all route to handle client-side routing and static files
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Don't serve index.html for API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Try to serve the file if it exists
    file_path = f"client/dist/{full_path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Otherwise, serve index.html for client-side routing
    if os.path.exists("client/dist/index.html"):
        return FileResponse("client/dist/index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' in the client directory.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
