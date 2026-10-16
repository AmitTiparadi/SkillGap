"""Custom local UI: python -m skillgap.app. No Node build is needed."""
from dataclasses import replace
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from skillgap.config import load_settings
from skillgap.graph.workflow import analyze
from skillgap.parsers import split_jobs
ROOT = Path(__file__).resolve().parent
app = FastAPI(title="SkillGap", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

@app.middleware("http")
async def local_browser_only(request: Request, call_next):
    # Block cross-origin submissions to the private local document service.
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and origin != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Use the local SkillGap page."}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response

@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")

@app.get("/api/config")
def configuration():
    settings = load_settings()
    # Always start locally. Never expose stored credentials to JavaScript.
    return {"model": settings.ollama_model}

@app.get("/api/sample")
def sample():
    return {"resume": (ROOT / "examples" / "resume.txt").read_text(encoding="utf-8"),
            "jobs": (ROOT / "examples" / "jobs.txt").read_text(encoding="utf-8")}

@app.post("/api/analyze")
async def analysis(resume: UploadFile = File(...), jobs: str = Form(...),
                   provider: str = Form("ollama"), model: str = Form(""),
                   api_model: str = Form(""), api_key: str = Form(""),
                   api_consent: bool = Form(False)):
    try:
        defaults = load_settings()
        if provider == "api" and not api_consent:
            raise ValueError("Confirm external processing before using an API model.")
        settings = replace(defaults, llm_provider=provider,
                           ollama_model=model.strip() or defaults.ollama_model,
                           api_model=api_model.strip() or defaults.api_model,
                           api_key=api_key or defaults.api_key)
        # Limit reads before the workflow allocates document or model memory.
        contents = await resume.read(settings.max_resume_bytes + 1)
        if len(contents) > settings.max_resume_bytes:
            raise ValueError("Resume exceeds the 5 MB upload limit.")
        if not contents or not split_jobs(jobs):
            raise ValueError("Add a resume and at least one job description.")
        # Blocking inference runs in a worker thread, keeping the UI responsive.
        # The workflow's file lock prevents concurrent model memory peaks.
        return await run_in_threadpool(analyze, contents, resume.filename or "resume.txt",
                                       split_jobs(jobs), settings)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(500, "Analysis could not finish. Check Ollama and database access, then retry.") from error
    finally:
        await resume.close()

def main():
    import uvicorn
    uvicorn.run("skillgap.app:app", host="127.0.0.1", port=8501, access_log=False)

if __name__ == "__main__":
    main()
