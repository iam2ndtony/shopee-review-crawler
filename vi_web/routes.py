"""HTTP surface of the web demo: saved products, live jobs, and the static page.

Every POST that starts work needs a JSON body, so a page on another site cannot trigger it
through the browser (a cross-origin JSON POST needs a preflight this server does not grant).
Local single-user use only.
"""
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analyze import analyze_saved, list_saved
from .config import WebConfig
from .jobs import Busy, JobStore

STATIC = Path(__file__).parent / "static"


class LinkIn(BaseModel):
    url: str = Field(max_length=2000)
    max_reviews: int | None = Field(default=None, ge=1)  # omitted: crawl every review


class SavedIn(BaseModel):
    file: str = Field(max_length=200)


def build_router(cfg: WebConfig | None = None, store: JobStore | None = None) -> APIRouter:
    cfg = cfg or WebConfig()
    store = store or JobStore(cfg)
    r = APIRouter(prefix="/analyze")

    def job_or_404(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown analysis")
        return job

    @r.get("/saved")
    def saved() -> list[dict]:
        return list_saved(cfg)

    @r.post("/saved")
    def saved_analysis(body: SavedIn) -> dict:
        try:
            return analyze_saved(body.file, cfg)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown saved product")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Held-out predictions for the served model are missing")

    @r.post("/link")
    def link(body: LinkIn) -> dict:
        try:
            job = store.start(body.url, body.max_reviews)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Busy:
            raise HTTPException(status_code=409, detail="Another analysis is already running")
        return {"job_id": job.id}

    @r.get("/active")
    def active() -> dict:
        job = store.active()
        return {"job_id": job.id if job else None, "state": job.state if job else None}

    @r.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = job_or_404(job_id)
        return {"state": job.state, "collected": job.collected, "error": job.error}

    @r.post("/jobs/{job_id}/ready")
    def job_ready(job_id: str) -> dict:
        job = job_or_404(job_id)
        job.ready.set()
        return {"state": job.state}

    @r.get("/jobs/{job_id}/result")
    def job_result(job_id: str) -> dict:
        job = job_or_404(job_id)
        if job.state != "done":
            raise HTTPException(status_code=409, detail=f"Analysis is {job.state}")
        return job.result

    return r


def mount(app: FastAPI, cfg: WebConfig | None = None, store: JobStore | None = None) -> None:
    """Add the routes, the static files at /static and the page at /."""
    app.include_router(build_router(cfg, store))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")
