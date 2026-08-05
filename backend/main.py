from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from routers.detect import router as detect_router
from db import init_db

app = FastAPI(title="FakeGuard AI API", version="1.0.0")
init_db()

# Vite picks the next free port when its default is taken (5173 -> 5174 -> ...),
# so a hardcoded origin list silently breaks the moment a stale dev server is
# still holding 5173. The failure is confusing rather than obvious: the request
# reaches the API and is served 200, then the browser discards the response for
# lacking the CORS header, and the client reports it as a generic "Network
# Error" with nothing useful in the server log.
#
# This regex covers any port on loopback only. Fine for a locally-run tool;
# if this is ever deployed somewhere reachable, replace it with an explicit
# allowlist of the real front-end origin(s) rather than widening it further.
LOCAL_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(detect_router)


@app.get("/health")
def health():
    return {"status": "ok"}
