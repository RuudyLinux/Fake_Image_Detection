import io
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

from models import blind_detector, comparative_detector
from reports.pdf_report import generate_blind_report, generate_comparative_report
import db

router = APIRouter(prefix="/detect", tags=["detection"])

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE = 20 * 1024 * 1024  # 20 MB
THUMB_SIZE = (320, 320)


def _validate_image(upload: UploadFile) -> None:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported image type: {upload.content_type}")


def _save_thumbnail(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    filename = f"thumb_{uuid.uuid4().hex[:8]}.jpg"
    img.save(STATIC_DIR / filename, quality=80)
    return f"/static/{filename}"


@router.post("/blind")
async def detect_blind(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    enable_gradcam: bool = Form(True),
):
    _validate_image(image)
    data = await image.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        result = blind_detector.analyze(data, confidence_threshold, enable_gradcam)
        thumbnail_url = _save_thumbnail(data)
        db.insert_detection("blind", result, thumbnail_url)
    except Exception as exc:
        raise HTTPException(500, f"Analysis error: {exc}") from exc
    return result


@router.post("/blind/report")
async def detect_blind_report(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    enable_gradcam: bool = Form(True),
):
    _validate_image(image)
    data = await image.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        result = blind_detector.analyze(data, confidence_threshold, enable_gradcam)
        pdf_bytes = generate_blind_report(data, result)
    except Exception as exc:
        raise HTTPException(500, f"Report generation error: {exc}") from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fakeguard_report.pdf"},
    )


@router.post("/compare")
async def detect_comparative(
    original: UploadFile = File(...),
    suspected: UploadFile = File(...),
):
    _validate_image(original)
    _validate_image(suspected)
    orig_data = await original.read()
    susp_data = await suspected.read()
    if len(orig_data) > MAX_SIZE or len(susp_data) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        result = comparative_detector.compare(orig_data, susp_data)
        thumbnail_url = _save_thumbnail(susp_data)
        history_result = {
            "prediction": "MANIPULATED" if result["is_manipulated"] else "AUTHENTIC",
            "confidence": result["similarity_score"],
            "forgery_type": None,
            "heatmap_url": result.get("highlighted_url") or result.get("diff_map_url"),
        }
        db.insert_detection("compare", history_result, thumbnail_url)
    except Exception as exc:
        raise HTTPException(500, f"Comparison error: {exc}") from exc
    return result


@router.post("/compare/report")
async def detect_comparative_report(
    original: UploadFile = File(...),
    suspected: UploadFile = File(...),
):
    _validate_image(original)
    _validate_image(suspected)
    orig_data = await original.read()
    susp_data = await suspected.read()
    if len(orig_data) > MAX_SIZE or len(susp_data) > MAX_SIZE:
        raise HTTPException(413, "Image exceeds 20 MB limit")
    try:
        result = comparative_detector.compare(orig_data, susp_data)
        pdf_bytes = generate_comparative_report(orig_data, susp_data, result)
    except Exception as exc:
        raise HTTPException(500, f"Report generation error: {exc}") from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fakeguard_comparative_report.pdf"},
    )


@router.get("/history")
async def get_history(limit: int = Query(50, ge=1, le=200)):
    return db.list_detections(limit)


@router.delete("/history")
async def delete_history():
    db.clear_detections()
    return {"status": "cleared"}
