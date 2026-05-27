from pathlib import Path
from uuid import uuid4
from typing import List
import traceback

import aiofiles
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.services.model_service import get_image_info, predict_image
from app.services.groq_service import ask_ai
from app.services.batch_service import process_batch_uploads


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATE_DIR = APP_DIR / "templates"
OUTPUT_DIR = STATIC_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_BATCH_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".zip", ".rar"}

MAX_UPLOAD_MB = 15
MAX_BATCH_FILE_MB = 300
CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="Meat Freshness Classifier")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATE_DIR)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/download_report/{filename}")
async def download_report(filename: str):
    safe_filename = Path(filename).name
    report_path = OUTPUT_DIR / safe_filename

    if not report_path.exists() or report_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo PDF.")

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename="bao_cao_du_doan_hang_loat.pdf"
    )


def clean_old_files(pattern, limit=50):
    files = sorted(OUTPUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    for p in files[limit:]:
        p.unlink(missing_ok=True)


async def save_upload(file: UploadFile, file_path: Path, max_mb: int):
    total = 0

    async with aiofiles.open(file_path, "wb") as f:
        while True:
            chunk = await file.read(CHUNK_SIZE)

            if not chunk:
                break

            total += len(chunk)

            if total > max_mb * 1024 * 1024:
                file_path.unlink(missing_ok=True)
                raise ValueError(f"File quá lớn. Giới hạn tối đa {max_mb}MB.")

            await f.write(chunk)

    return total


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = None

    try:
        original_name = file.filename or "uploaded_image"
        suffix = Path(original_name).suffix.lower()

        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            return JSONResponse(
                {"ok": False, "error": "Định dạng ảnh không hợp lệ. Chỉ hỗ trợ JPG, JPEG, PNG, BMP, WEBP."},
                status_code=400
            )

        filename = f"upload_{uuid4().hex}{suffix}"
        file_path = OUTPUT_DIR / filename

        await save_upload(file, file_path, MAX_UPLOAD_MB)

        info = await run_in_threadpool(get_image_info, file_path)
        info["name"] = original_name

        result = await run_in_threadpool(predict_image, file_path)
        image_url = f"/static/outputs/{filename}"

        clean_old_files("upload_*", limit=50)

        return JSONResponse({
            "ok": True,
            "image_url": image_url,
            "info": info,
            "result": result
        })

    except Exception as e:
        if file_path is not None:
            file_path.unlink(missing_ok=True)

        print(traceback.format_exc())

        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500
        )


@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    saved_files = []

    try:
        if not files:
            return JSONResponse(
                {"ok": False, "error": "Chưa có file nào được upload."},
                status_code=400
            )

        batch_id = uuid4().hex
        batch_input_dir = OUTPUT_DIR / f"batch_input_{batch_id}"
        batch_input_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            original_name = file.filename or f"upload_{uuid4().hex}"
            safe_original = Path(original_name).name
            suffix = Path(safe_original).suffix.lower()

            if suffix not in ALLOWED_BATCH_SUFFIXES:
                continue

            filename = f"{uuid4().hex}_{safe_original}"
            file_path = batch_input_dir / filename

            await save_upload(file, file_path, MAX_BATCH_FILE_MB)
            saved_files.append(file_path)

        if not saved_files:
            return JSONResponse(
                {"ok": False, "error": "Không tìm thấy ảnh, file .zip hoặc .rar hợp lệ."},
                status_code=400
            )

        result = await run_in_threadpool(process_batch_uploads, saved_files)

        clean_old_files("batch_input_*", limit=20)

        return JSONResponse({
            "ok": True,
            "result": result
        })

    except Exception as e:
        print(traceback.format_exc())

        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500
        )


@app.post("/chat")
async def chat(question: str = Form(...), context: str = Form("")):
    try:
        answer = await run_in_threadpool(ask_ai, question, context)

        return JSONResponse({
            "ok": True,
            "answer": answer
        })

    except Exception as e:
        print(traceback.format_exc())

        return JSONResponse(
            {"ok": False, "answer": str(e)},
            status_code=500
        )