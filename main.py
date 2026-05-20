import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class VideoRequest(BaseModel):
    prompt: str
    model: str = "veo-2.0-generate-001"
    duration: int = 8
    aspect: str = "16:9"
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None


@app.get("/")
def root():
    return {"status": "Gemini Video API running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/generate-video")
async def generate_video(req: VideoRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY가 설정되지 않았습니다.")

    # Veo API 전용 엔드포인트: predictLongRunning
    url = f"{BASE_URL}/models/{req.model}:predictLongRunning?key={GEMINI_API_KEY}"

    # instances 구성
    instance = {"prompt": req.prompt}
    if req.image_base64 and req.image_mime:
        instance["image"] = {
            "bytesBase64Encoded": req.image_base64,
            "mimeType": req.image_mime
        }

    # parameters 구성
    parameters = {
        "aspectRatio": req.aspect,
        "durationSeconds": req.duration,
        "sampleCount": 1
    }

    payload = {
        "instances": [instance],
        "parameters": parameters
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", f"HTTP {resp.status_code}")
        except Exception:
            msg = f"HTTP {resp.status_code}"
        raise HTTPException(status_code=resp.status_code, detail=msg)

    data = resp.json()

    # operation name 반환됨 → 폴링
    op_name = data.get("name")
    if not op_name:
        raise HTTPException(status_code=500, detail="operation name을 받지 못했습니다.")

    result = await poll_operation(op_name)
    return result


async def poll_operation(op_name: str, max_wait: int = 300, interval: int = 5):
    url = f"{BASE_URL}/{op_name}?key={GEMINI_API_KEY}"
    elapsed = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue

                data = resp.json()

                if not data.get("done", False):
                    continue

                if "error" in data:
                    raise HTTPException(
                        status_code=500,
                        detail=data["error"].get("message", "오류 발생")
                    )

                # 응답에서 영상 추출
                videos = (
                    data.get("response", {})
                    .get("generatedSamples", [])
                )
                if videos:
                    video = videos[0].get("video", {})
                    # URI 방식
                    if "uri" in video:
                        return {"type": "file", "uri": video["uri"]}
                    # base64 방식
                    if "bytesBase64Encoded" in video:
                        return {
                            "type": "inline",
                            "mimeType": "video/mp4",
                            "data": video["bytesBase64Encoded"]
                        }

                raise HTTPException(status_code=500, detail="영상 데이터를 찾을 수 없습니다.")

            except HTTPException:
                raise
            except Exception:
                continue

    raise HTTPException(status_code=504, detail="영상 생성 타임아웃 (5분 초과)")
