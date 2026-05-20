import os
import asyncio
import httpx
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    url = f"{BASE_URL}/models/{req.model}:predictLongRunning?key={GEMINI_API_KEY}"

    instance = {"prompt": req.prompt}
    if req.image_base64 and req.image_mime:
        instance["image"] = {
            "bytesBase64Encoded": req.image_base64,
            "mimeType": req.image_mime
        }

    parameters = {
        "aspectRatio": req.aspect,
        "durationSeconds": req.duration,
        "sampleCount": 1
    }

    payload = {
        "instances": [instance],
        "parameters": parameters
    }

    logger.info(f"요청 URL: {url}")
    logger.info(f"Payload (이미지 제외): prompt={req.prompt}, model={req.model}, duration={req.duration}, aspect={req.aspect}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)

    logger.info(f"초기 응답 status: {resp.status_code}")
    logger.info(f"초기 응답 body: {resp.text[:500]}")

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", f"HTTP {resp.status_code}")
        except Exception:
            msg = f"HTTP {resp.status_code}"
        raise HTTPException(status_code=resp.status_code, detail=msg)

    data = resp.json()
    op_name = data.get("name")
    if not op_name:
        logger.error(f"operation name 없음. 전체 응답: {data}")
        raise HTTPException(status_code=500, detail=f"operation name을 받지 못했습니다. 응답: {str(data)[:300]}")

    logger.info(f"Operation name: {op_name}")
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
                logger.info(f"폴링 [{elapsed}s] status: {resp.status_code}")

                if resp.status_code != 200:
                    continue

                data = resp.json()
                done = data.get("done", False)
                logger.info(f"폴링 [{elapsed}s] done: {done}")

                if not done:
                    continue

                logger.info(f"완료 응답 전체: {str(data)[:1000]}")

                if "error" in data:
                    raise HTTPException(
                        status_code=500,
                        detail=data["error"].get("message", "오류 발생")
                    )

                # 응답 구조 탐색
                response = data.get("response", {})
                logger.info(f"response 키들: {list(response.keys())}")

                # generatedSamples 방식
                samples = response.get("generatedSamples", [])
                if samples:
                    video = samples[0].get("video", {})
                    logger.info(f"video 키들: {list(video.keys())}")
                    if "uri" in video:
                        return {"type": "file", "uri": video["uri"]}
                    if "bytesBase64Encoded" in video:
                        return {"type": "inline", "mimeType": "video/mp4", "data": video["bytesBase64Encoded"]}

                # generatedVideos 방식 (SDK 스타일)
                gen_videos = response.get("generatedVideos", [])
                if gen_videos:
                    video = gen_videos[0].get("video", {})
                    logger.info(f"generatedVideos video 키들: {list(video.keys())}")
                    if "uri" in video:
                        return {"type": "file", "uri": video["uri"]}
                    if "bytesBase64Encoded" in video:
                        return {"type": "inline", "mimeType": "video/mp4", "data": video["bytesBase64Encoded"]}

                logger.error(f"영상 데이터 없음. response 전체: {str(response)[:500]}")
                raise HTTPException(status_code=500, detail=f"영상 데이터를 찾을 수 없습니다. 응답구조: {str(response)[:300]}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"폴링 예외: {e}")
                continue

    raise HTTPException(status_code=504, detail="영상 생성 타임아웃 (5분 초과)")
