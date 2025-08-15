import os
import pathlib
import uuid
import boto3
import modal
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("asd", "/asd", copy=True))


app = modal.App("ai-video-clipper", image=image)
volume = modal.Volume.from_name(
    "ai-video-clipper-volume", create_if_missing=True)
mount_path = "/root/.cache/torch"


auth_scheme = HTTPBearer()


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-clipper-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_modal(self):
        print("Loading models")
        pass

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        print(f"Processing Video {request.s3_key}")

        if not request.s3_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="s3_key is required")

        if token.credentials != os.environ.get("AUTH_TOKEN"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        video_path = base_dir / "input.mp4"

        client = boto3.client(
            service_name='s3',
            aws_access_key_id=os.environ["CLOUDFLARE_ACCESS_KEY"],
            aws_secret_access_key=os.environ["CLOUDFLARE_SECRET_KEY"],
            endpoint_url=os.environ["R2_END_POINT"],
        )

        client.download_file(
            "videoclipper", request.s3_key, str(video_path))

        print(os.listdir(base_dir))
        return {"status": "success", "video_path": str(video_path)}


@app.local_entrypoint()
def main():
    import requests
    ai_podcast_clipper = AiPodcastClipper()
    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "temp/file_example_MP4_480_1_5MG.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123",
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)
