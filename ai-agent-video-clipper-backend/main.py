import json
import os
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid
import boto3
import modal
import whisperx
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel
from google import genai


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

        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda"
        )
        print("Models loaded successfully")

        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Created gemini client...")

    def identify_moments(self, transcript: dict):
        response = self.gemini_client.models.generate_content(model="gemini-2.5-flash-preview-04-17", contents="""
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + str(transcript))
        print(f"Identified moments response: ${response.text}")
        return response.text

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(
            audio, batch_size=16)

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration_time = time.time() - start_time
        print("Transcription and alignment took " +
              str(duration_time) + " seconds")

        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"],
                })

        return json.dumps(segments)

    def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
        clip_name = f"clip_{clip_index}"
        s3_key_dir = os.path.dirname(s3_key)
        output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
        print(f"Output S3 key: {output_s3_key}")

        clip_dir = base_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
        vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
        subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

        (clip_dir / "pywork").mkdir(exist_ok=True)
        pyframes_path = clip_dir / "pyframes"
        pyavi_path = clip_dir / "pyavi"
        audio_path = clip_dir / "pyavi" / "audio.wav"

        pyframes_path.mkdir(exist_ok=True)
        pyavi_path.mkdir(exist_ok=True)

        duration = end_time - start_time
        cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                       f"{clip_segment_path}")
        subprocess.run(cut_command, shell=True, check=True,
                       capture_output=True, text=True)

        extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

        columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                            f"--videoFolder {str(base_dir)} "
                            f"--pretrainModel weight/finetuning_TalkSet.model")

        columbia_start_time = time.time()
        subprocess.run(columbia_command, cwd="/asd", shell=True)
        columbia_end_time = time.time()
        print(
            f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

        tracks_path = clip_dir / "pywork" / "tracks.pckl"
        scores_path = clip_dir / "pywork" / "scores.pckl"
        if not tracks_path.exists() or not scores_path.exists():
            raise FileNotFoundError("Tracks or scores not found for clip")

        with open(tracks_path, "rb") as f:
            tracks = pickle.load(f)

        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

        cvv_start_time = time.time()
        create_vertical_video(
            tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
        )
        cvv_end_time = time.time()
        print(
            f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

        create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                     end_time, vertical_mp4_path, subtitle_output_path, max_words=5)

        s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=os.environ["CLOUDFLARE_ACCESS_KEY"],
            aws_secret_access_key=os.environ["CLOUDFLARE_SECRET_KEY"],
            endpoint_url=os.environ["R2_END_POINT"],
        )
        s3_client.upload_file(
            subtitle_output_path, "ai-podcast-clipper", output_s3_key)

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

        transcripts_segments_json = self.transcribe_video(base_dir, video_path)
        transcripts_segments = json.loads(transcripts_segments_json)

        print(f"Transcripts segments: {transcripts_segments}")
        print("Identifying moments in the transcript...")
        identified_moments_json = self.identify_moments(transcripts_segments)

        cleaned_json_string = identified_moments_json.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()

        clip_moments = json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error: Identified moments is not a list")
            clip_moments = []

        print(clip_moments)

        for index, moment in enumerate(clip_moments[:5]):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + " from " +
                      str(moment["start"]) + " to " + str(moment["end"]))
                process_clip(base_dir, video_path, s3_key,
                             moment["start"], moment["end"], index, transcript_segments)

        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


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
