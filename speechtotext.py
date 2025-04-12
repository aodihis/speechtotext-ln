import os

from dotenv import load_dotenv
from pyannote.audio import Pipeline
import whisper
from pyannote.core import Segment

load_dotenv()
token = os.getenv("HF_TOKEN")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)

model = whisper.load_model("small")


def transcribe(audiopath):
    diarization = pipeline(audiopath)
    transcribe_result = model.transcribe(audiopath,  language="indonesian")
    conv = []
    for item in transcribe_result["segments"]:
        start = item["start"]
        end = item["end"]
        text = item["text"]
        seg = Segment(start, end)
        speaker = diarization.crop(seg).argmax()
        conv.append((speaker, seg, text))
    return conv


