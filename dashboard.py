import gradio as gr
import speechtotext

def transcribe(audiopath):
    result =  speechtotext.transcribe(audiopath)
    text = ""
    for seg in result:
        # print(seg)
        text += seg[0] + " [" + str(seg[1].start) + " - " + str(seg[1].end) + "]: " + seg[2] + "\n"
    return text
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
    title="🎙️ Whisper ASR (Bahasa Indonesia)",
    description="Record your voice and get real-time transcription using Whisper locally (no OpenAI API needed)."
)

interface.launch()