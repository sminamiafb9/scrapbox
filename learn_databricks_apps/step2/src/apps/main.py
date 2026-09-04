import os
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class EchoAfterSpeechProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        self._num_channels = 1

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._audio_buffer.extend(frame.audio)
            self._sample_rate = frame.sample_rate
            self._num_channels = frame.num_channels

            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            print(">>> Speech started")
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            print(f"<<< Speech stopped: bytes={len(self._audio_buffer)}")

            if self._audio_buffer:
                await self.push_frame(
                    OutputAudioRawFrame(
                        audio=bytes(self._audio_buffer),
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    )
                )

                self._audio_buffer.clear()

            return

        await self.push_frame(frame, direction)


async def run_pipeline(websocket: WebSocket):
    vad = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.5,
            min_volume=0.6,
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(
                vad_analyzer=vad,
                speech_activity_period=0.2,
                audio_idle_timeout=1.0,
            ),
            EchoAfterSpeechProcessor(),
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )

    runner = PipelineRunner()

    await runner.run(task)


app = FastAPI()

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/")
async def index():
    return RedirectResponse(url="gradio", status_code=307)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await run_pipeline(websocket)


demo = gr.Blocks()

with demo:
    gr.Markdown("# Voice Echo")
    gr.Markdown("Pipecat + WebSocket + Silero VAD")


app = gr.mount_gradio_app(
    app,
    demo,
    path="/gradio",
    head='<script type="module" src="/static/frontend.js"></script>',
    root_path="",
)


def entry_point():
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    entry_point()
