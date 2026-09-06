import asyncio

import gradio as gr
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

WS_URL = "ws://127.0.0.1:8000/ws"


# =========================
# WebSocket Backend
# =========================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def receive_loop():
        try:
            while True:
                audio_bytes = await websocket.receive_bytes()
                await input_queue.put(audio_bytes)

        except WebSocketDisconnect:
            pass

    async def send_loop():
        try:
            while True:
                audio_bytes = await output_queue.get()
                await websocket.send_bytes(audio_bytes)

        except WebSocketDisconnect:
            pass

    async def process_loop():
        while True:
            audio_bytes = await input_queue.get()

            # 音声処理
            result = audio_bytes

            await output_queue.put(result)

    receive_task = asyncio.create_task(receive_loop())
    send_task = asyncio.create_task(send_loop())
    process_task = asyncio.create_task(process_loop())

    try:
        await receive_task

    finally:
        for task in (
            receive_task,
            send_task,
            process_task,
        ):
            task.cancel()

        await asyncio.gather(
            receive_task,
            send_task,
            process_task,
            return_exceptions=True,
        )


# =========================
# Gradio Frontend
# =========================


class GradioSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._ws: websockets.ClientConnection | None = None

    async def connect(self) -> websockets.ClientConnection:
        self._ws = await websockets.connect(WS_URL)

    async def send_audio(self, audio: tuple[int, object]):
        if audio is None or self._ws is None:
            return

        _sample_rate, audio_array = audio
        audio_bytes = audio_array.tobytes()  # type: ignore
        await self._ws.send(audio_bytes)


def create_frontend():

    async def recv_audio():
        # BackendからEchoを受信
        response_bytes = ws.recv()

        if isinstance(response_bytes, str):
            print("Frontend received text frame")
            return None

        # print(f"Frontend received: {len(response_bytes)} bytes")

        # raw PCM bytes → NumPy array
        import numpy as np

        response_array = np.frombuffer(
            response_bytes,
            dtype=audio_array.dtype,  # type: ignore
        )

        # Gradio Audio Outputへ返す
        return (
            sample_rate,
            response_array,
        )

    def initialize_session(request: gr.Request):
        session_id = request.session_hash

    def disconnect(request: gr.Request):
        session_id = request.session_hash

        if session_id is None:
            return

        ws = connections.pop(session_id, None)

        if ws is not None:
            print(f"Frontend: WebSocket closing session={session_id}")

            ws.close()

    # =========================
    # Gradio UI
    # =========================

    with gr.Blocks() as demo:
        gr.Markdown("# WebSocket Audio Streaming Echo")

        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="音声入力",
        )

        audio_output = gr.Audio(
            type="numpy",
            streaming=True,
            autoplay=True,
            label="Echo Output",
        )

        audio_input.stream(
            fn=send_audio,
            inputs=[audio_input],
            stream_every=0.5,
        )

        demo.load(initialize_session)
        demo.unload(disconnect)

    return demo


# =========================
# Mount Gradio
# =========================

frontend = create_frontend()

app = gr.mount_gradio_app(
    app,
    frontend,
    path="/",
)


# =========================
# Server
# =========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
