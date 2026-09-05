import gradio as gr
import websocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

WS_URL = "ws://127.0.0.1:8000/ws"


# =========================
# WebSocket Backend
# =========================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("Backend: WebSocket connected")

    try:
        while True:
            # Binary Frameを受信
            audio_bytes = await websocket.receive_bytes()

            # print(f"Backend received: {len(audio_bytes)} bytes")

            # Echo
            await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        print("Backend: WebSocket disconnected")


# =========================
# Gradio Frontend
# =========================


def create_frontend():

    connections: dict[str, websocket.WebSocket] = {}

    def connect(request: gr.Request) -> websocket.WebSocket:
        session_id = request.session_hash

        if session_id is None:
            raise ValueError("Session ID is required")

        if session_id not in connections:
            print(f"Frontend: WebSocket connecting session={session_id}")

            connections[session_id] = websocket.create_connection(WS_URL)

            print(f"Frontend: WebSocket connected session={session_id}")

        return connections[session_id]

    def send_audio(
        audio: tuple[int, object],
        request: gr.Request,
    ):
        if audio is None:
            return None

        sample_rate, audio_array = audio

        # print(
        #     f"Frontend received audio chunk: "
        #     f"sample_rate={sample_rate}, "
        #     f"samples={len(audio_array)}"  # type: ignore
        # )

        ws = connect(request)

        # NumPy array → raw PCM bytes
        audio_bytes = audio_array.tobytes()  # type: ignore

        # print(f"Frontend send: {len(audio_bytes)} bytes")

        # Binary Frameとして送信
        ws.send(
            audio_bytes,
            opcode=websocket.ABNF.OPCODE_BINARY,
        )

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
            outputs=[audio_output],
            stream_every=0.5,
        )

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
