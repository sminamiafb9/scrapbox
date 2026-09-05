import os
import tempfile

import gradio as gr
import websocket

WS_URL = "ws://127.0.0.1:8000/ws"

connections: dict[str, websocket.WebSocket] = {}


def connect(request: gr.Request) -> websocket.WebSocket:
    session_id = request.session_hash

    if session_id is None:
        raise ValueError("Session ID is required")

    if session_id not in connections:
        connections[session_id] = websocket.create_connection(WS_URL)

    return connections[session_id]


def send_audio(
    audio_path: str,
    request: gr.Request,
):
    if audio_path is None:
        return None

    ws = connect(request)

    # 録音した音声ファイルをbytesにする
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Frontend send: {len(audio_bytes)} bytes")

    # WebSocketで送信
    ws.send(
        audio_bytes,
        opcode=websocket.ABNF.OPCODE_BINARY,
    )

    # Backendから受信
    response_bytes = ws.recv()
    if isinstance(response_bytes, str):
        return

    print(f"Frontend received: {len(response_bytes)} bytes")

    # 受信したbytesを一時ファイルに保存
    suffix = os.path.splitext(audio_path)[1]
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as f:
        f.write(response_bytes)
        output_path = f.name

    print(f"Output file: {output_path}")

    return output_path


def disconnect(request: gr.Request):
    session_id = request.session_hash

    if session_id is None:
        return

    ws = connections.pop(session_id, None)

    if ws is not None:
        ws.close()


with gr.Blocks() as demo:
    gr.Markdown("# WebSocket Audio Echo")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="音声入力",
    )

    send_button = gr.Button("WebSocketで送信")

    audio_output = gr.Audio(
        type="filepath",
        label="Backendから返ってきた音声",
    )

    send_button.click(
        fn=send_audio,
        inputs=[audio_input],
        outputs=[audio_output],
    )

    demo.unload(disconnect)


demo.launch()
