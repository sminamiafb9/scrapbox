import gradio as gr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(message + "!")
    except WebSocketDisconnect:
        print("WebSocket disconnected")


def create_frontend():
    connections = {}

    def connect(request: gr.Request):
        session_id = request.session_hash

        if session_id not in connections:
            import websocket

            connections[session_id] = websocket.create_connection(
                "ws://127.0.0.1:8000/ws"
            )

        return connections[session_id]

    def send_message(
        message: str,
        history: list[dict],
        request: gr.Request,
    ):
        ws = connect(request)

        ws.send(message)
        response = ws.recv()

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        return "", history

    def disconnect(request: gr.Request):
        session_id = request.session_hash

        ws = connections.pop(session_id, None)

        if ws is not None:
            ws.close()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="WebSocket Chat")

        textbox = gr.Textbox(
            placeholder="メッセージを入力してください",
            show_label=False,
        )

        send_button = gr.Button("送信")

        send_button.click(
            fn=send_message,
            inputs=[textbox, chatbot],
            outputs=[textbox, chatbot],
        )

        demo.unload(disconnect)

    return demo


frontend = create_frontend()

app = gr.mount_gradio_app(
    app,
    frontend,
    path="/",
)


def entry_point():
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    entry_point()
