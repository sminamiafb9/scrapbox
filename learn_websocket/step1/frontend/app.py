import gradio as gr
import websocket

WS_URL = "ws://localhost:8000/ws"

connections: dict[str, websocket.WebSocket] = {}


def connect(request: gr.Request) -> websocket.WebSocket:
    session_id = request.session_hash
    if session_id is None:
        raise ValueError("Session ID is required")

    if session_id not in connections:
        connections[session_id] = websocket.create_connection(WS_URL)

    return connections[session_id]


def send_message(
    message: str,
    history: list[dict],
    request: gr.Request,  # gradioが制御して渡す
):
    ws = connect(request)

    ws.send(message)
    response = ws.recv()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history


def disconnect(request: gr.Request):
    session_id = request.session_hash
    if session_id is None:
        return

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
        inputs=[textbox, chatbot],  # textboxの値とchatbotの履歴をsend_messageに渡す
        outputs=[textbox, chatbot],  # send_messageの返り値をtextboxとchatbotに渡す
    )

    demo.unload(disconnect)


demo.launch()
