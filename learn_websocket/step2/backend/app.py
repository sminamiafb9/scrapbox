from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            print(f"Backend received: {len(audio_bytes)} bytes")

            await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        print("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
