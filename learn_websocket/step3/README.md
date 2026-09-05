# learn websocket step3

音声ストリーミングをWebSocketでechoする

## シーケンス図

### 全体のフロー

```mermaid
sequenceDiagram
    actor User as User
    participant Browser as Browser<br/>Gradio UI
    participant Gradio as Gradio Server<br/>Python
    participant WSClient as websocket-client
    participant Backend as FastAPI<br/>WebSocket Backend

    Note over Browser,Backend: アプリ起動

    User->>Browser: http://127.0.0.1:8000 を開く
    Browser->>Gradio: HTTP Request
    Gradio-->>Browser: Gradio UIを返す

    Note over Browser,Backend: 音声ストリーミング開始

    User->>Browser: マイクで録音

    Browser->>Gradio: 音声Chunk
    Gradio->>Gradio: send_audio()

    Gradio->>Gradio: sample_rate, audio_arrayを取得
    Gradio->>Gradio: NumPy Array → raw PCM bytes

    alt 初回Chunk
        Gradio->>WSClient: connect(request)
        WSClient->>Backend: WebSocket接続
        Backend->>Backend: await websocket.accept()
        Backend-->>WSClient: WebSocket接続確立
        WSClient->>Gradio: WebSocketを返す
        Gradio->>Gradio: connections[session_id]に保存
    else 2回目以降
        Gradio->>Gradio: 保存済みWebSocketを利用
    end

    Gradio->>WSClient: Binary Frame送信
    WSClient->>Backend: audio bytes

    Backend->>Backend: receive_bytes()
    Backend->>Backend: audio bytesをそのまま保持

    Backend-->>WSClient: Binary Frame<br/>audio bytes
    WSClient-->>Gradio: response_bytes

    Gradio->>Gradio: raw PCM bytes → NumPy Array
    Gradio->>Gradio: (sample_rate, response_array)

    Gradio-->>Browser: Audio OutputへChunk
    Browser-->>User: Echo音声を再生

    Note over Browser,Backend: 次のChunkでも同じ処理を繰り返す

    User->>Browser: 録音終了

    Browser->>Gradio: stream終了
    Gradio->>Gradio: demo.unload()
    Gradio->>WSClient: ws.close()
    WSClient->>Backend: WebSocket切断
    Backend->>Backend: WebSocketDisconnect

    Note over Browser,Backend: WebSocket終了
```

- Gradio Frontend
  - Audio入出力のUI
- Gradio Server
  - Audioデータのストリーム受け取り
- FastAPI Server
  - Audtioデータを加工 (そのまま返す)
- 経路
  - Gradio FE <-Audio-> Gradio BE <-Audio/WS-> FastAPI
