# learn websocket step2

音声を録音してバイナリファイルとしてwebsocketでやりとりする

```
# プロセス1
uv run python backend/app.py

# プロセス2
uv run python frontend/app.py
```

## チャット実装との差分

| | テキストチャット | 音声 |
|---|---|---|
| 入力 | テキスト | 録音した音声ファイル |
| Pythonで取得 | `str` | `bytes` |
| WebSocket送信 | Text Frame | Binary Frame |
| FastAPI受信 | `receive_text()` | `receive_bytes()` |
| Backend処理 | `message + "!"` | そのまま |
| WebSocket返却 | `send_text()` | `send_bytes()` |
| Frontend表示 | Chatbot | Audio |

## シーケンス図

### WebSocketの初期化

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Frontend
    participant Backend as Backend

    Note over Frontend,Backend: 初期化

    User->>Frontend: セッション開始

    Frontend->>Frontend: session_hashを取得
    Frontend->>Frontend: connections[session_id] を確認

    alt WebSocket未接続
        Frontend->>Backend: WebSocket接続を開始
        Backend->>Backend: await websocket.accept()
        Backend-->>Frontend: 接続確立
        Frontend->>Frontend: connectionsに保存
    else WebSocket接続済み
        Frontend->>Frontend: 既存のWebSocketを利用
    end

    Note over Frontend,Backend: WebSocket接続を維持

    User->>Frontend: セッション終了

    Frontend->>Frontend: connections.pop(session_id)
    Frontend->>Backend: ws.close()
    Backend->>Backend: WebSocketDisconnect
    Backend->>Backend: 切断処理

    Note over Frontend,Backend: WebSocket終了
```

### Frontend

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Frontend<br/>Gradio
    participant Backend as Backend<br/>WebSocket

    User->>Frontend: 音声を録音
    User->>Frontend: 録音を終了

    Frontend->>Frontend: 音声ファイルのパスを取得
    Frontend->>Frontend: 音声ファイルをbytesに変換

    User->>Frontend: 「WebSocketで送信」をクリック

    Frontend->>Backend: WebSocket Binary Frame<br/>audio bytesを送信
    Backend-->>Frontend: WebSocket Binary Frame<br/>audio bytesを受信

    Frontend->>Frontend: 受信したbytesを一時ファイルに保存
    Frontend-->>User: 音声を再生
```

### Backend

```mermaid
sequenceDiagram
    participant Frontend as Frontend<br/>Gradio
    participant Backend as Backend<br/>WebSocket

    Frontend->>Backend: WebSocket Binary Frame<br/>audio bytes

    Backend->>Backend: await websocket.receive_bytes()
    Backend->>Backend: audio bytesを取得

    Backend->>Backend: len(audio_bytes)を確認

    Backend-->>Frontend: WebSocket Binary Frame<br/>同じaudio bytes

    Note over Backend: 音声データを加工せず<br/>そのまま返却

    Backend->>Backend: 次のWebSocketメッセージを待機
```
