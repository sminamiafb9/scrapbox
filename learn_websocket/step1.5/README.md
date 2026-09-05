# learn websocket step1.5

step1のFrontendのGradioサーバーとBackendのFastAPIサーバーをひとつのプロセスに統合したもの

```bash
uv run python server.py
```

```mermaid
flowchart LR
    B["Browser<br/>Gradio Front"]

    subgraph S["同じサーバー / Uvicorn :8000"]
        G["Gradio Server<br/>(Python)"]
        W["FastAPI<br/>WebSocket /ws"]
    end

    B <-->|"HTTP"| G
    G <-->|"WebSocket<br/>ws://127.0.0.1:8000/ws"| W
```

- Gradioは、ブラウザ側（Gradio Frontend）とサーバー側（Gradio Server / Python）から構成される
- WebSocket接続を張っているのは、Gradioのサーバー側PythonとFastAPIのWebSocketエンドポイント /ws の間
- かつ、同一ホスト内の同一プロセスでのローカルホスト通信となっている
