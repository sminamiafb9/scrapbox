# Learn FastRTC

[gradio-app/fastrtc | Github](https://github.com/gradio-app/fastrtc)

## memo

- setup
  - `uv sync`
- echo_audio.py
  - `uv run python echo_audio.py`
  - README記載のQuickstartのコードを動かす
  - サーバが起動して、WebブラウザでアクセスするとビルトインのUIが起動
  - 発話を拾って、区切りを検出すると発話をそのままエコーする
- llm_voice_chat.py
  - `uv run python llm_voice_chat.py`
  - README記載のQuickstartのコードをollamaなどローカルLLMで再現した物
  - 精度などは悪いが音声入力 > 書き起こし > LLM入力 > 結果受け取り > 音声合成 > 音声出力の流れを追うことができる
- gemini_voice_chat.py
  - `export GEMINI_API_KEY=<your key>`
  - `uv run python gemini_voice_chat.py`
  - README記載のQuickstartのコードをGeminiで再現した物
  - ローカルLLMよりは応答をしてくれる
  - gemini-3.1-flash-tts-previeはRateLimitの制限が厳しい
- echo_audio_live.py
  - `uv run python echo_audio_live.py`
  - AsyncStreamHandlerを用いた全二重通信のサンプル
  - 発話を即時そのままエコーする(ハウリングするため注意)
- delayed_echo_audio_live.py
  - `uv run python delayed_echo_audio_live.py`
  - AsyncStreamHandlerを用いた全二重通信のサンプル
  - 発話を1フレーム遅延させてそのままエコーする(ハウリングするため注意)

## note

### 逐次フローの考え方

```mermaid
sequenceDiagram
    autonumber
    actor User as ユーザー
    participant WebBrowser as Web Browser
    participant Frontend as Frontend<br/>(Gradio)
    participant Backend as Backend<br/>(FastRTC)
    participant Handler as Handler<br/>(VAD)
    participant STT as STT
    participant LLM as LLM
    participant TTS as TTS

    User->>WebBrowser: UI操作 / 音声入力
    WebBrowser->>Frontend: ui component
    Frontend->>Backend: WebRTC (音声ストリーム送信)
    Backend->>Handler: audio data (音声データ)
    
    rect rgba(255, 255, 255, 0.2)
        note over Handler, TTS: リアルタイム処理パイプライン
        Handler->>STT: audio chunk (発話区間の音声)
        STT->>LLM: text (認識テキスト)
        LLM->>TTS: text (応答テキスト)
    end

    TTS->>Backend: audio data (合成音声データ)
    Backend->>Frontend: WebRTC (音声ストリーム返送)
    Frontend->>WebBrowser: speaker (音声再生)
    WebBrowser-->>User: 音声出力
```

- VAD(Voice Activity Detection / 発話区間検出)で発話区切りが検出されると、Text to Speach、LLM、Speach to Textと逐次データが流れていく構成
- 半二重通信で交代で話す構成になっている
