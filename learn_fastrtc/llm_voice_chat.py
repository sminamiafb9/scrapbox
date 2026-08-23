import logging

import mlx_whisper
import numpy as np
import ollama
from fastrtc import AlgoOptions, ReplyOnPause, Stream
from supertonic import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHISPER_MODEL = "mlx-community/whisper-small-mlx"

ollama_client = ollama.Client(host="http://localhost:11434")
tts = TTS(auto_download=True)
tts_voice = tts.get_voice_style(voice_name="M1")


def response(
    audio: tuple[int, np.ndarray],
):
    # ################################################################################
    # STT
    # --------------------------------------------------------------------------------
    _sample_rate, audio_array = audio
    logger.info(
        "sample_rate=%s shape=%s dtype=%s size=%s min=%s max=%s",
        _sample_rate,
        audio_array.shape,
        audio_array.dtype,
        audio_array.size,
        audio_array.min() if audio_array.size else None,
        audio_array.max() if audio_array.size else None,
    )

    audio_array = audio_array.reshape(-1).astype(np.float32) / 32768.0
    result = mlx_whisper.transcribe(
        audio_array,
        path_or_hf_repo=WHISPER_MODEL,
        language="ja",
    )

    prompt = str(result["text"]).strip()

    if not prompt:
        return
    logger.info(f"User: {prompt}")

    # ################################################################################
    # LLM
    # --------------------------------------------------------------------------------
    response = ollama_client.chat(
        model="gemma4:e2b-it-qat",
        messages=[
            {
                "role": "system",
                "content": "あなたは音声会話アシスタントです。日本語で簡潔に答えてください。回答は短くしてください。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "num_predict": 128,
        },
    )
    response_text = response["message"]["content"]
    if not response_text:
        return
    logger.info(f"Assistant: {response_text}")

    # ################################################################################
    # TTS
    # --------------------------------------------------------------------------------
    wav, _duration = tts.synthesize(
        response_text,
        voice_style=tts_voice,
    )

    audio_array = np.asarray(wav)
    if audio_array.ndim == 1:
        audio_array = audio_array.reshape(1, -1)

    yield (44100, audio_array)


algo_options = AlgoOptions(
    audio_chunk_duration=0.4,
    started_talking_threshold=0.2,
    speech_threshold=0.1,
    max_continuous_speech_s=10.0,
)

stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(
        response,
        algo_options=algo_options,
    ),
)

stream.ui.launch()
