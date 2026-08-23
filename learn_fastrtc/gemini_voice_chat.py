import base64
import io
import logging
import wave

import numpy as np
from fastrtc import AlgoOptions, ReplyOnPause, Stream
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gemini_client = genai.Client()  # Set the API key in GEMINI_API_KEY.


def audio_to_wav_bytes(
    sample_rate: int,
    audio_array: np.ndarray,
) -> bytes:
    audio_array = audio_array.reshape(-1).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_array.tobytes())

    return buffer.getvalue()


def response(
    audio: tuple[int, np.ndarray],
):
    # ################################################################################
    # STT
    # --------------------------------------------------------------------------------
    sample_rate, audio_array = audio
    logger.info(
        "sample_rate=%s shape=%s dtype=%s size=%s min=%s max=%s",
        sample_rate,
        audio_array.shape,
        audio_array.dtype,
        audio_array.size,
        audio_array.min() if audio_array.size else None,
        audio_array.max() if audio_array.size else None,
    )

    audio_bytes = audio_to_wav_bytes(
        sample_rate,
        audio_array,
    )
    result = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            (
                "この音声を日本語で正確に文字起こししてください。\n"
                "文字起こし結果だけを返してください。"
            ),
            {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_bytes,
                }
            },
        ],
    )

    prompt = None
    if result.text is not None:
        prompt = result.text.strip()

    if not prompt:
        return
    logger.info(f"User: {prompt}")

    # ################################################################################
    # LLM
    # --------------------------------------------------------------------------------
    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    response_text = None
    if response.text is not None:
        response_text = response.text.strip()

    if not response_text:
        return
    logger.info(f"Assistant: {response_text}")

    # ################################################################################
    # TTS
    # --------------------------------------------------------------------------------
    tts_response = gemini_client.interactions.create(
        model="gemini-3.1-flash-tts-preview",
        input=response_text,
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]},
    )

    output_audio = tts_response.output_audio  # type: ignore
    if output_audio is None or output_audio.data is None:
        logger.error("Gemini TTS returned no audio")
        return

    audio_bytes = base64.b64decode(output_audio.data)

    audio_array = np.frombuffer(
        audio_bytes,
        dtype=np.int16,
    ).reshape(1, -1)

    yield (24000, audio_array)


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
