import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Literal

import numpy as np
from fastrtc import AsyncStreamHandler, Stream, wait_for_item
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GeminiHandler(AsyncStreamHandler):
    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            input_sample_rate=16000,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit = asyncio.Event()

    def copy(self) -> "GeminiHandler":
        return GeminiHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
        )

    async def _send_audio_loop(self, session) -> None:
        logger.info("Send audio loop started")

        try:
            async for audio_bytes in self.stream():
                if audio_bytes:
                    await session.send_realtime_input(
                        audio={
                            "mime_type": "audio/pcm;rate=16000",
                            "data": audio_bytes,
                        }
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in _send_audio_loop")

    def _clear_output_queue(self):
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def start_up(self):
        logger.info("Gemini Live start")

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"api_version": "v1alpha"},
        )

        config = LiveConnectConfig(
            response_modalities=["AUDIO"],  # type: ignore
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name="Puck",
                    )
                )
            ),
        )

        async with client.aio.live.connect(
            model="gemini-3.1-flash-live-preview",
            config=config,
        ) as session:
            logger.info("Gemini Live connected")

            send_task = asyncio.create_task(self._send_audio_loop(session))

            try:
                async for response in session.receive():
                    if self.quit.is_set():
                        break

                    server_content = response.server_content
                    if server_content is not None:
                        if server_content.interrupted:
                            logger.info(
                                "Barge-in detected: Clearing output audio queue."
                            )
                            self._clear_output_queue()

                        if (
                            server_content.model_turn is not None
                            and server_content.model_turn.parts is not None
                        ):
                            for part in server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    array = np.frombuffer(
                                        part.inline_data.data,
                                        dtype=np.int16,
                                    )
                                    self.output_queue.put_nowait(
                                        (self.output_sample_rate, array)
                                    )

            finally:
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        logger.info("stream")

        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), 0.1)
                yield audio
            except (asyncio.TimeoutError, TimeoutError):
                pass

    async def receive(
        self,
        frame: tuple[int, np.ndarray],
    ) -> None:
        _sample_rate, array = frame
        array = array.squeeze()

        if array.dtype != np.int16:
            if np.issubdtype(array.dtype, np.floating):
                array = (array * 32767).astype(np.int16)
            else:
                array = array.astype(np.int16)

        self.input_queue.put_nowait(array.tobytes())

    async def emit(self) -> tuple[int, np.ndarray] | None:
        return await wait_for_item(self.output_queue)

    def shutdown(self) -> None:
        logger.info("shutdown")

        self.quit.set()


stream = Stream(
    handler=GeminiHandler(),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
