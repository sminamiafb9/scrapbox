import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Callable
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


def convert_to_pcm16(array: np.ndarray) -> np.ndarray:
    """NumPy 配列を 16bit signed int (PCM) 配列に変換するヘルパー関数"""

    if array.dtype == np.int16:
        return array

    if np.issubdtype(array.dtype, np.floating):
        # [-1, 1] をint16の値域に変換する
        return (array * 32767).astype(np.int16)

    return array.astype(np.int16)


QUEUE_STREAM_INPUT_END = object()


async def queue_stream(
    input_queue: asyncio.Queue[bytes],
) -> AsyncGenerator[bytes, None]:
    while True:
        try:
            item = await input_queue.get()
            if item is QUEUE_STREAM_INPUT_END:
                return
            yield item
        except TimeoutError:
            pass


class FrontendHandler(AsyncStreamHandler):
    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
    ) -> None:
        """Stream初期化時に呼び出し"""

        logger.info("__init__")
        super().__init__(
            expected_layout,
            output_sample_rate,
            input_sample_rate=16000,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit = asyncio.Event()

    def copy(self) -> "FrontendHandler":
        """接続開始時に呼び出し"""

        logger.info("copy")

        return FrontendHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
        )

    async def start_up(self) -> None:
        """copy後の初期化処理(接続毎初期化)"""

        logger.info("start_up")

        gemini_handler = GeminiHandler(
            self.input_queue,
            self.output_queue,
            self.quit,
            self.output_sample_rate,
            self.clear_output_audio,
        )
        await gemini_handler.run()

    async def receive(
        self,
        frame: tuple[int, np.ndarray],
    ) -> None:
        """ブラウザから音声フレームを受け取った時の処理"""

        # logger.info("recive")

        _sample_rate, array = frame
        array = array.squeeze()
        array_pcm = convert_to_pcm16(array)

        # 受け取った音声をinput queueへ詰める
        self.input_queue.put_nowait(array_pcm.tobytes())

    async def emit(self) -> tuple[int, np.ndarray] | None:
        """ブラウザへ音声を送りたい時の処理"""

        # logger.info("emit")

        # output queueに値が入るのを待って返却する
        return await wait_for_item(self.output_queue)

    def shutdown(self) -> None:
        """ブラウザ切断/通話停止などの処理"""

        logger.info("shutdown")
        self.input_queue.put_nowait(QUEUE_STREAM_INPUT_END)
        self.quit.set()

    def clear_output_audio(self) -> None:
        if self._clear_queue is not None:
            self.clear_queue()


class GeminiHandler:
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        quit: asyncio.Event,
        output_sample_rate: int,
        on_barge_in_callback: Callable,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.quit = quit
        self.output_sample_rate = output_sample_rate
        self.on_barge_in_callback = on_barge_in_callback

        self._discard_interrupted_turn = False

    async def run(self):
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
            receive_task = asyncio.create_task(self._receive_audio_loop(session))

            try:
                await asyncio.gather(send_task, receive_task)
            finally:
                send_task.cancel()
                receive_task.cancel()

                await asyncio.gather(send_task, receive_task, return_exceptions=True)

    async def _send_audio_loop(self, session) -> None:
        logger.info("send audio loop started")

        try:
            async for audio_bytes in queue_stream(self.input_queue):
                if self.quit.is_set():
                    break

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

    async def _receive_audio_loop(self, session) -> None:
        logger.info("receive audio loop started")

        # session.reciveが1turnで終了するためwhileで会話を続ける
        while not self.quit.is_set():
            async for response in session.receive():
                if self.quit.is_set():
                    break

                server_content = response.server_content
                if server_content is None:
                    continue

                if server_content.interrupted:
                    logger.info("Barge-in detected: Clearing pending output audio.")
                    self._on_barge_in()

                if (
                    server_content.model_turn is not None
                    and server_content.model_turn.parts is not None
                    and not self._discard_interrupted_turn
                ):
                    for part in server_content.model_turn.parts:
                        self._enqueue_audio(part)

                if server_content.turn_complete:
                    self._discard_interrupted_turn = False

    def _on_barge_in(self) -> None:
        self._discard_interrupted_turn = True
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.on_barge_in_callback()

    def _enqueue_audio(self, part) -> None:
        if part.inline_data and part.inline_data.data:
            array = np.frombuffer(
                part.inline_data.data,
                dtype=np.int16,
            )
            self.output_queue.put_nowait((self.output_sample_rate, array))


stream = Stream(
    handler=FrontendHandler(),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
