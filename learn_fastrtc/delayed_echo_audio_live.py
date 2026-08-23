import asyncio
import base64
import logging
from collections import deque
from collections.abc import AsyncGenerator
from typing import Literal

import numpy as np
from fastrtc import AsyncStreamHandler, Stream, wait_for_item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def encode_audio(data: np.ndarray) -> str:
    return base64.b64encode(data.tobytes()).decode("UTF-8")


class DelayedEchoHandler(AsyncStreamHandler):
    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
        delay_chunks: int = 1,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            input_sample_rate=16000,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

        self.delay_buffer: deque[tuple[int, np.ndarray]] = deque()
        self.delay_chunks = delay_chunks
        self.quit = asyncio.Event()

    def copy(self) -> "DelayedEchoHandler":
        return DelayedEchoHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
            delay_chunks=self.delay_chunks,
        )

    async def start_up(self):
        logger.info("start up")
        self.worker = asyncio.create_task(self._process_audio())

    async def stream(self) -> AsyncGenerator[bytes, None]:
        logger.info("stream")

        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.input_queue.get(), 0.1)
                yield audio
            except (asyncio.TimeoutError, TimeoutError):
                pass

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        self.input_queue.put_nowait(frame)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        return await wait_for_item(self.output_queue)

    async def _process_audio(self):
        while not self.quit.is_set():
            frame = await self.input_queue.get()

            self.delay_buffer.append(frame)

            if len(self.delay_buffer) > self.delay_chunks:
                delayed_frame = self.delay_buffer.popleft()
                self.output_queue.put_nowait(delayed_frame)

    def shutdown(self) -> None:
        logger.info("shutdown")

        self.quit.set()

        if self.worker is not None:
            self.worker.cancel()


stream = Stream(
    handler=DelayedEchoHandler(),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
