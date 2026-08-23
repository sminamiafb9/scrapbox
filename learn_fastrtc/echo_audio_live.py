import asyncio
import base64
import logging
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


class EchoHandler(AsyncStreamHandler):
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
        self.queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()

    def copy(self) -> "EchoHandler":
        return EchoHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
        )

    async def start_up(self):
        logger.info("start up")

    async def stream(self) -> AsyncGenerator[bytes, None]:
        logger.info("stream")

        while not self.quit.is_set():
            try:
                audio = await asyncio.wait_for(self.queue.get(), 0.1)
                yield audio
            except (asyncio.TimeoutError, TimeoutError):
                pass

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        self.queue.put_nowait(frame)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        return await wait_for_item(self.queue)

    def shutdown(self) -> None:
        self.quit.set()


stream = Stream(
    handler=EchoHandler(),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
