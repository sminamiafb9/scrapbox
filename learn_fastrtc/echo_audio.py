import numpy as np
from fastrtc import ReplyOnPause, Stream


def echo(audio: tuple[int, np.ndarray]):
    yield audio


stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
