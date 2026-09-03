import numpy as np
from fastrtc import ReplyOnPause, Stream


def echo(audio: tuple[int, np.ndarray]):
    yield audio


def entry_point():
    stream = Stream(
        handler=ReplyOnPause(echo),
        modality="audio",
        mode="send-receive",
    )

    stream.ui.launch()


if __name__ == "__main__":
    entry_point()
