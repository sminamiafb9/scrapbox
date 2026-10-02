# %% [markdown]
# ## Note
#
# **ランク学習（Learning to Rank）:**
# $x_i \succ x_j$（$x_i$が$x_j$より上位）に対して、
# $$f(x_i) > f(x_j)$$
# となるように学習する。
#
# %% [markdown]
# **線形ランキングモデル:**
# ランキングのスコアを
# $$f(x) = w \cdot x$$
# とすると、$w$を学習して
# $$w \cdot x_i > w \cdot x_j$$
# となるようにしたい。
#
# ここで、右辺を左辺に移項すると
# $$w \cdot x_i - w \cdot x_j > 0$$
#
# 整理すると
# $$w \cdot (x_i - x_j) > 0$$
#
# 同様に、順位が逆の場合は
# $$\begin{align*}
# &w \cdot x_i < w \cdot x_j \\
# &w \cdot x_i - w \cdot x_j < 0 \\
# &w \cdot (x_i - x_j) < 0
# \end{align*}$$
#
# したがって、
# **2つのデータの順位関係を正解ラベルとする2値分類問題**
# として捉えることができる。
#
# %% [markdown]
# **Perceptronの学習への対応:**
#
# - **1. 切片項について**
#   - 一般的な線形モデルを
#     $$f(x) = w \cdot x + b$$
#     とする。
#   - ペア比較では、
#     $$f(x_i) - f(x_j)
#     = (w \cdot x_i + b) - (w \cdot x_j + b)$$
#     $$= w \cdot (x_i - x_j)$$
#     となり、切片$b$は相殺される。
#   - → **ペア比較によるランキング学習では、切片$b$を考える必要がない。**
#
# - **2. 正解ラベルについて**
#   - 通常の2値分類では、入力$x$と正解ラベル$y \in \{-1,+1\}$を用いる。
#   - ランキングでは、2つのデータの順位関係が正解となる。
#   - ただし、ペアを作る際に必ず
#     **「上位のデータ − 下位のデータ」**
#     の順に並べれば、
#     $$x_i - x_j$$
#     は常に「上位 − 下位」となる。
#   - この場合、常に正しい方向を$+1$とみなせるため、更新式から$y$を省略できる。
#   - → **ペア作成時に順位を揃えることで、すべてのペアを$+1$ラベルとして扱える。**
#
# つまり、ランキング学習では
# $$x_i \succ x_j$$
# を
# $$x_i - x_j$$
# という「差分特徴量」に変換することで、通常の2値分類、特にPerceptronの学習と対応づけられる。

# %%

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from step4_perceptron import (
    Classifier,
    History,
    LogItem,
    Perceptron,
)

rng = np.random.default_rng(seed=123)


def init_rank_dataset(
    weights: NDArray | ArrayLike,
    noise_std: float = 1.0,
    n_samples: int = 300,
) -> tuple[NDArray, NDArray]:
    weights = np.asarray(weights)
    n_features = weights.shape[0]

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, n_features),
    )

    r = X @ weights
    r += rng.normal(
        loc=0.0,
        scale=noise_std,
        size=n_samples,
    )

    return X, r


class PairwisePerceptron(Perceptron):
    def __init__(self, ndim: int = 2, eta: float = 0.001):
        super().__init__(ndim=ndim, eta=eta)
        self.b = 0

    def train(self, Dx: NDArray, Dy: NDArray, epochs: int = 100) -> History:
        history = History(Dx, Dy, epochs)
        history.append(LogItem(self.copy()))

        for i in range(epochs):
            errors = 0
            for x_i, y_i in self.batch(Dx, Dy):
                if y_i * self.predict(x_i) <= 0:
                    history.append(LogItem(self.copy(), x_i, y_i))
                    self.w, _ = self._train_step(x_i, y_i)
                    errors += 1
            print(f"epoch {i + 1}: {errors} errors")
            if errors == 0:
                break
        return history

    def batch(self, Dx, Dy):
        for i, (x_i, y_i) in enumerate(zip(Dx, Dy)):
            for x_j, y_j in zip(Dx[i + 1 :], Dy[i + 1 :]):
                if y_i > y_j:
                    yield x_i - x_j, 1
                elif y_i < y_j:
                    yield x_j - x_i, 1

    def score(self, X: NDArray) -> NDArray:
        return X @ self.w

    def _train_step(self, x, y):
        w = self.w + self.eta * y * x
        return w, 0

    def decision_boundary(
        self, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> NDArray:
        raise NotImplementedError()


class RankTrainLogAnimation:
    def __init__(
        self,
        history: History,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ):
        self.history = history
        self.xlim = xlim
        self.ylim = ylim

        x_range = np.linspace(*xlim, 40)
        y_range = np.linspace(*ylim, 40)
        self.X_grid, self.Y_grid = np.meshgrid(x_range, y_range)
        self.grid = np.stack([self.X_grid, self.Y_grid], axis=-1)

        if history.Dy is None:
            raise ValueError("Dy is None")

        order = np.argsort(-history.Dy)
        self.Dx = history.Dx[order, :]

    def plot(
        self,
        interval: int = 150,
        figsize: tuple[int, int] = (5, 5),
        frame_sampling_rate: float = 1.0,
    ) -> Image:
        fig, ax = plt.subplots(ncols=2, figsize=(figsize[0] * 2, figsize[1]))
        ax[0].set_aspect("equal")
        ax[1].set_aspect("equal")

        n_frames = len(self.history)
        n_samples = max(1, round(n_frames * frame_sampling_rate))

        frame_indexes = np.linspace(
            0,
            n_frames - 1,
            n_samples,
            dtype=int,
        )
        print(f"""{n_frames = }, {n_samples = }""")

        ani = FuncAnimation(
            fig,
            self._animate,
            frames=frame_indexes,
            interval=interval,
            repeat=False,
            fargs=(ax, fig),
        )
        plt.close(fig)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anim.gif"
            ani.save(path, fps=int(1000 / interval), writer="pillow")

            data = path.read_bytes()
        return Image(data=data, format="gif")

    def _animate(
        self,
        frame_index: int,
        ax: NDArray,
        fig: Figure,
    ):
        ax0: Axes = ax[0]
        ax0.clear()
        log = self.history[frame_index]
        score = log.model.score(self.Dx)
        margin = score[None, :] - score[:, None]
        ax0.imshow(margin, cmap="RdBu")

        ax1: Axes = ax[1]
        ax1.clear()
        self.plot_heatmap(ax1, log.model)

        self.configure_axes(fig, frame_index)

    def plot_heatmap(self, ax: Axes, model: Classifier):
        Z_grid = model.score(self.grid)
        ax.imshow(
            Z_grid,
            extent=(
                self.X_grid[0, 0],
                self.X_grid[0, -1],
                self.Y_grid[0, 0],
                self.Y_grid[-1, 0],
            ),
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
        )
        ax.scatter(
            self.history.Dx[:, 0], self.history.Dx[:, 1], c=self.history.Dy, cmap="bwr"
        )

    def configure_axes(self, fig: Figure, frame_index: int):
        fig.suptitle(f"Updated: {frame_index}")


def main():
    Dx, Dy = init_rank_dataset(weights=[2, 2], noise_std=0.0, n_samples=300)

    model = PairwisePerceptron()
    history = model.train(Dx, Dy)
    output_frames = 150
    frame_sampling_rate = min(1.0, output_frames / len(history))

    log_anim = RankTrainLogAnimation(
        history,
        xlim=(-5, 5),
        ylim=(-5, 5),
    )
    display(
        log_anim.plot(
            figsize=(3, 3),
            frame_sampling_rate=frame_sampling_rate,
        )
    )


# %%

if __name__ == "__main__":
    main()
