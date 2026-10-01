# %%


from IPython.display import display
from numpy.typing import NDArray
from step4_perceptron import (
    History,
    LogItem,
)
from step5_linear_svm import LinearSupportVectorMachine
from step9_pairwise_perceptron import RankTrainLogAnimation, init_rank_dataset


class PairwiseLinearSupportVectorMachine(LinearSupportVectorMachine):
    def __init__(
        self,
        ndim: int = 2,
        eta: float = 0.0005,
        C: float = 5,
        tolerance: float = 0.01,
        patience: int = 10,
    ):
        super().__init__(ndim=ndim, eta=eta)
        self.C = C
        self.tolerance = tolerance
        self.patience = patience
        self._convergence_count = 0

        self.b = 0

    def train(self, Dx: NDArray, Dy: NDArray, epochs: int = 100) -> History:
        history = History(Dx, Dy, epochs)
        history.append(LogItem(self.copy()))

        for i in range(epochs):
            w_old = self.w[:]
            b_old = self.b
            for x_i, y_i in zip(Dx, Dy):
                history.append(LogItem(self.copy(), x_i, y_i))
                self.w, _ = self._train_step(x_i, y_i)

            delta = self._calculate_delta(w_old, b_old)
            print(f"epoch {i + 1}: {delta = }")
            if self._is_converged(delta):
                break
        return history

    def batch(self, Dx, Dy):
        Px = Dx[Dy == 1]
        Nx = Dx[Dy == -1]

        for x_pos in Px:
            for x_neg in Nx:
                yield x_pos - x_neg, 1

    def score(self, X: NDArray) -> NDArray:
        return X @ self.w

    def _train_step(self, x: NDArray, y: float) -> tuple[NDArray, float]:
        margin = y * self.score(x)
        if margin <= 1:
            w = (1 - self.eta) * self.w + self.eta * self.C * y * x
        else:
            w = (1 - self.eta) * self.w
        return w, 0

    def decision_boundary(
        self, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> NDArray:
        raise NotImplementedError()


def main():
    Dx, Dy = init_rank_dataset(weights=[2, 2], noise_std=0.0, n_samples=300)

    model = PairwiseLinearSupportVectorMachine()
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
