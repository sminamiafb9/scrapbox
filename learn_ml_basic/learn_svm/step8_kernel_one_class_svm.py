# %%

from IPython.display import display
from numpy.typing import NDArray
from step4_perceptron import TrainLogAnimation
from step6_kernel_svm import History, LogItem, RandomFourierFeatures
from step7_one_class_svm import (
    LinearOneClassSupportVectorMachine,
    init_single_class_dataset,
)


class KernelOneClassSupportVectorMachine(LinearOneClassSupportVectorMachine):
    def __init__(
        self,
        ndim: int = 2,
        gamma: float = 0.5,
        D: int = 3000,
        eta: float = 0.0003,
        nu: float = 0.05,
        tolerance: float = 0.1,
        patience: int = 30,
    ):
        super().__init__(ndim=D * 2, eta=eta)
        self.nu = nu
        self.rff = RandomFourierFeatures(input_features=ndim, gamma=gamma, D=D)
        self.D = D
        self.tolerance = tolerance
        self.patience = patience
        self._convergence_count = 0

    def train(self, Dx: NDArray, Dy: NDArray, epochs: int = 100) -> History:
        history = History(Dx, Dy, epochs)
        history.append(LogItem(self.copy()))

        Dz = self.rff.transform(Dx)

        for i in range(epochs):
            w_old = self.w[:]
            b_old = self.b
            for x_i, y_i, z_i in zip(Dx, Dy, Dz):
                history.append(LogItem(self.copy(), x_i, y_i))
                self.w, self.b = self._train_step(z_i, y_i)

            delta = self._calculate_delta(w_old, b_old)
            print(f"epoch {i + 1}: {delta = }")
            if self._is_converged(delta):
                break
        return history

    def score(self, X: NDArray) -> NDArray:
        if X.shape[-1] != 2 * self.D:
            X = self.rff.transform(X)
        return super().score(X)


# %%


def main():
    Dx, Dy = init_single_class_dataset(mean=[0, 0], std=1.5)

    model = KernelOneClassSupportVectorMachine()
    history = model.train(Dx, Dy)
    output_frames = 150
    frame_sampling_rate = min(1.0, output_frames / len(history))

    log_anim = TrainLogAnimation(
        history,
        xlim=(-5, 5),
        ylim=(-5, 5),
    )
    display(
        log_anim.plot(
            figsize=(3, 3), frame_sampling_rate=frame_sampling_rate, update=False
        )
    )


if __name__ == "__main__":
    main()
