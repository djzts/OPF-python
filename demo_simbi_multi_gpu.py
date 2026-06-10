from qhdopt import QHD


def main():
    # Small quadratic demo; run from a Python environment that has qhdopt,
    # simulated-bifurcation, torch, and CUDA installed.
    qhd_model = QHD.QP(
        Q=[
            [1.0, 0.2],
            [0.2, 1.5],
        ],
        b=[-0.8, -0.4],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
    )

    qhd_model.simbi_setup(
        resolution=20,
        shots=128,
        agents=8192,
        max_steps=8000,
        device="cuda",
        multi_gpu=True,
        num_gpus=None,
        best_only=False,
    )

    response = qhd_model.optimize(refine=False, verbose=1)
    print("raw sample count:", len(response.samples))
    print("minimizer:", response.minimizer)
    print("minimum:", response.minimum)


if __name__ == "__main__":
    main()
