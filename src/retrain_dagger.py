"""M13: 現在のサロゲートモデルでDAgger風データを収集し、再学習する。

収集対象ICはM5の18通りのグリッド + M12の15通りの角速度スイープ
(既知のリグレッション事例を優先的にカバー)。
"""
import numpy as np
import torch

from src.dagger import collect_dagger_windows
from src.diagnose_amplitude_sweep import THETA0 as SWEEP_THETA0
from src.diagnose_amplitude_sweep import THETA_DOT0_RANGE
from src.evaluate_pid import ANGLE_OFFSETS, VELOCITY_OFFSETS
from src.model import NSSModel
from src.train import CURRICULUM, DAMPING, K_MAX, SEED, train

MODEL_PATH = "outputs/nss_model.pt"
DAGGER_N_STEPS = 400  # M5/M12と同じ8秒
DAGGER_STRIDE = 10


def load_current_model() -> NSSModel:
    model = NSSModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def make_collection_ics() -> np.ndarray:
    """M5の18通りグリッド + M12の15通りスイープを合わせた収集対象IC。"""
    m5_ics = [
        [np.pi + angle_offset, velocity_offset]
        for angle_offset in ANGLE_OFFSETS
        for velocity_offset in VELOCITY_OFFSETS
    ]
    m12_ics = [[SWEEP_THETA0, -v0] for v0 in THETA_DOT0_RANGE]
    return np.array(m5_ics + m12_ics)


if __name__ == "__main__":
    old_model = load_current_model()
    ics = make_collection_ics()
    print(f"collecting DAgger data from {len(ics)} ICs (old model closed-loop PID rollouts)...")

    extra_windows = collect_dagger_windows(
        old_model, ics, n_steps=DAGGER_N_STEPS, k=K_MAX, c=DAMPING, stride=DAGGER_STRIDE
    )
    print(f"collected {extra_windows[0].shape[0]} DAgger windows")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    new_model = train(curriculum=CURRICULUM, extra_windows=extra_windows, model_cls=NSSModel)

    torch.save(new_model.state_dict(), MODEL_PATH)
    print(f"saved to {MODEL_PATH}")
