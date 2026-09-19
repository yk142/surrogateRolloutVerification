import numpy as np

from src.diagnose_linearization import true_jacobian


def test_true_jacobian_matches_saddle_structure():
    """倒立平衡点は鞍点(不安定)であること: 固有値の一方は|λ|>1, 他方は|λ|<1。"""
    A, _ = true_jacobian()
    eigvals = np.abs(np.linalg.eigvals(A))
    assert np.any(eigvals > 1.0)
    assert np.any(eigvals < 1.0)
