import numpy as np

def fps_on_sphere(directions: np.ndarray, n_select: int,
                  seed: int = None) -> np.ndarray:
    """
    Farthest Point Sampling on unit sphere.
    Equation 2 from SARDInn paper (Wang et al. 2025).

    Args:
        directions: (N, 3) unit vectors on sphere
        n_select:   number of directions to select
        seed:       random seed for reproducibility

    Returns:
        indices: (n_select,) indices into directions array
    """
    N = len(directions)
    assert n_select >= 1
    assert n_select <= N, f"Cannot select {n_select} from {N}"

    # Normalize to unit sphere
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    dirs  = directions / np.clip(norms, 1e-8, None)

    rng      = np.random.default_rng(seed)
    selected = [int(rng.integers(N))]
    dist     = np.full(N, np.inf)

    for _ in range(n_select - 1):
        last     = selected[-1]
        diff     = dirs - dirs[last]
        new_dist = np.sqrt((diff ** 2).sum(axis=1))
        dist     = np.minimum(dist, new_dist)
        dist_tmp = dist.copy()
        dist_tmp[selected] = -1.0
        selected.append(int(np.argmax(dist_tmp)))

    return np.array(selected, dtype=np.int64)


def validate_fps(directions: np.ndarray, n_select: int,
                 n_trials: int = 5) -> bool:
    """
    Validate FPS gives better angular coverage than random.
    Returns True if FPS consistently outperforms random.
    """
    dirs = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def min_gap(idx):
        d    = dirs[idx]
        dots = np.abs(d @ d.T)
        dots = np.clip(dots, -1, 1)
        angs = np.arccos(dots) * 180 / np.pi
        np.fill_diagonal(angs, 0)
        return angs[angs > 0].min()

    fps_gaps, rand_gaps = [], []
    for t in range(n_trials):
        fps_gaps.append(min_gap(fps_on_sphere(dirs, n_select, seed=t)))
        rand_gaps.append(min_gap(
            np.random.choice(len(dirs), n_select, replace=False)))

    fps_mean  = np.mean(fps_gaps)
    rand_mean = np.mean(rand_gaps)
    print(f"FPS  min angular gap: {fps_mean:.2f}°")
    print(f"Rand min angular gap: {rand_mean:.2f}°")
    print(f"FPS better: {fps_mean > rand_mean}")
    return fps_mean > rand_mean


if __name__ == '__main__':
    # Quick self-test
    N      = 90
    golden = np.pi * (3 - np.sqrt(5))
    dirs   = []
    for i in range(N):
        y = 1 - (i / (N - 1)) * 2
        r = np.sqrt(max(0, 1 - y * y))
        dirs.append([r * np.cos(golden * i), y, r * np.sin(golden * i)])
    dirs = np.array(dirs)

    print("Testing FPS on 90 HCP-like directions, selecting 30...")
    result = validate_fps(dirs, 30)
    print("PASSED" if result else "FAILED")
