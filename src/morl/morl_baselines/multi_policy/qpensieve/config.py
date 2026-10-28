"""Independent single-policy baseline settings."""


def reference_qpensieve_kwargs(seed=0, net_arch=None, batch_size=256):
    """Return a fresh configuration for the single-policy baseline."""
    return dict(
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=batch_size,
        buffer_size=1_000_000,
        net_arch=list(net_arch) if net_arch is not None else [256, 256],
        prefer_set_size=4,
        q_replay_size=4,
        q_replay_start=20_000,
        q_replay_freq=5000,
        learning_starts=10_000,
        train_preference=None,
        train_preference_concentration=0.0,
        seed=seed,
    )
