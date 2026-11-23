"""SQDDPG exploration noise changes must reach the underlying action sampler."""
import numpy as np

from algorithms.SQDDPG.fosqddpg.fosqddpg_adapter import FOSQDDPGAdapter


def test_sqddpg_adapter_propagates_updated_noise_scale():
    adapter = FOSQDDPGAdapter(
        state_dim=3, action_dim=5, num_agents=2,
        hidden_dim=16, batch_size=2, sample_size=2, device="cpu",
        noise_scale=0.3,
    )
    obs = {mid: np.zeros(3, dtype=np.float32) for mid in adapter.agent_ids}
    adapter.noise_scale = 0.05
    assert adapter.fosqddpg.noise_scale != adapter.noise_scale

    adapter.select_actions(obs, deterministic=False)
    assert adapter.fosqddpg.noise_scale == 0.05
