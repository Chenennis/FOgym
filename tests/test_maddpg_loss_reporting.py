"""MADDPG training history must not fabricate losses before any optimizer update."""
from algorithms.MADDPG.fomaddpg.fomaddpg_training_methods import maddpg_loss_values


def test_maddpg_training_losses_are_taken_from_optimizer_output():
    losses = maddpg_loss_values({
        "actor_loss": -0.37, "critic_loss": 2.4,
        "training_iterations": 8,
    })
    assert losses == (-0.37, 2.4)


def test_maddpg_training_without_update_has_no_fake_losses():
    assert maddpg_loss_values(None) == (None, None)


def test_maddpg_training_rejects_incomplete_update_record():
    try:
        maddpg_loss_values({"actor_loss": 1.0})
    except KeyError:
        pass
    else:
        raise AssertionError("Incomplete optimizer records must not appear successful")
