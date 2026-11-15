import torch

from battery_soh.training.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_scheduler_and_training_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=1,
    )
    scheduler.step(0.5)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        metrics={"val_recon": 0.5},
        training_state={"best_val": 0.5, "epochs_without_improvement": 2},
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=2.0e-3)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        restored_optimizer,
        factor=0.5,
        patience=1,
    )
    payload = load_checkpoint(
        path,
        restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert payload["epoch"] == 4
    assert payload["training_state"]["best_val"] == 0.5
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for restored, expected in zip(restored_model.parameters(), model.parameters()):
        assert torch.equal(restored, expected)
