from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='/path/to/store/weights.ckpt',
    save_best_only=True,
    verbose=True,
    monitor='val_loss',
    mode='min'
)