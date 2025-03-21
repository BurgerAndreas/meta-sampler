"""
PyTorch Lightning has a rich set of lifecycle hooks that are automatically called during different stages of training

LightningModule Hooks
Initialization Hooks
__init__ - Model initialization
setup(stage) - Called on every process when using distributed training
Training Loop Hooks
on_fit_start() - Called at the beginning of fit
on_fit_end() - Called at the end of fit
on_train_start() - Called at the beginning of training
on_train_end() - Called at the end of training
on_train_epoch_start() - Called at the beginning of a training epoch
on_train_epoch_end() - Called at the end of a training epoch
on_train_batch_start(batch, batch_idx) - Called at the beginning of a training batch
on_train_batch_end(outputs, batch, batch_idx) - Called at the end of a training batch
Validation Loop Hooks
on_validation_start() - Called at the beginning of validation
on_validation_end() - Called at the end of validation
on_validation_epoch_start() - Called at the beginning of a validation epoch
on_validation_epoch_end() - Called at the end of a validation epoch
on_validation_batch_start(batch, batch_idx) - Called at the beginning of a validation batch
on_validation_batch_end(outputs, batch, batch_idx) - Called at the end of a validation batch
Test Loop Hooks
on_test_start() - Called at the beginning of a test
on_test_end() - Called at the end of a test
on_test_epoch_start() - Called at the beginning of a test epoch
on_test_epoch_end() - Called at the end of a test epoch
on_test_batch_start(batch, batch_idx) - Called at the beginning of a test batch
on_test_batch_end(outputs, batch, batch_idx) - Called at the end of a test batch
Prediction Hooks
on_predict_start() - Called at the beginning of prediction
on_predict_end() - Called at the end of prediction
on_predict_epoch_start() - Called at the beginning of a prediction epoch
on_predict_epoch_end() - Called at the end of a prediction epoch
on_predict_batch_start(batch, batch_idx) - Called at the beginning of a prediction batch
on_predict_batch_end(outputs, batch, batch_idx) - Called at the end of a prediction batch
Step Methods
training_step(batch, batch_idx) - Training step
validation_step(batch, batch_idx) - Validation step
test_step(batch, batch_idx) - Test step
predict_step(batch, batch_idx) - Prediction step
Optimizer Hooks
configure_optimizers() - Configure optimizers
configure_callbacks() - Configure callbacks
optimizer_step() - Perform optimizer step
optimizer_zero_grad() - Zero gradients
backward(loss, optimizer, optimizer_idx) - Backward pass
State Hooks
on_save_checkpoint(checkpoint) - Called when saving a checkpoint
on_load_checkpoint(checkpoint) - Called when loading a checkpoin
"""

import lightning as L
import inspect

# Get all hooks from LightningModule
lightning_hooks = [
    method
    for method in dir(L.LightningModule)
    if method.startswith("on_")
    or method.endswith("_step")
    or method in ["setup", "configure_optimizers", "configure_callbacks"]
]
print("LightningModule hooks:")
for hook in lightning_hooks:
    print("", hook)

# Get all hooks from Callback
callback_hooks = [
    method
    for method in dir(L.Callback)
    if method.startswith("on_") or method.endswith("_step")
]
print("Callback hooks:")
for hook in callback_hooks:
    print("", hook)
