# Training runs

Purpose: stores training-run outputs. Producing stage: run the training command. Expected filename: `*/weights/best.pt`. Success signal: the run completes and writes `best.pt` under its `weights` directory. The artifact is absent from Git because trained model weights are generated and large.
