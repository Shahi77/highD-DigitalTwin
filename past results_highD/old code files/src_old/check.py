import torch
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("Device count:", torch.cuda.device_count())
print("Selected device:", "mps" if torch.backends.mps.is_available() else "cpu")
