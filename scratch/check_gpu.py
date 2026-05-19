import torch
import sys
import psutil
import os

def check_gpu():
    print("-" * 30)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA (GPU) Available? {'YES' if cuda_available else 'NO'}")
    
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"VRAM Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Current GPU Device: {torch.cuda.current_device()}")
    else:
        print("WARNING: CUDA is NOT available. PyTorch is using CPU.")
        
    # System memory info
    mem = psutil.virtual_memory()
    print(f"Total System RAM: {mem.total / 1024**3:.2f} GB")
    print(f"Available System RAM: {mem.available / 1024**3:.2f} GB")
    print(f"RAM Usage: {mem.percent}%")
    
    # Current process info
    process = psutil.Process(os.getpid())
    print(f"Current Process Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")
    print("-" * 30)

if __name__ == "__main__":
    check_gpu()
