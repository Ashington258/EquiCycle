import torch


def test_pytorch_version():
    print("Testing PyTorch version...")
    print("PyTorch version:", torch.__version__)

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("CUDA version:", torch.version.cuda)
        print("Number of available GPUs:", torch.cuda.device_count())
        print("Current GPU device:", torch.cuda.current_device())
        print(
            "Current GPU device name:",
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    test_pytorch_version()
