param(
    [string]$Python = (Get-Command python -ErrorAction Stop).Source,
    [string]$BuildDirectory = "build/emc-cuda-ninja"
)

$ErrorActionPreference = "Stop"

Write-Output "NVIDIA"
& nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
& nvidia-smi | Select-Object -First 3

Write-Output "CUDA toolkit"
& nvcc --version | Select-Object -Last 4

Write-Output "Python / PyTorch"
& $Python -c "import json,sys,torch; print(json.dumps({'python':sys.version.split()[0], 'torch':torch.__version__, 'torch_cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(), 'cuda_available':torch.cuda.is_available(), 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'compute_capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, 'torch_path':torch.__path__[0]}, indent=2))"

$Cache = Join-Path $BuildDirectory "CMakeCache.txt"
if (Test-Path $Cache) {
    Write-Output "Native CMake resolution"
    Select-String -Path $Cache -Pattern "^(Torch_DIR|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_ARCHITECTURES):" |
        ForEach-Object { $_.Line }
}
