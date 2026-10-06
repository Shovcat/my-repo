import sys
import numpy, pandas, sklearn, scipy, matplotlib, seaborn
import torch, torchvision, torchinfo, thop, onnx, onnxruntime
import mlflow, psutil, codecarbon, fastapi, uvicorn, pytest
import httpx, locust, requests, pyarrow, joblib, tqdm

packages = [
    ("Python", sys.version.split()[0]),
    ("numpy", numpy.__version__),
    ("pandas", pandas.__version__),
    ("scikit-learn", sklearn.__version__),
    ("scipy", scipy.__version__),
    ("matplotlib", matplotlib.__version__),
    ("seaborn", seaborn.__version__),
    ("torch", torch.__version__),
    ("torchvision", torchvision.__version__),
    ("torchinfo", torchinfo.__version__),
    ("onnx", onnx.__version__),
    ("onnxruntime", onnxruntime.__version__),
    ("mlflow", mlflow.__version__),
    ("psutil", psutil.__version__),
    ("codecarbon", codecarbon.__version__),
    ("fastapi", fastapi.__version__),
    ("uvicorn", uvicorn.__version__),
    ("pytest", pytest.__version__),
    ("httpx", httpx.__version__),
    ("locust", locust.__version__),
    ("requests", requests.__version__),
    ("pyarrow", pyarrow.__version__),
    ("joblib", joblib.__version__),
    ("tqdm", tqdm.__version__),
]

with open("lab01/results/versions.txt", "w") as f:
    for name, ver in packages:
        line = f"{name}=={ver}\n"
        print(line, end="")
        f.write(line)
