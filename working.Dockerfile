Working Docker

FROM pytorch/torchserve:0.8.0-gpu as service_gpu

RUN pip install onnx onnxruntime-gpu transformers

USER model-server
