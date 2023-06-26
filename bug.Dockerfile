FROM pytorch/torchserve:latest-gpu as service_gpu

RUN pip install onnx onnxruntime-gpu transformers

USER model-server
