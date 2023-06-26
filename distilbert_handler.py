import os

import onnxruntime
import psutil
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


class LanguageModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model_config = None
        self.config = None
        self.tokenizer = None
        self.model = None

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        print(properties)
        model_path = self.manifest['model']['serializedFile']
        if not os.path.isfile(model_path):
            raise RuntimeError("Missing serialized .onnx file")

        sess_options = onnxruntime.SessionOptions()
        sess_options.optimized_model_filepath = model_path
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = False

        run_options = onnxruntime.RunOptions()
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0;gpu:0")

        gpu_and_cpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.config = {}
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = onnxruntime.InferenceSession(model_path, sess_options, providers=gpu_and_cpu_providers)
        if self.model:
            print("Initialized Model")
        else:
            print("Failed to initialize model")

    def preprocess(self, inference_samples):
        pass

    def inference(self, inputs, *args):
        pass

    def postprocess(self, model_output):
        return [model_output]

    def handle(self, data, context):
        # ONNX Runtime expects NumPy arrays as input
        print("In handler")
        return [[1]]

