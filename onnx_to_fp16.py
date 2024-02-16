import onnx
from onnxconverter_common import float16

model = onnx.load("./weights/yolov7-w6.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "./weights/yolov7-w6_fp16.onnx")