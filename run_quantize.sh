input_file=./weights/yolov7-w6.onnx
preprocessed_file=./weights/yolov7-w6_infer.onnx
quantized_file=./weights/yolov7-w6_quant.onnx

calibrate_dataset=/data/coco128/images/train2017/

python -m onnxruntime.quantization.preprocess --input $input_file --output $preprocessed_file

python quantize_onnx.py --input_model $preprocessed_file --output_model $quantized_file --calibrate_dataset $calibrate_dataset
