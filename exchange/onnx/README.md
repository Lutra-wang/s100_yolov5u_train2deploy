# ONNX exchange artifact

Purpose: stores the exported interchange model. Producing stage: run the ONNX export command after training. Expected filename: `yolov5nu_coco2017_val128_s100.onnx`. Success signal: the export completes and this ONNX file exists. The artifact is absent from Git because it is a generated model artifact.
