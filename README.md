# RDK S100 YOLOv5u: training to deployment

This repository documents and automates a YOLOv5u workflow across three execution environments: a Windows development host for data preparation and source control, a Linux/x86 environment for training, export, and compilation, and an RDK S100 board for BPU inference and performance validation.

See the [operation manual](docs/RDK_S100_YOLOv5u_从训练到部署.md) for the complete procedure.

Git supplies the code, the 128-image project dataset, documentation, and directory structure. Downloadable and generated large artifacts are supplied separately by `wget`, training, and compilation.
