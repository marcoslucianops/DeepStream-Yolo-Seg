# DeepStream-Yolo-Seg

NVIDIA DeepStream SDK 8.0 / 7.1 / 7.0 / 6.4 / 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 application for YOLO-Seg models

--------------------------------------------------------------------------------------------------
### YOLO object detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------
### Important: Please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------

### Getting started

* [Supported models](#supported-models)
* [Instructions](#basic-usage)
* [YOLOv5-Seg usage](docs/YOLOv5_Seg.md)
* [YOLOv7-Seg usage](docs/YOLOv7_Seg.md)
* [YOLOv7-Mask usage](docs/YOLOv7_Mask.md)
* [YOLOv8-Seg usage](docs/YOLOv8_Seg.md)
* [YOLO11-Seg usage](docs/YOLO11_Seg.md)
* [YOLO26-Seg usage](docs/YOLO26_Seg.md)
* [RF-DETR-Seg usage](docs/RFDETR_Seg.md)
* [NMS configuration](#nms-configuration)
* [Detection threshold configuration](#detection-threshold-configuration)

##

### Supported models

* [RF-DETR-Seg](https://github.com/roboflow/rf-detr)
* [YOLO26-Seg](https://github.com/ultralytics/ultralytics)
* [YOLO11-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv8-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv7-Mask](https://github.com/WongKinYiu/yolov7/tree/mask)
* [YOLOv7-Seg](https://github.com/WongKinYiu/yolov7/tree/u7/seg)
* [YOLOv5-Seg](https://github.com/ultralytics/yolov5)

##

### Instructions

#### 1. Download the DeepStream-Yolo-Seg repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git
cd DeepStream-Yolo-Seg
```

#### 2. Compile the libs

2.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 = 10.2
  ```

2.2. Make the libs

```
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg
```

#### 3. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

##

### NMS configuration

For now, the NMS is configured in the ONNX exporter file.

**NOTE**: Make sure to set `cluster-mode=4` in the config_infer file.

##

### Detection threshold configuration

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
[class-attrs-all]
pre-cluster-threshold=0.25
```

##

My projects: https://www.youtube.com/MarcosLucianoTV
