# YOLOv7-Seg usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yoloV7_seg file](#edit-the-config_infer_primary_yolov7_seg-file)

##

### Convert model

#### 1. Download the YOLOv7 repo and install the requirements

```
git clone -b u7 https://github.com/WongKinYiu/yolov7
cd yolov7/seg
pip3 install -r requirements.txt
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloV7_seg.py` file from `DeepStream-Yolo-Seg/utils` directory to the `yolov7/seg` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) releases (example for YOLOv7-Seg)

```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt
```

**NOTE**: You can use your custom model.

#### 4. Reparameterize your model (for custom models)

Custom YOLOv7 models cannot be directly converted to engine file. Therefore, you will have to reparameterize your model using the code [here](https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb). Make sure to convert your custom checkpoints in YOLOv7 repository, and then save your reparmeterized checkpoints for conversion in the next step.

#### 5. Convert model

Generate the ONNX model file (example for YOLOv7-Seg)

```
python3 export_yoloV7_seg.py -w yolov7-seg.pt --dynamic
```

**NOTE**: Confidence threshold (example for conf-thres = 0.25)

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
--conf-thres 0.25
```

**NOTE**: NMS IoU threshold (example for iou-thres = 0.45)

```
--iou-thres 0.45
```

**NOTE**: Maximum detections (example for max-det = 100)

```
--max-det 100
```

**NOTE**: To convert a P6 model

```
--p6
```

**NOTE**: To change the inference size (defaut: 640 / 1280 for `--p6` models)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 6. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo-Seg` folder.

##

### Edit the config_infer_primary_yoloV7_seg file

Edit the `config_infer_primary_yoloV7_seg.txt` file according to your model (example for YOLOv7-Seg)

```
[property]
...
onnx-file=yolov7-seg.onnx
model-engine-file=yolov7-seg.onnx_b1_gpu0_fp32.engine
...
```

**NOTE**: To output the masks, use

```
[property]
...
output-instance-mask=1
segmentation-threshold=0.5
...
```

**NOTE**: The **YOLOv7-Seg** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```
