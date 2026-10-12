# YOLO26-Seg usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolo26_seg file](#edit-the-config_infer_primary_yolo26_seg-file)
* [Create the TensortRT Engine](#create-the-tensorrt-engine)

##

### Convert model

#### 1. Download the YOLO26 repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime onnxscript
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolo26_seg.py` file from `DeepStream-Yolo-Seg/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLO26](https://github.com/ultralytics/assets/releases/) releases (example for YOLO26s-Seg)

```
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO26s-Seg)

```
python3 export_yolo26_seg.py -w yolo26s-seg.pt
```

**NOTE**: Minimum detection confidence threshold (example for conf-threshold = 0.25)

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
--conf-threshold 0.25
```

**NOTE**: NMS IoU threshold (example for iou-threshold = 0.45)

```
--iou-threshold 0.45
```

**NOTE**: Maximum number of output detections (example for max-detections = 300)

```
--max-detections 300
```

**NOTE**: To change the inference size (defaut: 640)

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

#### 5. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo-Seg` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo-Seg` folder and compile the lib

2. Set the `CUDA_VER` according to your DeepStream version

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

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg
```

##

### Edit the config_infer_primary_yolo26_seg file

Edit the `config_infer_primary_yolo26_seg.txt` file according to your model (example for YOLO26s-Seg)

```
[property]
...
onnx-file=yolo26s-seg.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYoloSeg
...
infer-dims=channels;height;width
```

**NOTE**: To output the masks, use

```
[property]
...
output-instance-mask=1
segmentation-threshold=0.5
...
```

**NOTE**: The **YOLO26-Seg** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```

##

### Create the TensorRT Engine

If you exported the ONNX model on a different device, copy the following files to your Jetson device before generating the TensorRT engine:

- `yolo26s-seg.onnx`
- `yolo26s-seg.onnx.data` (if generated during export)
- `labels.txt` (if generated during export)

Make sure the ONNX model and its associated data file are in the same directory.

Next, run the following command on your Jetson device, replacing the placeholder paths with the actual paths on your system:

```
trtexec \
  --onnx=<path/to/yolo26s-seg.onnx> \
  --saveEngine=<path/to/yolo26s-seg.onnx_b1_gpu0_fp32.engine> \
  --plugins=<path/to/nvdsinfer_custom_impl_Yolo_seg/libnvdsinfer_custom_impl_Yolo_seg.so>
```

TensorRT will build the engine for your Jetson device. **The build process may take several minutes**, depending on the model, hardware, and TensorRT configuration.

Once the engine has been generated, update `config_infer_primary_yolo26_seg.txt` to reference the generated engine if required by your configuration:

```
[property]
...
model-engine-file=<path/to/yolo26s-seg.onnx_b1_gpu0_fp32.engine>
...
```

**Note:** TensorRT engines are generally hardware- and software-version-specific. Generate the engine on the target Jetson device to ensure compatibility.

A couple of technical details worth noting:

- The `--plugins` option should point to the compiled custom parser/plugin library. Ensure that the library has been built successfully before running `trtexec`.
- If your ONNX export produced an external data file, keep `yolo26s-seg.onnx.data` alongside the ONNX model so TensorRT can access the model weights.

##

### NOTEs

**Tested environment**

The following software versions were used to test the YOLO26-Seg export and DeepStream integration:

| Component         | Version |
|-------------------|---------|
| PyTorch           | 2.14.1  |
| Ultralytics       | 8.4.175 |
| ONNX              | 1.23.2  |
| TensorRT          | 8.6.2.3 |
| JetPack           | 6.0     |
| NVIDIA DeepStream | 7.0     |

**Note:** Compatibility may vary depending on the operating system, GPU architecture, CUDA version, and TensorRT plugins. TensorRT engines should be generated on the target device.