# Third-party notices

## Platrix (plate models and plate-reading steps)

`src/plate.js` follows the plate detection and reading steps of Platrix
(https://github.com/AliAkrami1375/Platrix). The ONNX models in `models/` come from
https://huggingface.co/Dibachain/Platrix.

```
MIT License

Copyright (c) 2026 AliAkrami1375

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Licence caution: the plate detector is a YOLOv8 model trained with Ultralytics tooling;
Ultralytics claims AGPL also covers trained weights. That is legally unsettled.

## DINOv2-small (`models/dinov2_small_int8.onnx`)

Apache License 2.0, Meta Platforms. https://huggingface.co/facebook/dinov2-small
Converted to ONNX and quantized to int8 by `tools/export_dinov2.py`.

## ONNX Runtime Web 1.30.0 (`lib/ort.wasm.min.js`, `lib/ort-wasm-simd-threaded.*`)

MIT License, Copyright (c) Microsoft Corporation. https://github.com/microsoft/onnxruntime

## OpenCV.js 5.0.0 (`lib/opencv.js`, npm package @techstark/opencv-js)

Apache License 2.0, https://github.com/opencv/opencv

## Vazirmatn font (`Vazirmatn-Variable.woff2`)

SIL Open Font License 1.1, The Vazirmatn Project Authors. Full text in `Vazirmatn-OFL.txt`.

## IranPlate-Vision (province, city and plate type table)

`src/plate_regions.json` is `plate_data.json` from IranPlate-Vision
(https://github.com/saeed205/IranPlate-Vision). Only the data table is used, not its code or models.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
