# Based On Personal Computer

## Requirements
- An efficient GPU
- Colab
- Ubuntu Operating System
- Ubuntu account

## Steps

1. Training Yolov5 model
 please go below OwO

     [ultralytics
/
yolov5](https://github.com/ultralytics/yolov5/tree/master)
2. requirement download

    ```bash
    pip install -r requirements.txt    
    ```
3. Train

    ```bash
    python train.py  --batch 15 --epochs 150  --data fruit.yaml --weights yolov5s.pt --cache     
    ```
4. Test

    ```bash
    python detect.py --weight runs/train/exp11/weights/best.pt  --source <path/to/image>
    ```
5. [Run  fruits_calculator o(*￣▽￣*)ブ](fruits_calculator.ipynb)

    記得路徑要改一下

## Reference
[ultralytics
yolov5](https://github.com/ultralytics/yolov5/tree/master)

[PyTorch 自行訓練YOLOv5 物件偵測模型教學與範例](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj065KMq7WDAxVysVYBHSLWBTAQFnoECBAQAQ&url=https%3A%2F%2Fofficeguide.cc%2Fpytorch-yolo-v5-object-egg-detection-models-tutorial-examples%2F&usg=AOvVaw1Aix8WFZvmZ6SdkTJ-Jj79&opi=89978449)

[https://universe.roboflow.com/tan-loc-qsttm/lv3/dataset/1](https://universe.roboflow.com/tan-loc-qsttm/lv3/dataset/1)
