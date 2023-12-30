# Based On KV260
## Requirements
- An efficient GPU
- Colab
- Ubuntu Operating System
- Ubuntu account
- Vitis ai 2.5 environment
- KV260 Board

## Models Training
1. 激活函數修改

    由于yolov5的6.0版本激活函数已经被是SiLU函数了，而该DPU是不支持该激活函数的，在Vitis-AI的定制OP功能中应该可以实现SiLU函数，但是我还没有摸索清楚，所以这里将模型中的SiLU激活函数替换回了老版本yolov5模型的LeakyReLU函数。具体需要修改的文件为common.py和experimental.py文件，作如下修改。我一共修改了3处激活函数，解决了在量化时因为SiLU激活函数报错的问题。

    ```bash
    # 修改前
    self.act = nn.SiLU
    # 修改后
    self.act = nn.LeakyReLU(0.1, inplace=True)    
    ```
    in common.py
    ```bash
    class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
        default_act = nn.LeakyReLU(0.1, inplace=True)  # THIS

    ---------------------分隔線----------------------

    class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion<mark style="background-color: lightblue">Marked text</mark>
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True) #THIS
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    ```
    in experimental.py
    ```bash
    class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) #THIS

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
    ```

2. Training Yolov5 model
 
    please go below OwO

     [ultralytics
/
yolov5](https://github.com/ultralytics/yolov5/tree/master)
3. requirement download

    ```bash
    pip install -r requirements.txt    
    ```
4. Train

    ```bash
    python train.py  --batch 15 --epochs 150  --data fruit.yaml --weights yolov5s.pt --cache     
    ```
5. Test

    ```bash
    python detect.py --weight runs/train/exp11/weights/best.pt  --source <path/to/image>
    ```
## Quantize your model
1. Install Vitis AI Environment

   [GO HERE (┬┬﹏┬┬)](https://github.com/Xilinx/Vitis-AI)

2. Pytorch 

    在Vitis AI環境內clone yolov5 並執行 pip install -r requirement.txt
   
    需要先將pytorch重新安裝
    
    ```bash
    pip uninstall torch torchvision    
    ```
    
    ```bash
    pip install torch==1.10.1
    pip install torchvision==0.11.2
    pip install ultralytics==8.0.117
    ```
4. 修改yolo.py

    这里就需要从代码层面来分析yolov5模型的特征提取过程，整个特征提取过程都是直接使用pytorch的torch张量的相关算子对数据进行处理的，但是在检测层，有一段对最终的三层特征进行处理的代码没有使用torch张量的相关算子，所以在对模型做量化时，需要注释掉这一段代码，并将其添加在检测函数中。该代码位于yolo.py文件的Detect类中，如下所示：
    ### 修改前
    ```bash
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)

        return x if self.training else (torch.cat(z, 1), x)    
    ```
    ### 修改後
    ```bash
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x
    ```
5. 將訓練好的模型進行量化

    [GO HERE (┬┬﹏┬┬)](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/09-mnist_pyt/files/quantize.py)
    ```bash
    import os
    import sys
    import argparse
    import random
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel
    from common import *

    from models.common import DetectMultiBackend
    from models.yolo import Model

    DIVIDER = '-----------------------------------------'

    def quantize(build_dir,quant_mode,batchsize):

    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
        print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    # load trained model
    model = DetectMultiBackend("./v5n_ReLU_best.pt", device=device)

    # force to merge BN with CONV for better quantization accuracy
    optimize = 1

    # override batchsize if in test mode
    if (quant_mode=='test'):
        batchsize = 1
    
    rand_in = torch.randn([batchsize, 3, 960, 960])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model

    # create a Data Loader
    test_dataset = CustomDataset('../../train/JPEGImages',transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batchsize, 
                                                shuffle=False)

    t_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1 if quant_mode == 'test' else 10, 
                                                shuffle=False)

    # evaluate 
    test(quantized_model, device, t_loader)

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
    
    return

    def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
    ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b',  '--batchsize',  type=int, default=50,        help='Testing batchsize - must be an integer. Default is 100')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--quant_mode   : ',args.quant_mode)
    print ('--batchsize    : ',args.batchsize)
    print(DIVIDER)

    quantize(args.build_dir,args.quant_mode,args.batchsize)

    return

    if __name__ == '__main__':
        run_main()
    ```
6. Model Compilation

    执行python文件来生成量化配置
    ```bash
    python quantize.py -q calib    
    ```
    运行test来生成xmodel
    ```bash
    python quantize.py -q test -b 1
    ```
    使用xilinx提供的compiler去把这个xmodel编译成DPU支持的，基于XIR的xmodel，运行如下指令：
    ```bash
    vai_c_xir -x ./build/quant_model/DetectMultiBackend_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o ./ -n my_model
    ```
## Deploy
1. 部署前准备
    
    我们需要一个DPU Design 的 Hardware，可以用Vivado手动Block Design搭一个，不过这会牵扯到很多麻烦的地址设置，我会另写文章单独讲，在这里我们简单用一下Xilinx搭建的标准DPU Hardware，在DPU-PYNQ仓库的boards文件夹下就有。根据README构建Design，需要安装xrt和Vitis。官方的脚本写的比较死板，只认2022.1的版本，可以编辑check_env.sh绕过检查

    ```bash
    cd DPU-PYNQ/boards
    source <vitis-install-path>/Vitis/2022.2/settings64.sh
    source <xrt-install-path>/xilinx/xrt/setup.sh
    make BOARD=kv260_som    
    ```

    脚本运行完后，会生成三个文件

    dpu.bit
    
    dpu.hwh
    
    dpu.xclbin
    
    再加上之前生成的

    my_model.xmodel
    
    需要的文件都准备完成，接下来可以在pynq上进行部署。
2. 安裝DPU-PYNQ

    ```bash
    pip install pynq-dpu --no-build-isolation
    cd $PYNQ_JUPYTER_NOTEBOOKS
    pynq get-notebooks pynq-dpu -p
    ```
3. Deploy YOLOV5 (≧∇≦)ﾉ

    沒什麼特別的，直接套用[dpu_yolov5_fruit_calculator.ipynb](HW\dpu_yolov5_fruit_calculator.ipynb)即可，記得檔案路徑要改一下
    

# Reference
[DPU_yolov3](https://github.com/Xilinx/DPU-PYNQ/blob/master/pynq_dpu/notebooks/dpu_yolov3.ipynb)

[https://blog.csdn.net/qq_36745999/article/details/126981630](https://blog.csdn.net/qq_36745999/article/details/126981630)

[https://lgyserver.top/index.php/2023/05/08/xilinx-vitis-ai%E9%87%8F%E5%8C%96%E9%83%A8%E7%BD%B2yolov5%E8%87%B3dpu-pynq/](https://lgyserver.top/index.php/2023/05/08/xilinx-vitis-ai%E9%87%8F%E5%8C%96%E9%83%A8%E7%BD%B2yolov5%E8%87%B3dpu-pynq/)

[https://xilinx.eetrend.com/blog/2022/100565582.html](https://xilinx.eetrend.com/blog/2022/100565582.html)

[https://blog.csdn.net/m0_45287781/article/details/127947918](https://blog.csdn.net/m0_45287781/article/details/127947918)

[https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/09-mnist_pyt/files/quantize.py](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/09-mnist_pyt/files/quantize.py)

