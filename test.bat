mkdir outputs
mkdir outputs\DeepLab_DUC_HDC
mkdir outputs\DeepLabResNet
mkdir outputs\DeepLabXception
mkdir outputs\ENet
mkdir outputs\FCN8
mkdir outputs\GCN
mkdir outputs\PSPDenseNet
mkdir outputs\PSPNet
mkdir outputs\SegNet
mkdir outputs\SegResNet
mkdir outputs\UNet
mkdir outputs\UNetResnet
mkdir outputs\UperNet

python inference.py -c config\DeepLab_DUC_HDC.json -m saved\DeepLab_DUC_HDC\01-08_01-17\checkpoint-epoch100.pth -i data\images\validation -o outputs\DeepLab_DUC_HDC 
python inference.py -c config\DeepLabResNet.json -m saved\DeepLab\01-08_03-52\checkpoint-epoch100.pth -i data\images\validation -o outputs\DeepLabResNet 
python inference.py -c config\DeepLabXception.json -m saved\DeepLab\01-08_04-46\checkpoint-epoch100.pth -i data\images\validation -o outputs\DeepLabXception 
python inference.py -c config\ENet.json -m saved\ENet\01-08_05-50\checkpoint-epoch100.pth -i data\images\validation -o outputs\ENet 
python inference.py -c config\FCN8.json -m saved\FCN8\01-08_06-17\checkpoint-epoch100.pth -i data\images\validation -o outputs\FCN8 
python inference.py -c config\GCN.json -m saved\GCN\01-08_08-16\checkpoint-epoch100.pth -i data\images\validation -o outputs\GCN 
python inference.py -c config\PSPDenseNet.json -m saved\PSPDenseNet\01-08_08-46\checkpoint-epoch100.pth -i data\images\validation -o outputs\PSPDenseNet 
python inference.py -c config\PSPNet.json -m saved\PSPNet\01-08_11-12\checkpoint-epoch100.pth -i data\images\validation -o outputs\PSPNet 
python inference.py -c config\SegNet.json -m saved\SegNet\01-08_12-44\checkpoint-epoch50.pth -i data\images\validation -o outputs\SegNet 
python inference.py -c config\SegResNet.json -m saved\SegResNet\01-08_13-25\checkpoint-epoch100.pth -i data\images\validation -o outputs\SegResNet 
python inference.py -c config\UNet.json -m saved\UNet\01-08_14-11\checkpoint-epoch100.pth -i data\images\validation -o outputs\UNet 
python inference.py -c config\UNetResnet.json -m saved\UNetResnet\01-08_15-31\checkpoint-epoch100.pth -i data\images\validation -o outputs\UNetResnet 
python inference.py -c config\UperNet.json -m saved\UperNet\01-08_16-49\checkpoint-epoch100.pth -i data\images\validation -o outputs\UperNet 