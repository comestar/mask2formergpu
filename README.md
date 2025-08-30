conda create -n mask2 python=3.8初始化

activate mask2激活

pip install torch==1.8.0 torchvision==0.9.0库 或

从文件下载torch-1.12.0+cu113-cp38-cp38-linux_x86_64

这个系列版本的东西

pip install pycocotools库

pip install opencv-python库

cd miniconda\envs\mask2(定位到虚拟环境的具体为位置)

git clone https://github.com/facebookresearch/detectron2.git detectron2源码库

pip install -e .  （.表示当前目录）/pip install -e detectron2

pip list检测