export CUDA_VISIBLE_DEVICES="1"
export PROJECT_HOME='/home/xuchengjun/ZXin/smap'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python config.py -log
python -m torch.distributed.launch --nproc_per_node=1 train.py