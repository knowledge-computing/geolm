# lgl
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/lgl/baseline' --backbone_option='roberta-base' --lr=1e-5 
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/lgl/baseline' --backbone_option='spanbert-base' --lr=1e-5 
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/lgl/baseline' --backbone_option='simcse-bert-base' --lr=1e-5 
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/lgl/baseline' --backbone_option='simcse-roberta-base' --lr=1e-5 

# geowebnews
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/geowebnews/baseline' --backbone_option='bert-base' --lr=1e-5 
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/geowebnews/baseline' --backbone_option='roberta-base' --lr=1e-5 --input_file_path='/home/zekun/toponym_detection/geowebnews/GWN.json'
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/geowebnews/baseline' --backbone_option='spanbert-base' --lr=1e-5  --input_file_path='/home/zekun/toponym_detection/geowebnews/GWN.json'
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/geowebnews/baseline' --backbone_option='simcse-bert-base' --lr=1e-5 --input_file_path='/home/zekun/toponym_detection/geowebnews/GWN.json'
# CUDA_VISIBLE_DEVICES='3' python3 train_baseline_toponym.py --model_save_dir='/home/zekun/weights_toponym/geowebnews/baseline' --backbone_option='simcse-roberta-base' --lr=1e-5 --input_file_path='/home/zekun/toponym_detection/geowebnews/GWN.json'