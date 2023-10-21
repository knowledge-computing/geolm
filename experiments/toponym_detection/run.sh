

# CUDA_VISIBLE_DEVICES='2' python3 train_geobert_toponym.py --model_save_dir='/home/zekun/weights_toponym/lgl/geobert-0511' --model_option='geobert-base' --model_checkpoint_path='/home/zekun/weights_base_run2/ep14_iter96000_0.0382.pth' --lr=1e-5 --epochs=30


# for geowebnews 
# CUDA_VISIBLE_DEVICES='3' python3 train_geobert_toponym.py --model_save_dir='/home/zekun/weights_toponym/geowebnews/geobert-0511' --model_option='geobert-base' --model_checkpoint_path='/home/zekun/weights_base_run2/ep14_iter96000_0.0382.pth' --lr=1e-5 --epochs=30  --input_file_path='/home/zekun/toponym_detection/geowebnews/GWN.json'

