echo 'LGL weights base run2 '
CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep14_iter04000_0.0183.pth'  --out_dir='debug'


echo 'LGL weights base 0511, ep4_iter04000_1.1645.pth'

CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0511/' --spatial_bert_weight_name='ep4_iter04000_1.1645.pth'  --out_dir='debug'

echo 'LGL weights base 0511, ep5_iter04000_1.1184.pt'

CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0511/' --spatial_bert_weight_name='ep5_iter04000_1.1184.pt'  --out_dir='debug'


echo 'LGL weights base 0511, ep6_iter04000_1.0868.pth'

CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0511/' --spatial_bert_weight_name='ep6_iter04000_1.0868.pth'  --out_dir='debug'


echo 'LGL weights base 0511, ep7_iter04000_0.9850.pth'

CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0511/' --spatial_bert_weight_name='ep7_iter04000_0.9850.pth'  --out_dir='debug'


echo 'LGL weights base 0511, ep8_iter04000_0.9723.pth'

CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0511/' --spatial_bert_weight_name='ep8_iter04000_0.9723.pth'  --out_dir='debug'


# echo "joint-base-geowebnews"
# python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/geowebnews/GWN.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_geowebnews/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep14_iter88000_0.0039.pth'

# echo "bert-base-geowebnews"
# python3 link_geonames.py --model_name='bert-base' --query_dataset_path='/home/zekun/toponym_detection/geowebnews/GWN.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_geowebnews/geoname-ids.json'

# echo "bert-large-geowebnews"
# python3 link_geonames.py --model_name='bert-large' --query_dataset_path='/home/zekun/toponym_detection/geowebnews/GWN.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_geowebnews/geoname-ids.json'


# echo "joint-base-lgl"
# CUDA_VISIBLE_DEVICES='1' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep14_iter88000_0.0039.pth' 

# echo "bert-base-lgl"
# CUDA_VISIBLE_DEVICES='1' python3 link_geonames.py --model_name='bert-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json'

# echo "bert-large-lgl"
# CUDA_VISIBLE_DEVICES='1' python3 link_geonames.py --model_name='bert-large' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json'

