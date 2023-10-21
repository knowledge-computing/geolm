CUDA_VISIBLE_DEVICES='2' python3 multi_link_geonames.py --model_name='joint-base'  --query_dataset_path='../../data/WikToR.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_wiktor/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=900 --spatial_bert_weight_dir='/home/zekun/weights_0614/' --spatial_bert_weight_name='ep3_iter64000_0.3270.pth'  --out_dir='debug'


CUDA_VISIBLE_DEVICES='3' python3 multi_link_geonames.py --model_name='joint-base'  --query_dataset_path='/home/zekun/datasets/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=900 --spatial_bert_weight_dir='/home/zekun/weights_0614/' --spatial_bert_weight_name='ep3_iter64000_0.3270.pth'  --out_dir='debug'

