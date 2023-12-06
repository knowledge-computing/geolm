# GeoLM: Empowering Language Models for Geospatially Grounded Language Understanding

[[Paper link](https://arxiv.org/pdf/2310.14478.pdf)] [[Toponym Detection Demo](https://huggingface.co/zekun-li/geolm-base-toponym-recognition)] [[CodeForHuggingFace](https://github.com/zekun-li/transformers/tree/geolm)]

## Install 

1. Clone this repository:

```Shell
git clone git@github.com:knowledge-computing/geolm.git
cd geolm 
```

2. Install packages
```Shell
conda create -n geolm_env python=3.8 -y
conda activate geolm_env
pip install --upgrade pip
pip install -r requirements.txt 
```

## Pre-Train 
1. Change directory to the pre-training script folder
```
cd src
```
2. Run `train_joint.py`
```
python3 train_joint.py --model_save_dir=OUTPUT_WEIGHT_DIR --pseudo_sentence_dir='../../datasets/osm_pseudo_sent/world/' --nl_sentence_dir='../../datasets/wikidata/world_georelation/joint_v2/' --batch_size=28   --lr=1e-5 --spatial_dist_fill=900 --placename_to_osmid_path='../../datasets/osm_pseudo_sent/name-osmid-dict/placename_to_osmid.json' 
```

## Downstream Tasks

### Toponym Detection 
1. Train with in-domain dataset
```
cd experiments/toponym_detection/

python3 train_geobert_toponym.py --model_save_dir=OUTPUT_TOPONYM_WEIGHT_DIR --model_option='geobert-base' --model_checkpoint_path=PRETRAINED_MODEL_WEIGHT --lr=1e-5 --epochs=30  --input_file_path=DATASET_PATH
```

2. Test with in-domain dataset
```
cd experiments/toponym_detection/

python3 test_geobert_toponym.py  --model_option='geobert-base' --model_save_path=TOPONYM_MODEL_PATH --input_file_path=DATASET_PATH --spatial_dist_fill=90000

```

### Toponym Linking



### Geo-entity Typing 



## Cite 
```
@article{li2023geolm,
  title={GeoLM: Empowering Language Models for Geospatially Grounded Language Understanding},
  author={Li, Zekun and Zhou, Wenxuan and Chiang, Yao-Yi and Chen, Muhao},
  journal={arXiv preprint arXiv:2310.14478},
  year={2023}
}
```


