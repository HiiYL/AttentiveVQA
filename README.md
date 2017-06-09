# AttentiveVQA

To run, place datafiles in the following directories

    data
    └── Questions
        ├── v2_OpenEnded_mscoco_test2015_questions.json
        ├── v2_OpenEnded_mscoco_train2014_questions.json
        └── ...
    └── Images
        └── mscoco
            ├── merged2014
            └── test2015
    └── Annotations
        ├── v2_mscoco_train2014_annotations.json
        └── v2_mscoco_val2014_annotations.json
        
Usage
1. `python preprocess.py` to preprocess comments
2. `python generate_features_coco.py` to generate image features from the mscoco dataset
3. `python train_multimodal.py`
            
