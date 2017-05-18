import json

val_coco = json.load(open('data/coco/captions_val2014.json', 'r'))
train_coco = json.load(open('data/coco/captions_train2014.json', 'r'))


train_anno = json.load(open("data/Annotations/v2_mscoco_train2014_annotations.json", "r"))
train_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_train2014_questions.json", "r"))

val_anno   = json.load(open("data/Annotations/v2_mscoco_val2014_annotations.json", "r"))
val_ques = json.load(open("data/Questions/v2_OpenEnded_mscoco_val2014_questions.json", "r"))

# a_vqa = json.load(open('vqa_val.json', 'r'))
# b_vqa = json.load(open('vqa_train.json', 'r'))


val_split['images'] = images[:5000]
test_split['images'] = images[5000:10000]
train_split['images'] = images[10000:]


images = a['images'] + b['images']
annotations = a_vqa['annotations'] + b_vqa['annotations']

# for efficiency lets group annotations by image
itoa = {}
for a in annotations:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)


train_split['annotations'] = []
for image in train_split['images']:
    image_id = image['id']
    train_split['annotations'].extend(itoa[image_id])

val_split['annotations'] = []
for image in val_split['images']:
    image_id = image['id']
    val_split['annotations'].extend(itoa[image_id])

test_split['annotations'] = []
for image in test_split['images']:
    image_id = image['id']
    test_split['annotations'].extend(itoa[image_id])
