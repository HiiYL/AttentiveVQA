import json

a = json.load(open('data/coco/captions_val2014.json', 'r'))
b = json.load(open('data/coco/captions_train2014.json', 'r'))

images = a['images'] + b['images']
annotations = a['annotations'] + b['annotations']

train_split = {}
val_split = {}
test_split = {}
val_split['images'] = images[:5000]
test_split['images'] = images[5000:10000]
train_split['images'] = images[10000:]



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


json.dump(train_split, open('data/coco/captions_train2014_karpathy_split.json', 'w'))
json.dump(val_split, open('data/coco/captions_val2014_karpathy_split.json', 'w'))
json.dump(test_split, open('data/coco/captions_test2014_karpathy_split.json', 'w'))