import pickle, json
import os
kagle_json = '../annotations/dataset_coco.json'

INPUT_SIZE = '500'

new_data_path = f'../post_processed_karpthy_coco_{INPUT_SIZE}'
if not os.path.exists(new_data_path):
    os.makedirs(new_data_path)

new_json_train = f'../post_processed_karpthy_coco_{INPUT_SIZE}/train.json'
new_json_test = f'../post_processed_karpthy_coco_{INPUT_SIZE}/test.json'
new_json_val = f'../post_processed_karpthy_coco_{INPUT_SIZE}/val.json'



def map_format_kaggle_to_clipcap(INPUT_SIZE):
    def extract_imgid_from_name(filename):
        return str(int(filename.split('.')[0].split('_')[-1]))

    with open(kagle_json) as f:
        kaggle_data = json.load(f)
    train_data = []
    test_data = []
    val_data = []
    if INPUT_SIZE == '10k':
        train_count = 10000
        test_count = 2000
        val_count = 2000
    elif INPUT_SIZE == '5k':
        train_count = 5000
        test_count = 1000
        val_count = 1000
    elif INPUT_SIZE == '1k':
        train_count = 1000
        test_count = 200
        val_count = 200
    elif INPUT_SIZE == '500':
        train_count = 500
        test_count = 100
        val_count = 100

    splits = {'train': train_data, 'test': test_data, 'val': val_data, 'restval': train_data}
    out_names = {'train': new_json_train, 'test': new_json_test, 'val': new_json_val}
    for img in kaggle_data['images']:
        flag = False
        if img['split'] == 'train' or img['split'] == 'restval':
            if train_count:
                train_count = train_count - 1
                flag = True
        elif img['split'] == 'test' and test_count:
            test_count = test_count - 1
            flag = True
        elif img['split'] == 'val' and val_count:
            val_count = val_count - 1
            flag = True
        if flag:
            imgid = extract_imgid_from_name(img['filename'])
            for cap in img['sentences']:
                correct_format = {"image_id": int(imgid), "caption": cap['raw'], "id": int(cap['sentid'])}
                splits[img['split']].append(correct_format)

    DBG = False
    if not DBG:
        for name in out_names:
            with open(out_names[name], 'w') as f:
                json.dump(splits[name], f)

        for name in out_names:
            with open(out_names[name][:-5] + '_metrics_format.json', 'w') as f:
                annos = splits[name]
                ids = [{"id": int(a["image_id"])} for a in annos]
                final = {"images": ids, "annotations": annos}
                json.dump(final, f)
    print("Done")

    if DBG:
        # rons annotations
        with open('annotations/train_caption_of_real_training.json') as f:
        # with open('../../train_caption.json') as f:
            cur_data = json.load(f)
        ids = [str(int(c['image_id'])) for c in cur_data]
        new_ids = [str(int(c['image_id'])) for c in train_data]
        ids.sort()  # inplace
        new_ids.sort()
        assert ids == new_ids
        print('OK')


if __name__ == '__main__':
    map_format_kaggle_to_clipcap(INPUT_SIZE)