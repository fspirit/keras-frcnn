import pandas as pd
# Select 10000 Day imgs
# Train 5 fold CV

images_meta = pd.read_csv('./train_data/train.csv')
bboxes = pd.read_csv('./train_data/train_boxes.csv')

# print bboxes.head()

day_only = images_meta[images_meta.lighting == 'Day']
day_only_random_15K = day_only.sample(15000)

day_only_random_15K['image_filename'] = day_only_random_15K['image_filename'].apply(lambda f: '/train_data/img/' + f)
train = day_only_random_15K[:10000]
test = day_only_random_15K[10000:]

# print day_only_random_10K.head()

train = pd.merge(train, bboxes, on='image_filename')
train = train[['image_filename', 'x0', 'y0', 'x1', 'y1']]
train['class_name'] = 'car'
train = train.round({'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0})

# print result.head()

train.to_csv('./train_data/frcnn_train.csv', index=False, header=False)
test.to_csv('./train_data/frcnn_test.csv', index=False, header=False, columns=['image_filename'])