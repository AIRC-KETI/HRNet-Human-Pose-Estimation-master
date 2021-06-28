from pycocotools.coco import COCO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
...

print(os.getcwd())
train_annot_path = '../data/coco/annotations/person_keypoints_train2017.json'
val_annot_path = '../data/coco/annotations/person_keypoints_val2017.json'
train_coco = COCO(train_annot_path) # load annotations for training set
val_coco = COCO(val_annot_path) # load annotations for validation set
...
# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # basic parameters of an image
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        # retrieve metadata for all persons in the current image
        anns = coco.loadAnns(ann_ids)

        yield [img_id, img_file_name, w, h, anns]

...

# iterate over images
for img_id, img_fname, w, h, meta in get_meta(train_coco):
    ...
    # iterate over all annotations of an image
    for m in meta:
        # m is a dictionary
        keypoints = m['keypoints']
        ...
...
def convert_to_df(coco):
    images_data = []
    persons_data = []
    # iterate over all images
    for img_id, img_fname, w, h, meta in get_meta(coco):
        images_data.append({
            'image_id': int(img_id),
            'path': img_fname,
            'width': int(w),
            'height': int(h)
        })
        # iterate over all metadata
        for m in meta:
            persons_data.append({
                'image_id': m['image_id'],
                'is_crowd': m['iscrowd'],
                'bbox': m['bbox'],
                'area': m['area'],
                'num_keypoints': m['num_keypoints'],
                'keypoints': m['keypoints'],
            })
    # create dataframe with image paths
    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)
    # create dataframe with persons
    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('image_id', inplace=True)
    return images_df, persons_df

images_df, persons_df = convert_to_df(train_coco)
train_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
train_coco_df['source'] = 0

images_df, persons_df = convert_to_df(val_coco)
val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
val_coco_df['source'] = 1

coco_df = pd.concat([train_coco_df, val_coco_df], ignore_index=True)

# count number of annotations per image

annotated_persons_df = coco_df[coco_df['is_crowd'] == 0]
crowd_df = coco_df[coco_df['is_crowd'] == 1]

print("Number of people in total: " + str(len(annotated_persons_df)))
print("Number of crowd annotations: " + str(len(crowd_df)))

persons_in_img_df = pd.DataFrame({
    'cnt': annotated_persons_df['path'].value_counts()
})
persons_in_img_df.reset_index(level=0, inplace=True)
persons_in_img_df.rename(columns = {'index':'path'}, inplace = True)

# group by cnt so we will get the dataframe with number of annotated people in a single image

persons_in_img_df = persons_in_img_df.groupby(['cnt']).count()

# extract the arrays

x_occurences = persons_in_img_df.index.values
y_images = persons_in_img_df['path'].values

# plot

plt.bar(x_occurences, y_images)
plt.title('People on a single image ')
plt.xticks(x_occurences, x_occurences)
plt.xlabel('Number of people in a single image')
plt.ylabel('Number of images')
plt.show()

annotated_persons_nokp_df = coco_df[(coco_df['is_crowd'] == 0) & (coco_df['num_keypoints'] == 0)]
annotated_persons_kp_df = coco_df[(coco_df['is_crowd'] == 0) & (coco_df['num_keypoints'] > 0)]

print("Number of people (with keypoints) in total: " +
        str(len(annotated_persons_kp_df)))
print("Number of people without any keypoints in total: " +
        str(len(annotated_persons_nokp_df)))

persons_in_img_kp_df = pd.DataFrame({
    'cnt': annotated_persons_kp_df[['path','source']].value_counts()
})
persons_in_img_kp_df.reset_index(level=[0,1], inplace=True)
persons_in_img_cnt_df = persons_in_img_kp_df.groupby(['cnt']).count()
x_occurences_kp = persons_in_img_cnt_df.index.values
y_images_kp = persons_in_img_cnt_df['path'].values

f = plt.figure(figsize=(14, 8))
width = 0.4
plt.bar(x_occurences_kp, y_images_kp, width=width, label='with keypoints')
plt.bar(x_occurences + width, y_images, width=width, label='no keypoints')

plt.title('People on a single image ')
plt.xticks(x_occurences + width/2, x_occurences)
plt.xlabel('Number of people in a single image')
plt.ylabel('Number of images')
plt.legend(loc = 'best')
plt.show()

from sklearn.base import BaseEstimator, TransformerMixin


class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, num_keypoints, w_ix, h_ix, bbox_ix, kp_ix):
        """
        :param num_keypoints: number of keypoints
        :param w_ix: index of a column containing image width
        :param h_ix: index of a column containing image height
        :param bbox_ix: index of a column containing bounding box data
        :param kp_ix: index of a column containing keypoints data
        """
        self.num_keypoints = num_keypoints
        self.w_ix = w_ix
        self.h_ix = h_ix
        self.bbox_ix = bbox_ix
        self.kp_ix = kp_ix

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # retrieve specific columns
        # print(self.w_ix, self.h_ix)
        # print(X[:, self.w_ix], X[:, self.h_ix])
        w = X[:, self.w_ix]
        h = X[:, self.h_ix]
        bbox = np.array(X[:, self.bbox_ix].tolist())  # to matrix
        keypoints = np.array(X[:, self.kp_ix].tolist())  # to matrix

        # calculate scale factors for bounding boxes

        scale_x = bbox[:, 2] / w
        scale_y = bbox[:, 3] / h
        aspect_ratio = w / h

        # calculate scale categories

        scale_cat = pd.cut(bbox[:,3],
                           bins=[0., 32., 64., 96., 128., 160., 192., 224., 256., np.inf],
                           labels=['<032', '<064', '<096', '<128', '<160', '<192', '<224', '<256', 'etc'])
        '''
        conclusion: XXXL, XXL major
        dataset preprocessing: balanced? or 
        '''
        '''
        scale_cat = pd.cut(bbox[:,3] * bbox[:,2],
                           bins=[0., 32.*32., 64.*64., 96.*96., 128.*128., 256.*256.,np.inf],
                           labels=['S', 'M', 'L', 'XL', 'XXL', 'XXXL'])
        '''

        return np.c_[X, scale_x, scale_y, scale_cat, aspect_ratio, keypoints]


# transformer object that is used to add new columns

attr_adder = AttributesAdder(num_keypoints=17, w_ix=1, h_ix=2, bbox_ix=4, kp_ix=7)
# print(coco_df.values[0])
coco_extra_attribs = attr_adder.transform(coco_df.values)

# create new columns list

keypoints_cols = [['x' + str(idx), 'y' + str(idx), 'v' + str(idx)]
                  for idx, k in enumerate(range(attr_adder.num_keypoints))]
keypoints_cols = np.concatenate(keypoints_cols).tolist()

# crate a new richer dataframe

coco_extra_attribs_df = pd.DataFrame(
    coco_extra_attribs,
    columns=list(coco_df.columns) +
            ["scale_x", "scale_y", "scale_cat", "aspect_ratio"] +
            keypoints_cols,
    index=coco_df.index)

# only horizontal images to normalize keypoints coordinates

horiz_imgs_df = coco_extra_attribs_df[coco_extra_attribs_df['aspect_ratio'] >= 1.]

# get the mean width and height - used to scale keypoint coordinates

avg_w = int(horiz_imgs_df['width'].mean())
avg_h = int(horiz_imgs_df['height'].mean())


class NoseAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, avg_w, avg_h, w_ix, h_ix, x1_ix, y1_ix, v1_ix):
        self.avg_w = avg_w
        self.avg_h = avg_h
        self.w_ix = w_ix
        self.h_ix = h_ix
        self.x1_ix = x1_ix
        self.y1_ix = y1_ix
        self.v1_ix = v1_ix

        def fit(self, X, y=None):
            return self  # nothing else to do

    def transform(self, X):
        w = X[:, self.w_ix]
        h = X[:, self.h_ix]
        x1 = X[:, self.x1_ix]
        y1 = X[:, self.y1_ix]

        # normalize nose coordinates to provided average width and height

        scale_x = self.avg_w / w
        scale_y = self.avg_h / h
        nose_x = x1 * scale_x
        nose_y = y1 * scale_y

        return np.c_[X, nose_x, nose_y]

    # transformer object for normalized nose coordinates columns


w_ix = horiz_imgs_df.columns.get_loc('width')
h_ix = horiz_imgs_df.columns.get_loc('height')
x1_ix = horiz_imgs_df.columns.get_loc('x0')  # x coord of a nose is in the column 'x0'
y1_ix = horiz_imgs_df.columns.get_loc('y0')  # y coord of a nose is in the column 'y0'
v1_ix = horiz_imgs_df.columns.get_loc('v0')  # visibility of a nose

attr_adder = NoseAttributesAdder(avg_w, avg_h, w_ix, h_ix, x1_ix, y1_ix, v1_ix)
coco_noses = attr_adder.transform(horiz_imgs_df.values)

# crate dataframe with new normalized coordinates

coco_noses_df = pd.DataFrame(
    coco_noses,
    columns=list(horiz_imgs_df.columns) + ["normalized_nose_x", "normalized_nose_y"],
    index=horiz_imgs_df.index)

# filtering - only visible noses

coco_noses_df = coco_noses_df[coco_noses_df["v0"] == 2]

coco_noses_df.plot(kind="scatter", x="normalized_nose_x",
                   y="normalized_nose_y", alpha=0.3).invert_yaxis()


low_noses_df = coco_noses_df[coco_noses_df['normalized_nose_y'] > 430 ]
low_noses_df

y_images = coco_extra_attribs_df['num_keypoints'].value_counts()
x_keypoints = y_images.index.values

# plot

plt.figsize=(10,5)
plt.bar(x_keypoints, y_images.values)
plt.title('Histogram of keypoints')
plt.xlabel('Number of keypoints')
plt.ylabel('Number of bboxes')
plt.show()

# percentage of bboxes (column) with a number of keypoints (rows)

kp_df = pd.DataFrame({
    "Num keypoints %": coco_extra_attribs_df[
                           "num_keypoints"].value_counts() / len(coco_extra_attribs_df)
}).sort_index()

persons_df = coco_extra_attribs_df[coco_extra_attribs_df['num_keypoints'] > 0]
persons_df.sort_values('scale_cat', inplace=True)
# persons_df['scale_cat'].hist()
# print(persons_df['scale_cat'].hist())
plt.hist(persons_df['scale_cat'])
plt.show()

scales_props_df = pd.DataFrame({
    "Scales": persons_df["scale_cat"].value_counts() / len(persons_df)
})
scales_props_df

persons_df = coco_extra_attribs_df[coco_extra_attribs_df['num_keypoints'] > 0]
train_df = persons_df[persons_df['source'] == 0]
val_df = persons_df[persons_df['source'] == 1]

scales_props_df = pd.DataFrame({
    "Scales in train set %": train_df["scale_cat"].value_counts() / len(train_df),
    "Scales in val set %": val_df["scale_cat"].value_counts() / len(val_df)
})
scales_props_df["Diff 100%"] = 100 * \
    np.absolute(scales_props_df["Scales in train set %"] -
                scales_props_df["Scales in val set %"])

train_df = coco_extra_attribs_df[coco_extra_attribs_df['source'] == 0]
val_df = coco_extra_attribs_df[coco_extra_attribs_df['source'] == 1]

kp_props_df = pd.DataFrame({
    "Num keypoints in train set %": train_df["num_keypoints"].value_counts() /
    len(train_df),
    "Num keypoints in val set %": val_df["num_keypoints"].value_counts() /
    len(val_df)
}).sort_index()

kp_props_df["Diff 100%"] = 100 * \
    np.absolute(kp_props_df["Num keypoints in train set %"] -
                kp_props_df["Num keypoints in val set %"])

