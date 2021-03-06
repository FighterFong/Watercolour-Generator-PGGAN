# encoding=utf-8
import os
import errno
import numpy as np
import scipy
import scipy.misc
import h5py
import cv2

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def print_msg(tag, message):
    print('== [' + str(tag) + ']', message)

class CelebA(object):
    def __init__(self, image_path):
        self.dataname = "CelebA"
        self.channel = 3
        # image_list是数据集所有图片的本地存储路径地址
        self.image_list = self.load_celebA(image_path=image_path)

    def load_celebA(self, image_path):
        # get the list of image path
        images_list = read_image_list(image_path)
        # get the data array of image
        return images_list

    # 对那一个batch的path读取成图片。load images by path and resize it
    def getShapeForData(self, filenames, resize_w=64):
        array = [get_image(batch_file, image_size=256, is_crop=False, resize_w=resize_w,  # image_size=128
                           is_grayscale=False) for batch_file in filenames]  # is_crop=True, 人脸crop的意义何在，反正在jellyfish上采用crop局部效果贼差

        try:
            sample_images = np.array(array)
        except Exception as e:
            print('[DEBUG] Data Format Error,Ignore it!The Image Path Is:',sample_images)
        # ('==', <type 'numpy.ndarray'>)
        # print('==',type(sample_images))
        # ('==', (16, 4, 4, 3))
        # print('==',sample_images.shape)
        
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    # 拿了一个batch的image path。acquire next batch image path list
    def getNextBatch(self, batch_num=0, batch_size=64):
        ro_num = len(self.image_list) / batch_size - 1
        if batch_num % ro_num == 0:
            length = len(self.image_list)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.image_list = np.array(self.image_list)
            self.image_list = self.image_list[perm]
            print("images shuffle！")

        # 在shuffle之后的image_list里面取一个batch返回
        return self.image_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

class CelebA_HQ(object):
    def __init__(self, image_path):
        self.dataname = "CelebA_HQ"
        resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64', \
                      'data128x128', 'data256x256', 'data512x512', 'data1024x1024']
        self.channel = 3
        self.image_list = self.load_celeba_hq(image_path=image_path)
        self._base_key = 'data'
        self._len = {k: len(self.image_list[k]) for k in resolution}

    def load_celeba_hq(self, image_path):
        # get the list of image path
        images_list = h5py.File(os.path.join(image_path, "celebA_hq"), 'r')
        # get the data array of image
        return images_list

    def getNextBatch(self, batch_size=64, resize_w=64):
        key = self._base_key + '{}x{}'.format(resize_w, resize_w)
        idx = np.random.randint(self._len[key], size=batch_size)
        batch_x = np.array([self.image_list[key][i] / 127.5 - 1.0 for i in idx], dtype=np.float32)

        return batch_x

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    # load images and resize
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def get_image_dat(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread_dat(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx=64, is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
        # cropped_image = center_crop(image, npx/2, npx, resize_w=resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    # 长宽一致
    if crop_w is None:
        crop_w = crop_h
    # 图像尺寸：长、宽
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)

    if rate < 0.5:
        # 矩阵左右翻转
        x = np.fliplr(x)

    # 先从原图中从中心crop出crop_h*crop_w大小的图片，再resize
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])

        # return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

    # return scipy.misc.imresize(x[45: 45 + 128, 25:25 + 128], [resize_w, resize_w])

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        # image = scipy.misc.imread(path).astype(np.float)
        image = cv2.imread(path)
        # print('=='*20)
        # print(image.shape)
        # print('=='*20)
        return image

def imread_dat(path, is_grayscale):
    return np.load(path)

# 一个 batch 图片整合在一起，整合成一个图片保存
# def imsave(images, size, path):
#     return scipy.misc.imsave(path, merge(images, size))

# 一个 batch 的图片，逐个保存
def imsave(images, size, path):
    # print('=='*20)
    # print('2-',type(images))
    # print('2-',images.shape)
    merged_image = merge(images, size)
    # print(type(merged_image))
    # print(merged_image.shape)

    for idx, image in enumerate(images):
        # ./output/Experiment_6_30_1/0/sample_6_False/00_real.jpg
        index = '_' + str(idx)
        saved_path = path.replace('.jpg', index) + '.jpg'
        # print(saved_path)
        scipy.misc.imsave(saved_path, image)
    # return True

# 将 batch_size 张图片，堆叠成size的尺寸，即多张小图显示在一张大图上
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image
    return img

# transfer 0-1 to 0-255
def inverse_transform(image):
    return ((image + 1.)* 127.5).astype(np.uint8)

def read_image_list(category):
    filenames = []
    # print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if 'jpg' in file:
            filenames.append(category + "/" + file)
    # print("list file ending!")
    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

# def read_image_list(category):
#     # 可以使用 glob 来重写
#     pass

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)




