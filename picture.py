import numpy as np
import os
import glob
import io
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 靴子图片路径
Boot_path = "Shoe vs Sandal vs Boot Dataset/Boot"
# 凉鞋图片路径
Sandal_path = "Shoe vs Sandal vs Boot Dataset/Sandal"
# 平底鞋图片路径
Shoe_path = "Shoe vs Sandal vs Boot Dataset/Shoe"
# 需要识别的类型
classes = {'Boot', 'Sandal', 'Shoe'}
# 样本总数
num_sample = 15000

Boot = []
label_Boot = []
Sandal = []
label_Sandal = []
Shoe = []
label_Shoe = []


# step1：获取所有的图片路径名，存放到对应的列表中，同时贴上标签，存放到label列表中。
def get_file():
    for file in os.listdir(Boot_path):
        Boot.append(Boot_path + '/' + file)
        label_Boot.append(0)
    for file in os.listdir(Sandal_path):
        Sandal.append(Sandal_path + '/' + file)
        label_Sandal.append(1)
    for file in os.listdir(Shoe_path):
        Shoe.append(Shoe_path + '/' + file)
        label_Shoe.append(2)

    # step2：对生成的图片路径和标签List做打乱处理把所有的数据合起来组成一个list（img和lab）
    image_list = np.hstack((Boot, Sandal, Shoe))
    label_list = np.hstack((label_Boot, label_Sandal, label_Shoe))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    # 将img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 将照片读取出来
    image = []
    for im in all_image_list:
        # print(im)
        img = cv2.imread(im)
        img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
        image.append(img)

    image = np.array(image)
    labels = np.array(all_label_list)
    # print(image.shape)
    # print(labels.shape)

    return image, labels


if __name__ == '__main__':
    get_file()
