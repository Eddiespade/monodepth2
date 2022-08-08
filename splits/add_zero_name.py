import os.path


def rename(img_folder):
    for img_name in os.listdir(img_folder):  # os.listdir()： 列出路径下所有的文件
        # os.path.join() 拼接文件路径
        img_name_list = img_name.split(".")[0]
        src = os.path.join(img_folder, img_name)  # src：要修改的目录名
        dst = os.path.join(img_folder, '0000' + img_name_list + '.png')  # dst： 修改后的目录名      注意此处str(num)将num转化为字符串,继而拼接
        os.rename(src, dst)  # 用dst替代src


def main():
    img_folder0 = '/home/yk/dxl/test-dl/work/image_/01/image_03/data'  # 图片的文件夹路径    直接放你的文件夹路径即可
    img_folder1 = '/home/yk/dxl/test-dl/work/image_/01/image_02/data'  # 图片的文件夹路径    直接放你的文件夹路径即可
    img_folder2 = '/home/yk/dxl/test-dl/work/image_/02/image_03/data'  # 图片的文件夹路径    直接放你的文件夹路径即可
    img_folder3 = '/home/yk/dxl/test-dl/work/image_/02/image_02/data'  # 图片的文件夹路径    直接放你的文件夹路径即可
    num = 0
    rename(img_folder0)
    rename(img_folder1)
    rename(img_folder2)
    rename(img_folder3)


if __name__ == "__main__":
    main()
