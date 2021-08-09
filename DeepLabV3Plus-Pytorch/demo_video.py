from matplotlib import image
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import cv2

# img_dir = '/data1/users/caijh28/data/cityscapes/leftImg8bit/demo_video/stuttgart_00'
# files_name = os.listdir(img_dir)
# files_name = sorted(files_name)
# index = 0
# for file_name in files_name:
#     img = image.imread(os.path.join(img_dir,file_name))
#     msk = Image.open('./close-set-imgs/preds_base_'+str(index)+'.jpg')
#     # msk = Image.open('./incre-imgs/preds_base_' + str(index) + '.jpg')
#     # msk = Image.open('./anomaly/preds_anomaly_' + str(index) + '.jpg')
#     out = msk.resize((2048, 1024), Image.ANTIALIAS)
#     plt.figure()
#     plt.imshow(img, alpha=0.5)
#     plt.imshow(out, alpha=0.5)
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.savefig('close-set-imgs-add/preds_base_' + str(index) + '.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
#     # plt.show()
#     plt.close()
#     index += 1

# img_dir = './incre-imgs-add/'
# video_dir = './incre-5-car-newhead.avi'
# files_name = os.listdir(img_dir)
# # files_name = sorted(files_name)
# index = 0
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
# fps = 20
# # img_size = (1488,2976)
# img_size = (2976,1488)
# videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size, True)
# for i in range(599):
#     file_name = 'preds_base_' + str(i) + '.jpg'
#     frame = cv2.imread(os.path.join(img_dir, file_name))
#     videoWriter.write(frame)
# # for file_name in files_name:
# #     print(file_name)
# #     frame = cv2.imread(os.path.join(img_dir,file_name))
# #     videoWriter.write(frame)
# videoWriter.release()

msk_dir = './close-set-imgs/'
video_dir = './incre-imgs-3000-new.avi'
imgs_dir = '/data1/users/caijh28/data/cityscapes/leftImg8bit/demo_video/stuttgart_00'
files_name = os.listdir(imgs_dir)
files_name = sorted(files_name)
index = 0
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
fps = 20
# img_size = (1488,2976)
img_size = (2976,1488)
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size, True)
for i in range(599):
    msk_name = 'preds_base_' + str(i) + '.jpg'
    msk = cv2.imread(os.path.join(msk_dir, msk_name))
    # print(msk.shape)
    image = files_name[i]
    image = cv2.imread(os.path.join(imgs_dir,image))
    image = cv2.resize(image, (2976,1488), interpolation=cv2.INTER_CUBIC)
    # print(image.shape)
    frame = cv2.addWeighted(image, 0.1, msk, 0.9, 0)
    # frame = msk
    videoWriter.write(frame)
# for file_name in files_name:
#     print(file_name)
#     frame = cv2.imread(os.path.join(img_dir,file_name))
#     videoWriter.write(frame)
videoWriter.release()



