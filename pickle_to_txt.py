import pickle
import os
from PIL import Image
for i in range(10002,10045):
    if not os.path.exists(r"E:\Pose detection\AwA-Pose\Annotations\antelope\antelope_"+str(i)+".pickle"):
        continue
    with open(r"E:\Pose detection\AwA-Pose\Annotations\antelope\antelope_"+str(i)+".pickle", 'rb') as f:
        x = pickle.load(f)
    # with open("antelope_"+str(i)+".txt", 'w') as f:
    #     f.write(str(x))
    # with open(r"E:\Pose detection\AwA-Pose\Annotations\antelope\antelope_10002.pickle", 'rb') as f:
    #         x = pickle.load(f)
    class_labels = [
        "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
        "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
        "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
        "neck_base", "neck_end", "throat_base", "throat_end", "back_base", "back_end", "back_middle",
        "tail_base", "tail_end", "front_left_thai", "front_left_knee", "front_left_paw",
        "front_right_thai", "front_right_paw", "front_right_knee", "back_left_knee", "back_left_paw",
        "back_left_thai", "back_right_thai", "back_right_paw", "back_right_knee", "belly_bottom",
        "body_middle_right", "body_middle_left"
    ]
    x=x["a1"]
    with Image.open(r"E:\Pose detection\pose-detection-keypoints-estimation-yolov8\data\images\train\antelope_"+str(i)+".jpg") as img:
        width, height = img.size
    print(len(x),"asd")
    s="0 "+str(x["bbox"][0]/width)+" "+str(x["bbox"][1]/height)+" "+str(x["bbox"][2]/width)+" "+str(x["bbox"][3]/height)
    for j in class_labels:
        if x[j][0] == -1:
            s+=" 0 0 0"
        else:
            s+=" "+str(x[j][0]/width)+" "+str(x[j][1]/height)+" "+"2"
    with open("antelope_"+str(i)+".txt", 'w') as f:
        f.write(s)
