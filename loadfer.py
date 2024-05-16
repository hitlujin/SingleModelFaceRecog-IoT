import pandas as pd
import os
from PIL import Image
import glob
import numpy as np
import torch
import clip
from celebAload import load_samples
from PIL import Image
from matplotlib import pyplot as plt

# load model
clip_type = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 
             'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'][4]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_type, device=device)

def load_raf_original_dataset(image_path, label_path,num=1000):
    # 读取标签文件
    label_df = pd.read_csv(label_path, sep=' ', names=['image_name', 'emotion_label'])

    # 获取图像文件名列表
    image_filenames = os.listdir(image_path)

    # 初始化图像和标签列表
    images = []
    labels = []

    # iter 1000 images
    for img_name in image_filenames[:num]:
        # 检查是否为 jpg 文件
        if img_name.endswith(".jpg"):
            # 读取图像文件
            img = Image.open(os.path.join(image_path, img_name))
            images.append(img)

            # 获取对应的标签
            label = label_df[label_df['image_name'] == img_name]['emotion_label'].values[0]
            labels.append(label)

    return images, labels

# 使用函数加载数据
images_path = "F:/DLdataset/basic/image/original"  # 使用原始图像
label_path = "F:/DLdataset/basic/EmoLabel/list_patition_label.txt"
original_images, original_labels = load_raf_original_dataset(images_path, label_path)
# 输出加载的图像和标签数量
print(f"Loaded {len(original_images)} original images and {len(original_labels)} labels.")

prompt_one = ["A photo featuring a person with a look of surprise", 
              "An image depicting a frightened individual", 
              "A picture capturing a person expressing disgust", 
              "A snapshot showcasing a happy person", 
              "A photograph displaying a sad individual", 
              "An image capturing an angry person", 
              "A photo presenting a neutral expression"]
prompt_two = ["This photo features a person with a look of surprise, their eyes widened and mouth open in astonishment.",
            "In this image, a frightened individual is depicted, their eyes wide and face tense with fear.",
            "The picture captures a person expressing disgust, their nose wrinkled and mouth curled in distaste.",
            "This snapshot showcases a happy person, their eyes sparkling and mouth curved in a joyful smile.",
            "The photograph displays a sad individual, their eyes downcast and lips turned downward in sorrow.",
            "This image captures an angry person, their eyebrows furrowed and lips pressed tightly together in frustration.",
            "The photo presents a neutral expression, the person's face showing no particular emotion."]

prompt_three = [
    "surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"
]
prompt_four = ["face surprise eyes widened mouth open", 
               "face fear eyes wide tense", 
               "face disgust nose wrinkled mouth curled", 
               "face happy eyes sparkling smile", 
               "face sad eyes downcast lips downward", 
               "face angry eyebrows furrowed lips tight", 
               "face neutral expression emotionless"]

prompt_five = ["A photo featuring a person with a look of surprise", 
              "An photo depicting a frightened individual", 
              "A photo capturing a person expressing disgust", 
              "A photo showcasing a happy person", 
              "A photo displaying a sad individual", 
              "A photo capturing an angry person", 
              "A photo presenting a neutral expression"]

prompt_six = ["surprised facial expression eyes wide mouth open", 
              "fearful facial expression wide eyes tense face", 
              "disgusted facial expression wrinkled nose curled lips", 
              "happy facial expression sparkling eyes joyful smile", 
              "sad facial expression downcast eyes downturned lips", 
              "angry facial expression furrowed eyebrows tight lips", 
              "neutral facial expression emotionless face"]
images_features = []
text_features = []
for img in original_images:
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feature = model.encode_image(img)
    images_features.append(img_feature)
images_features = torch.cat(images_features, dim=0)


for prompt in prompt_six:
    text_features.append(model.encode_text(clip.tokenize(prompt).to(device)))
text_features = torch.cat(text_features, dim=0)
similarity_matrix = images_features @ text_features.T
similarity_matrix = torch.softmax(similarity_matrix, dim=1)
arg_max = torch.argmax(similarity_matrix, dim=1).to("cpu") + 1
acc = torch.sum(arg_max == torch.tensor(original_labels)).item() / len(original_labels)
print(f"Accuracy: {acc:.4f}")

# prompt_one 0.6010
# prompt_two 0.26
# prompt_three 0.8000
# prompt_four 0.5590
# prompt_five 0.4840
# prompt_six 0.6700



 