import argparse
import os
import random
import mmcv
import torch
import matplotlib.pyplot as plt
from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import pyautogui as auto
def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/zeniqa/zeniqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default='../RealCamera25k/RealCamera', help='path to input image file')
    parser.add_argument('--csv_path', default='real_camera_distributions_sets.csv', help='path to output csv file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 初始化模型
    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # 获取输入目录下所有图片文件
    image_files = [f for f in os.listdir(args.file_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # 如果CSV文件已存在，读取它；否则创建新的DataFrame
    if os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        # 确保有需要的列
        required_columns = ['image_name', 'score1', 'score2', 'score3', 'score4', 
                          'score5', 'score6', 'score7', 'final_score']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan if col != 'image_name' else ''
    else:
        df = pd.DataFrame(columns=['image_name', 'score1', 'score2', 'score3', 'score4', 
                                 'score5', 'score6', 'score7', 'final_score'])
    
    # 处理每张图片
    new_rows = []
    for image_name in tqdm(image_files, desc='Processing images'):
        # 检查是否已经处理过这张图片
        # if image_name in df['image_name'].values:
        #     continue
            
        # 进行推理
        try:
            output, attributes = restoration_inference(
                model, 
                os.path.join(args.file_path, image_name), 
                return_attributes=True
            )
            output = output.float().detach().cpu().numpy()          
            
            # 提取并计算各项分数
            scores = {
                'image_name': image_name,
                'score1': output[0][0] * 100,
                'score2': output[0][1] * 100,
                'score3': output[0][2] * 100,
                'score4': output[0][3] * 100,
                'score5': output[0][4] * 100,
                'score6': output[0][5] * 100,
                'score7': output[0][6] * 100,
                'final_score': np.average([
                    output[0][0], 
                    output[0][1], 
                    np.min(output[0][2:4]), 
                    np.min(output[0][4:6]), 
                    output[0][6]
                ]) * 100
            }
            
            # 添加到新行列表
            new_rows.append(scores)
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    # 如果有新数据，合并到原DataFrame
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(subset=['image_name'], keep='last')
    
    # 保存结果到CSV文件
    df.to_csv(args.csv_path, index=False)
    print(f"Results saved to {args.csv_path}")

    # 可选：显示一些示例结果
    if len(df) > 0:
        display_random_images(df, args)


def display_random_images(df, args):
    sample_size = min(10, len(df))
    random_indices = random.sample(range(len(df)), sample_size)
    
    plt.figure(figsize=(20, 10))  # 增大图像尺寸以适应更多信息

    for idx, row_idx in enumerate(random_indices):
        row = df.iloc[row_idx]
        img_path = os.path.join(args.file_path, row['image_name'])
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Image not found or corrupted")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(2, 5, idx + 1)
            plt.imshow(img)
            plt.axis('off')
            
            # 创建更详细的标题
            title = (
                f"{os.path.basename(row['image_name'])}\n"
                f"Score1: {row['score1']:.1f}\n"
                f"Score2: {row['score2']:.1f}\n"
                f"Score3: {row['score3']:.1f}\n"
                f"Score4: {row['score4']:.1f}\n"
                f"Score5: {row['score5']:.1f}\n"
                f"Score6: {row['score6']:.1f}\n"
                f"Score7: {row['score7']:.1f}\n"
                f"Final: {row['final_score']:.1f}"
            )
            plt.title(title, fontsize=8)
            
        except Exception as e:
            print(f"Error displaying {row['image_name']}: {str(e)}")
            continue

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()