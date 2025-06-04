import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ZENIQAPreprocessor:
    """处理与训练时完全一致的图像预处理"""
    def __init__(self, input_resolution=224):
        # 与CLIP模型一致的mean和std
        self.mean=[0.48145466, 0.4578275, 0.40821073]
        self.std=[0.26862954, 0.26130258, 0.27577711]
        self.input_resolution=input_resolution

        # 构建预处理pipeline
        self.transform=transforms.Compose([
            transforms.Resize((input_resolution, input_resolution),
            transforms.InterpolationMode.BICUBIC,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess(self, image_path):
        """从文件路径预处理图像"""
        img=Image.open(image_path).convert('RGB')
        return self.preprocess_pil(img)

    def preprocess_pil(self, pil_image):
        """从PIL图像预处理"""
        return self.transform(pil_image).unsqueeze(0)  # 添加batch维度

    def preprocess_array(self, np_array):
        """从numpy数组预处理"""
        pil_img=Image.fromarray(np_array)
        return self.preprocess_pil(pil_img)

class ZENIQAModelWrapper:
    """封装COOP-CLIP-IQA模型加载和推理逻辑"""
    def __init__(self, model_path, classnames, backbone_name='RN50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device=device
        self.classnames=classnames
        self.backbone_name=backbone_name

        # 动态导入必要的模块
        from mmedit.models.backbones.sr_backbones.coopclipiqa import CLIPIQAFixed
        from mmedit.models.components.clip import clip

        self.model=self._load_model(model_path, CLIPIQAFixed)
        self.preprocessor=ZENIQAPreprocessor()

    def _load_model(self, model_path, model_class):
        """加载训练好的COOP-CLIP-IQA模型"""
        # 初始化模型
        model=model_class(
            classnames=self.classnames,
            backbone_name=self.backbone_name
        )

        # 加载预训练权重
        checkpoint=torch.load(model_path, map_location=self.device)

        # 处理可能的并行训练保存的模型
        if 'state_dict' in checkpoint:
            state_dict=checkpoint['state_dict']
            # 移除可能的前缀
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name=k[7:]  # 移除 'module.' 前缀
                    new_state_dict[name]=v
                else:
                    new_state_dict[k]=v
            state_dict=new_state_dict
        else:
            state_dict=checkpoint

        # 加载状态字典，忽略不匹配的键
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input):
        """
        输入可以是: 文件路径/PIL图像/numpy数组
        返回: 质量分数和属性预测
        """
        # 预处理
        if isinstance(image_input, str):
            tensor=self.preprocessor.preprocess(image_input)
        elif isinstance(image_input, Image.Image):
            tensor=self.preprocessor.preprocess_pil(image_input)
        elif isinstance(image_input, np.ndarray):
            tensor=self.preprocessor.preprocess_array(image_input)
        else:
            raise ValueError("不支持的输入类型，请提供文件路径、PIL图像或numpy数组")

        # 推理
        with torch.no_grad():
            tensor=tensor.to(self.device)
            pred_score, attributes=self.model(tensor)

        # 转换为numpy数组并返回
        return {
            'quality_score': pred_score.cpu().numpy().squeeze(),
            'attributes': attributes.cpu().numpy().squeeze()
        }

    def export_to_onnx(self, output_path="zeniqa_coop.onnx", input_shape=(1, 3, 224, 224)):
        """导出模型为ONNX格式"""
        dummy_input=torch.randn(*input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["quality_score", "attributes"],
            dynamic_axes={
                'input': {0: 'batch'},
                'quality_score': {0: 'batch'},
                'attributes': {0: 'batch'}
            },
            opset_version=11,
            export_params=True,
            do_constant_folding=True
        )
        print(f"模型已导出到 {output_path}")

# 示例使用代码
if __name__ == "__main__":
    # 定义classnames (与训练时一致)
    classnames=[
        ['sharp photo, good photo.', 'blurred photo, bad photo.'],  # Sharp
        ['noiseless photo, good photo.', 'noise photo, bad photo.'],  # Noise
        ['well exposed photo, good photo.',
            'white out overexposed photo, bad photo.'],  # Bright1
        ['well exposed photo, good photo.',
            'black out underexposed photo, bad photo.'],  # Bright2
        ['colorful photo, good photo.',
            'high saturation photo, bad photo.'],  # Color1(Edit)
        ['colorful photo, good photo.',
            'low saturation photo, bad photo.'],  # Color2(Edit)
        ['high contrast photo, good photo.',
            'low contrast photo, bad photo.'],  # Contrast
    ]

    # 初始化
    model_wrapper=ZENIQAModelWrapper(
        model_path="path_to_your_model.pth",
        classnames=classnames,
        backbone_name='RN50'
    )

    # 示例预测
    result=model_wrapper.predict("example_image.jpg")
    print("质量分数:", result['quality_score'])
    print("属性预测:", result['attributes'])

    # 导出ONNX
    model_wrapper.export_to_onnx()
