import os
import sys
import torch
import numpy as np
import json
import clip
import cv2
import torch.nn as nn
import torchvision.models as models
import faiss
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time
import signal
import glob

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 允许numpy.core.multiarray._reconstruct为安全全局变量
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])

# 定义终止标志
TERMINATE = False

# 信号处理器
def signal_handler(sig, frame):
    global TERMINATE
    print("\n收到终止信号，正在安全地停止处理...")
    TERMINATE = True
    # 增加直接退出功能，确保Ctrl+C可以立即响应
    print("程序将在5秒后退出，再次按Ctrl+C可立即退出...")
    time.sleep(1)  # 给一点时间打印消息
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 配置参数
DATASET_ROOT = "./datasets"  # 数据集根目录
RESULTS_DIR = "./retrieval_results"  # 结果保存在当前目录
LAMAINPAINT_DIR = "../lamainpaint"  # lamainpaint目录
SHOTS = [1, 5, 10]  # 要处理的shot数量
os.makedirs(RESULTS_DIR, exist_ok=True)

# ResNet50提取器 - 只使用低层特征
class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # 只提取低级特征
        self.enc_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
    
    def forward(self, x):
        return self.enc_layers(x)

# 计算风格特征的均值和方差
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# 清理图像路径的函数，移除多余的"pipeline/"前缀
def clean_image_path(path):
    """清理图像路径，移除多余的pipeline/前缀"""
    if isinstance(path, str):
        # 替换错误路径中的"../../pipeline/datasets"为"../../datasets"
        if "../../pipeline/datasets" in path:
            return path.replace("../../pipeline/datasets", "../../datasets")
        # 替换../../datasets/coco为./coco
        if "../../datasets/coco" in path:
            return path.replace("../../datasets/coco", "./coco")
    return path

# 获取inpainted图像及其所属类别
def get_inpainted_images(dataset_name, shot_count):
    """从[shot数量]_shot目录中获取所有.jpg图像及其类别信息
    
    参数:
        dataset_name: 数据集名称
        shot_count: shot数量 (1, 5, 10)
        
    返回:
        dict: 样本ID到图像路径的映射
        dict: 样本ID到类别名称的映射（这里使用样本ID作为类别）
    """
    # 构建shot目录路径
    shot_dir = os.path.join(LAMAINPAINT_DIR, dataset_name, f"{shot_count}_shot")
    if not os.path.exists(shot_dir):
        print(f"错误：找不到数据集 {dataset_name} 的 {shot_count}_shot 目录: {shot_dir}")
        return {}, {}
    
    print(f"使用shot目录: {shot_dir}")
    
    # 获取所有jpg图像文件
    image_files = glob.glob(os.path.join(shot_dir, "*.jpg"))
    
    if not image_files:
        print(f"错误：在 {shot_dir} 中找不到任何jpg图像")
        return {}, {}
    
    # 创建样本ID到图像路径的映射
    sample_to_image = {}
    # 创建样本ID到类别的映射（这里暂时使用样本ID作为类别）
    sample_to_category = {}
    
    # 尝试加载类别映射文件（如果存在）
    category_mapping = {}
    category_mapping_file = os.path.join(shot_dir, "category_mapping.json")
    
    if os.path.exists(category_mapping_file):
        try:
            with open(category_mapping_file, 'r') as f:
                category_mapping = json.load(f)
            print(f"已加载类别映射文件: {category_mapping_file}")
        except Exception as e:
            print(f"加载类别映射文件时出错: {e}")
    
    # 处理每个jpg图像
    for img_path in image_files:
        # 从文件名中提取样本ID（去掉扩展名）
        sample_id = os.path.splitext(os.path.basename(img_path))[0]
        sample_to_image[sample_id] = img_path
        
        # 如果在category_mapping中找到匹配项，使用映射的类别名
        if sample_id in category_mapping:
            sample_to_category[sample_id] = category_mapping[sample_id]
        else:
            # 否则使用样本ID作为类别名
            sample_to_category[sample_id] = sample_id
    
    print(f"找到 {len(sample_to_image)} 个inpainted图像")
    
    # 按类别统计图像数量
    category_counts = {}
    for sample_id, category in sample_to_category.items():
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    
    print("类别统计:")
    for category, count in category_counts.items():
        print(f"  - {category}: {count}张图像")
    
    return sample_to_image, sample_to_category

# 从整体图像提取CLIP特征
def extract_clip_global_features(image_path, preprocess, model, device):
    """从整个图像提取CLIP特征"""
    try:
        # 清理图像路径
        image_path = clean_image_path(image_path)
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        return image_embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"提取CLIP特征时出错: {e}, 图像: {image_path}")
        return None

# 计算ResNet低层特征
def compute_resnet_features(image_path, model, device):
    """计算图像的ResNet风格特征"""
    try:
        # 清理图像路径
        image_path = clean_image_path(image_path)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像 {image_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img = img.to(device)
        
        with torch.no_grad():
            features = model(img)
            mean, std = calc_mean_std(features)
        
        return torch.cat([mean.squeeze(), std.squeeze()]).cpu().numpy()
    except Exception as e:
        print(f"计算ResNet特征时出错: {e}, 图像: {image_path}")
        return None

# 初始化CLIP模型
def init_clip_model(device):
    """初始化CLIP模型"""
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("成功加载CLIP模型 ViT-B/32")
        return model, preprocess
    except Exception as e:
        print(f"加载CLIP模型时出错: {e}")
        print("正在尝试备选方法...")
        try:
            from clip.clip import load as clip_load
            model, preprocess = clip_load("ViT-B/32", device=device)
            print("成功使用备选方法加载CLIP模型")
            return model, preprocess
        except Exception as e2:
            print(f"加载CLIP模型失败: {e2}")
            sys.exit(1)

# 初始化ResNet模型
def init_resnet_model(device):
    """初始化ResNet模型"""
    try:
        model = ResNetEncoder().to(device).eval()
        print("成功加载ResNet特征提取器")
        return model
    except Exception as e:
        print(f"加载ResNet模型时出错: {e}")
        sys.exit(1)

# 计算COCO图像的CLIP特征
def compute_coco_clip_features(coco_dir, model, preprocess, device):
    """计算COCO数据集的CLIP特征"""
    global TERMINATE
    
    # 确定COCO图像目录
    coco_images_dir = None
    for subdir in ["images", "train2017", "val2017"]:
        test_path = os.path.join(coco_dir, subdir)
        if os.path.exists(test_path) and os.path.isdir(test_path):
            coco_images_dir = test_path
            break
    
    if coco_images_dir is None:
        print(f"错误：找不到COCO图像目录，已检查 {coco_dir}")
        return [], []
    
    print(f"使用COCO图像目录: {coco_images_dir}")
    
    # 获取所有图像文件
    image_paths = []
    for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
        found_files = list(Path(coco_images_dir).glob(f"**/{img_ext}"))
        image_paths.extend([str(p) for p in found_files])
    
    if not image_paths:
        print(f"错误：在 {coco_images_dir} 中找不到任何图像")
        return [], []
    
    print(f"找到 {len(image_paths)} 张COCO图像")
    
    # 计算特征
    features = []
    valid_paths = []
    
    for i, img_path in enumerate(tqdm(image_paths, desc="提取COCO CLIP特征")):
        # 检查是否应该终止
        if TERMINATE or (i > 0 and i % 10 == 0 and TERMINATE):
            print(f"正在终止COCO特征提取，已完成 {i}/{len(image_paths)} 张图像")
            break
            
        try:
            # 清理图像路径
            img_path = clean_image_path(img_path)
            
            image = Image.open(img_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model.encode_image(image_tensor)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                
            features.append(feature.cpu().numpy()[0])
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"处理图像时出错 {img_path}: {e}")
            continue
        
        # 每1000张图像保存一次进度
        if i > 0 and i % 1000 == 0:
            print(f"已处理 {i}/{len(image_paths)} 张图像")
    
    return np.array(features), valid_paths

# 添加新函数：获取并计算inpainted图像的CLIP特征
def compute_inpainted_clip_features(dataset_name, shot_count, model, preprocess, device):
    """获取并计算指定数据集和指定shot的inpainted图像的CLIP特征"""
    global TERMINATE
    
    # 构建shot目录路径
    shot_dir = os.path.join(LAMAINPAINT_DIR, dataset_name, f"{shot_count}_shot")
    if not os.path.exists(shot_dir):
        print(f"警告：找不到数据集 {dataset_name} 的 {shot_count}_shot 目录: {shot_dir}")
        return [], []
    
    print(f"使用shot目录: {shot_dir}")
    
    # 获取所有jpg图像文件
    image_paths = glob.glob(os.path.join(shot_dir, "*.jpg"))
    
    if not image_paths:
        print(f"警告：在 {shot_dir} 中找不到任何jpg图像")
        return [], []
    
    print(f"找到 {len(image_paths)} 张inpainted图像")
    
    # 计算特征
    features = []
    valid_paths = []
    
    for i, img_path in enumerate(tqdm(image_paths, desc=f"提取{dataset_name} {shot_count}_shot inpainted CLIP特征")):
        # 检查是否应该终止
        if TERMINATE or (i > 0 and i % 10 == 0 and TERMINATE):
            print(f"正在终止inpainted特征提取，已完成 {i}/{len(image_paths)} 张图像")
            break
            
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model.encode_image(image_tensor)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                
            features.append(feature.cpu().numpy()[0])
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"处理inpainted图像时出错 {img_path}: {e}")
            continue
        
        # 每100张图像打印一次进度
        if i > 0 and i % 100 == 0:
            print(f"已处理 {i}/{len(image_paths)} 张inpainted图像")
    
    return np.array(features), valid_paths

# 可视化检索结果
def visualize_results(query_image_path, result_image_paths, output_path):
    """将查询图像和检索结果可视化"""
    plt.figure(figsize=(15, 8))
    
    # 显示查询图像
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"警告：无法读取查询图像 {query_image_path}")
        return
    
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 4, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")
    
    # 显示结果图像
    for i, img_path in enumerate(result_image_paths):
        if i >= 11:  # 最多显示11个结果
            break
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图像 {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 4, i + 2)
            plt.imshow(img)
            plt.title(f"Top {i+1}")
            plt.axis("off")
        except Exception as e:
            print(f"可视化图像时出错 {img_path}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"已保存可视化结果到 {output_path}")

# 使用CLIP进行第一阶段检索（支持多数据集）
def clip_first_stage_retrieval(query_feature, dataset_features, dataset_paths, top_k=100):
    """第一阶段：使用CLIP特征检索top_k个结果
    
    dataset_features和dataset_paths是字典，键为数据集名称，值为特征和路径
    """
    # 合并所有数据集的特征和路径
    all_features = []
    all_paths = []
    all_sources = []  # 记录每个特征的来源数据集
    
    for dataset_name, features in dataset_features.items():
        if features is not None and len(features) > 0:
            paths = dataset_paths[dataset_name]
            print(f"将从{dataset_name}数据集({len(features)}张图像)中检索")
            all_features.append(features)
            all_paths.extend(paths)
            all_sources.extend([dataset_name] * len(paths))
    
    if not all_features:
        print("错误：没有可用的数据集特征")
        return []
    
    # 合并特征
    all_features = np.vstack(all_features)
    
    print(f"总共使用 {len(all_features)} 张图像进行检索")
    
    # 创建FAISS索引
    d = query_feature.shape[0]  # 特征维度
    index = faiss.IndexFlatIP(d)  # 使用余弦相似度
    
    try:
        # 添加到索引
        features_np = np.array(all_features).astype(np.float32)
        index.add(features_np)
        
        # 搜索
        query_np = np.array([query_feature]).astype(np.float32)
        D, I = index.search(query_np, min(top_k, len(all_features)))
        
        # 返回结果
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(all_paths):  # 安全检查
                results.append({
                    "similarity": float(D[0][i]),
                    "image_path": all_paths[idx],
                    "source_dataset": all_sources[idx],  # 添加数据集来源
                    "index": int(idx)
                })
        
        return results
        
    except Exception as e:
        print(f"CLIP检索时出错: {e}")
        return []

# 使用ResNet进行第二阶段检索
def resnet_second_stage_rerank(query_image_path, first_stage_results, resnet_model, device):
    """第二阶段：使用ResNet低层特征对CLIP结果重新排序"""
    # 清理查询图像路径
    query_image_path = clean_image_path(query_image_path)
    
    # 获取查询图像的ResNet特征
    query_resnet_feature = compute_resnet_features(query_image_path, resnet_model, device)
    if query_resnet_feature is None:
        print(f"警告：无法计算查询图像的ResNet特征: {query_image_path}")
        return first_stage_results
    
    # 计算第一阶段结果的ResNet特征
    rerank_results = []
    
    for result in tqdm(first_stage_results, desc="ResNet二次排序"):
        img_path = clean_image_path(result["image_path"])
        resnet_feature = compute_resnet_features(img_path, resnet_model, device)
        
        if resnet_feature is not None:
            # 计算L2距离
            distance = np.linalg.norm(query_resnet_feature - resnet_feature)
            
            # 添加到结果 - 保留CLIP相似度用于参考
            rerank_results.append({
                "clip_similarity": result["similarity"],
                "resnet_distance": float(distance),
                "image_path": img_path,
                "source_dataset": result.get("source_dataset", "unknown")  # 保留数据集来源信息
            })
    
    # 按ResNet距离排序
    rerank_results.sort(key=lambda x: x["resnet_distance"])
    
    # 为结果添加排名并转换为与clip_retrieval.py一致的格式
    formatted_results = []
    for i, result in enumerate(rerank_results):
        formatted_results.append({
            "rank": i + 1,
            "similarity": float(1.0 / (1.0 + result["resnet_distance"])),  # 将距离转换为相似度
            "image_path": result["image_path"],
            "source_dataset": result["source_dataset"]  # 包含数据集来源
        })
    
    return formatted_results

# 加载或计算COCO特征
def load_or_compute_coco_features(args, device, clip_model, clip_preprocess):
    """加载或计算COCO数据集的CLIP特征"""
    coco_features = None
    coco_paths = None
    
    # 如果强制重新计算，则跳过所有加载特征的步骤
    if args.force_recompute:
        print("强制重新计算COCO特征...")
    else:
        # 1. 首先尝试加载全局预提取特征（如果指定）
        if args.global_features:
            # 尝试多个可能的全局特征文件路径
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coco_embeddings_global.pt"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_clip_vision", "coco_embeddings_global.pt")
            ]
            
            found_global_file = False
            for global_features_path in possible_paths:
                if os.path.exists(global_features_path):
                    print(f"尝试加载全局预提取特征: {global_features_path}")
                    try:
                        # 使用weights_only=False加载以解决兼容性问题
                        data = torch.load(global_features_path, map_location=device, weights_only=False)
                        if isinstance(data, dict) and 'embeddings' in data and 'image_paths' in data:
                            coco_features = data['embeddings']
                            coco_paths = data['image_paths']
                            # 清理所有路径
                            coco_paths = [clean_image_path(path) for path in coco_paths]
                            print(f"成功从全局特征文件加载并清理 {len(coco_features)} 个COCO特征")
                            found_global_file = True
                            break
                        else:
                            print(f"文件 {global_features_path} 格式不正确，尝试其他解析方式...")
                            # 尝试其他可能的格式
                            if isinstance(data, dict):
                                for key in data:
                                    print(f"文件包含键: {key}")
                                if 'features' in data and 'image_paths' in data:
                                    coco_features = data['features']
                                    coco_paths = data['image_paths']
                                    # 清理所有路径
                                    coco_paths = [clean_image_path(path) for path in coco_paths]
                                    print(f"使用替代键名成功加载并清理 {len(coco_features)} 个COCO特征")
                                    found_global_file = True
                                    break
                    except Exception as e:
                        print(f"加载全局特征文件 {global_features_path} 时出错: {e}")
            
            if not found_global_file:
                print("无法找到或加载全局特征文件")
        
        # 2. 尝试加载指定的预提取特征
        if coco_features is None and args.pretrained_coco_features is not None:
            if os.path.exists(args.pretrained_coco_features):
                print(f"加载预提取特征: {args.pretrained_coco_features}")
                try:
                    # 判断文件类型并加载
                    if args.pretrained_coco_features.endswith('.pt'):
                        # 使用weights_only=False加载以解决兼容性问题
                        data = torch.load(args.pretrained_coco_features, map_location=device, weights_only=False)
                        # 尝试不同的键名
                        if isinstance(data, dict):
                            if 'embeddings' in data:
                                coco_features = data['embeddings']
                            elif 'features' in data:
                                coco_features = data['features']
                            else:
                                print("警告：特征文件中找不到embeddings或features键")
                                coco_features = data  # 直接使用整个数据
                        else:
                            coco_features = data
                    elif args.pretrained_coco_features.endswith('.npy'):
                        coco_features = np.load(args.pretrained_coco_features)
                        # 在.npy格式下，初始化data为None，避免后续引用错误
                        data = None
                        print(f"从.npy文件成功加载特征形状: {coco_features.shape}")
                        
                    # 尝试从特征文件中提取路径
                    if data is not None and isinstance(data, dict) and ('image_paths' in data or 'paths' in data):
                        if 'image_paths' in data:
                            coco_paths = data['image_paths']
                        else:
                            coco_paths = data['paths']
                        print(f"从特征文件中提取了 {len(coco_paths)} 个图像路径")
                        
                        # 清理所有路径
                        coco_paths = [clean_image_path(path) for path in coco_paths]
                        print(f"已清理 {len(coco_paths)} 个图像路径")
                    else:
                        print("警告：未从特征文件中提取到路径，需要单独的路径文件")
                        # 确保coco_paths被初始化，避免后续引用错误
                        if coco_paths is None:
                            coco_paths = []
                    
                    # 加载图像路径
                    if args.pretrained_coco_paths and os.path.exists(args.pretrained_coco_paths):
                        with open(args.pretrained_coco_paths, 'r') as f:
                            coco_paths = json.load(f)
                        print(f"从 {args.pretrained_coco_paths} 加载了 {len(coco_paths)} 个图像路径")
                        
                        # 清理所有路径
                        coco_paths = [clean_image_path(path) for path in coco_paths]
                        print(f"已清理 {len(coco_paths)} 个图像路径")
                        
                    if coco_features is not None:
                        print(f"成功加载 {len(coco_features)} 个预提取特征")
                except Exception as e:
                    print(f"加载预提取特征时出错: {e}")
                    coco_features = None
            else:
                print(f"警告：指定的预提取特征文件不存在: {args.pretrained_coco_features}")
        
        # 3. 尝试从本地缓存加载
        if coco_features is None:
            coco_features_file = os.path.join(RESULTS_DIR, "coco_clip_features.npy")
            coco_paths_file = os.path.join(RESULTS_DIR, "coco_image_paths.json")
            
            if os.path.exists(coco_features_file) and os.path.exists(coco_paths_file):
                print(f"从本地缓存加载COCO特征...")
                try:
                    coco_features = np.load(coco_features_file)
                    with open(coco_paths_file, "r") as f:
                        coco_paths = json.load(f)
                    # 清理所有路径
                    coco_paths = [clean_image_path(path) for path in coco_paths]
                    print(f"成功从本地缓存加载并清理 {len(coco_features)} 个COCO特征")
                except Exception as e:
                    print(f"加载本地缓存特征时出错: {e}")
                    coco_features = None
    
    # 4. 如果以上方法都失败，则重新计算特征
    if coco_features is None or coco_paths is None or len(coco_features) == 0 or len(coco_paths) == 0 or args.force_recompute:
        print("重新计算COCO特征...")
        coco_features, coco_paths = compute_coco_clip_features(args.coco_dir, clip_model, clip_preprocess, device)
        
        if len(coco_features) > 0:
            print(f"保存 {len(coco_features)} 个COCO特征到本地缓存...")
            coco_features_file = os.path.join(RESULTS_DIR, "coco_clip_features.npy")
            coco_paths_file = os.path.join(RESULTS_DIR, "coco_image_paths.json")
            
            # 清理所有图像路径
            clean_coco_paths = [clean_image_path(path) for path in coco_paths]
            
            np.save(coco_features_file, coco_features)
            with open(coco_paths_file, "w") as f:
                json.dump(clean_coco_paths, f)
            
            # 更新路径列表
            coco_paths = clean_coco_paths
    
    if coco_features is None or len(coco_features) == 0:
        print("警告：没有COCO特征")
        return None, None
        
    return coco_features, coco_paths

# 加载或计算Mini ImageNet特征
def load_or_compute_mini_imagenet_features(args, device, clip_model, clip_preprocess):
    """加载或计算Mini ImageNet数据集的CLIP特征"""
    mini_imagenet_features = None
    mini_imagenet_paths = None
    
    # 如果强制重新计算，则跳过所有加载特征的步骤
    if args.force_recompute:
        print("强制重新计算Mini ImageNet特征...")
    else:
        # 1. 尝试加载指定的预提取特征
        if args.pretrained_mini_imagenet_features is not None:
            if os.path.exists(args.pretrained_mini_imagenet_features):
                print(f"加载预提取Mini ImageNet特征: {args.pretrained_mini_imagenet_features}")
                try:
                    # 判断文件类型并加载
                    if args.pretrained_mini_imagenet_features.endswith('.pt'):
                        # 使用weights_only=False加载以解决兼容性问题
                        data = torch.load(args.pretrained_mini_imagenet_features, map_location=device, weights_only=False)
                        # 尝试不同的键名
                        if isinstance(data, dict):
                            if 'embeddings' in data:
                                mini_imagenet_features = data['embeddings']
                            elif 'features' in data:
                                mini_imagenet_features = data['features']
                            else:
                                print("警告：特征文件中找不到embeddings或features键")
                                mini_imagenet_features = data  # 直接使用整个数据
                        else:
                            mini_imagenet_features = data
                    elif args.pretrained_mini_imagenet_features.endswith('.npy'):
                        mini_imagenet_features = np.load(args.pretrained_mini_imagenet_features)
                        # 在.npy格式下，初始化data为None，避免后续引用错误
                        data = None
                        print(f"从.npy文件成功加载Mini ImageNet特征形状: {mini_imagenet_features.shape}")
                        
                    # 尝试从特征文件中提取路径
                    if data is not None and isinstance(data, dict) and ('image_paths' in data or 'paths' in data):
                        if 'image_paths' in data:
                            mini_imagenet_paths = data['image_paths']
                        else:
                            mini_imagenet_paths = data['paths']
                        print(f"从特征文件中提取了 {len(mini_imagenet_paths)} 个Mini ImageNet图像路径")
                        
                        # 清理所有路径
                        mini_imagenet_paths = [clean_image_path(path) for path in mini_imagenet_paths]
                        print(f"已清理 {len(mini_imagenet_paths)} 个Mini ImageNet图像路径")
                    else:
                        print("警告：未从特征文件中提取到路径，需要单独的路径文件")
                        # 确保mini_imagenet_paths被初始化，避免后续引用错误
                        if mini_imagenet_paths is None:
                            mini_imagenet_paths = []
                    
                    # 加载图像路径
                    if args.pretrained_mini_imagenet_paths and os.path.exists(args.pretrained_mini_imagenet_paths):
                        with open(args.pretrained_mini_imagenet_paths, 'r') as f:
                            mini_imagenet_paths = json.load(f)
                        print(f"从 {args.pretrained_mini_imagenet_paths} 加载了 {len(mini_imagenet_paths)} 个Mini ImageNet图像路径")
                        
                        # 清理所有路径
                        mini_imagenet_paths = [clean_image_path(path) for path in mini_imagenet_paths]
                        print(f"已清理 {len(mini_imagenet_paths)} 个Mini ImageNet图像路径")
                        
                    if mini_imagenet_features is not None:
                        print(f"成功加载 {len(mini_imagenet_features)} 个预提取Mini ImageNet特征")
                except Exception as e:
                    print(f"加载预提取Mini ImageNet特征时出错: {e}")
                    mini_imagenet_features = None
            else:
                print(f"警告：指定的预提取Mini ImageNet特征文件不存在: {args.pretrained_mini_imagenet_features}")
        
        # 2. 尝试从本地缓存加载
        if mini_imagenet_features is None:
            mini_imagenet_features_file = os.path.join(RESULTS_DIR, "mini_imagenet_clip_features.npy")
            mini_imagenet_paths_file = os.path.join(RESULTS_DIR, "mini_imagenet_image_paths.json")
            
            if os.path.exists(mini_imagenet_features_file) and os.path.exists(mini_imagenet_paths_file):
                print(f"从本地缓存加载Mini ImageNet特征...")
                try:
                    mini_imagenet_features = np.load(mini_imagenet_features_file)
                    with open(mini_imagenet_paths_file, "r") as f:
                        mini_imagenet_paths = json.load(f)
                    # 清理所有路径
                    mini_imagenet_paths = [clean_image_path(path) for path in mini_imagenet_paths]
                    print(f"成功从本地缓存加载并清理 {len(mini_imagenet_features)} 个Mini ImageNet特征")
                except Exception as e:
                    print(f"加载本地缓存Mini ImageNet特征时出错: {e}")
                    mini_imagenet_features = None
    
    # 3. 如果以上方法都失败，则重新计算特征
    if mini_imagenet_features is None or mini_imagenet_paths is None or len(mini_imagenet_features) == 0 or len(mini_imagenet_paths) == 0 or args.force_recompute:
        print("重新计算Mini ImageNet特征...")
        mini_imagenet_features, mini_imagenet_paths = compute_mini_imagenet_clip_features(args.mini_imagenet_dir, clip_model, clip_preprocess, device)
        
        if len(mini_imagenet_features) > 0:
            print(f"保存 {len(mini_imagenet_features)} 个Mini ImageNet特征到本地缓存...")
            mini_imagenet_features_file = os.path.join(RESULTS_DIR, "mini_imagenet_clip_features.npy")
            mini_imagenet_paths_file = os.path.join(RESULTS_DIR, "mini_imagenet_image_paths.json")
            
            # 清理所有图像路径
            clean_mini_imagenet_paths = [clean_image_path(path) for path in mini_imagenet_paths]
            
            np.save(mini_imagenet_features_file, mini_imagenet_features)
            with open(mini_imagenet_paths_file, "w") as f:
                json.dump(clean_mini_imagenet_paths, f)
            
            # 更新路径列表
            mini_imagenet_paths = clean_mini_imagenet_paths
    
    if mini_imagenet_features is None or len(mini_imagenet_features) == 0:
        print("警告：没有Mini ImageNet特征")
        return None, None
        
    return mini_imagenet_features, mini_imagenet_paths

# 按类别分组的检索函数（支持多数据源）
def retrieve_by_category_multi_source(dataset_name, shot_count, clip_model, clip_preprocess, resnet_model, dataset_features, dataset_paths, device, force_recompute_inpainted=False):
    """为数据集中的每个类别进行检索（支持多数据源）"""
    # 获取inpainted图像和类别信息
    sample_to_image, sample_to_category = get_inpainted_images(dataset_name, shot_count)
    if not sample_to_image:
        print(f"错误：找不到数据集 {dataset_name} 的 {shot_count}_shot inpainted图像")
        return None
    
    # 反转sample_to_category创建类别到样本的映射
    category_to_samples = {}
    for sample_id, category in sample_to_category.items():
        if category not in category_to_samples:
            category_to_samples[category] = []
        category_to_samples[category].append(sample_id)
    
    print(f"为数据集 {dataset_name} ({shot_count}_shot) 找到 {len(category_to_samples)} 个类别")
    
    # 确保输出目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 获取当前数据集的inpainted图像特征
    inpainted_features_file = os.path.join(RESULTS_DIR, f"{dataset_name}_{shot_count}_shot_inpainted_clip_features.npy")
    inpainted_paths_file = os.path.join(RESULTS_DIR, f"{dataset_name}_{shot_count}_shot_inpainted_image_paths.json")
    
    inpainted_features = None
    inpainted_paths = None
    
    # 如果不强制重新计算，尝试加载缓存的inpainted特征
    if not force_recompute_inpainted and os.path.exists(inpainted_features_file) and os.path.exists(inpainted_paths_file):
        try:
            print(f"从缓存加载{dataset_name} {shot_count}_shot的inpainted特征...")
            inpainted_features = np.load(inpainted_features_file)
            with open(inpainted_paths_file, "r") as f:
                inpainted_paths = json.load(f)
            print(f"成功加载{len(inpainted_features)}个inpainted特征")
        except Exception as e:
            print(f"加载inpainted特征时出错: {e}")
            inpainted_features = None
            inpainted_paths = None
    
    # 如果没有缓存的特征或强制重新计算，计算并缓存它们
    if inpainted_features is None or inpainted_paths is None or force_recompute_inpainted:
        print(f"计算{dataset_name} {shot_count}_shot的inpainted特征...")
        inpainted_features, inpainted_paths = compute_inpainted_clip_features(dataset_name, shot_count, clip_model, clip_preprocess, device)
        
        if len(inpainted_features) > 0:
            print(f"保存{len(inpainted_features)}个inpainted特征到缓存...")
            np.save(inpainted_features_file, inpainted_features)
            with open(inpainted_paths_file, "w") as f:
                json.dump(inpainted_paths, f)
    
    # 按类别处理
    all_results = {}
    
    for category_name, samples in tqdm(category_to_samples.items(), desc=f"处理 {dataset_name} ({shot_count}_shot) 的类别"):
        print(f"\n处理类别: {category_name}")
        print(f"找到 {len(samples)} 个类别 {category_name} 的样本")
        
        # 类别的所有结果
        category_results = []
        
        # 使用图像进行检索
        for sample_id in samples:
            image_path = sample_to_image[sample_id]
            
            # 提取整体图像特征
            image_feature = extract_clip_global_features(image_path, clip_preprocess, clip_model, device)
            
            if image_feature is None:
                print(f"警告：无法从 {image_path} 提取CLIP特征")
                continue
            
            # 使用图像特征作为查询
            # 第一阶段：CLIP检索（从所有可用数据集中检索）
            clip_results = clip_first_stage_retrieval(
                image_feature, 
                dataset_features,  # 使用多数据集特征字典
                dataset_paths,     # 使用多数据集路径字典
                top_k=100
            )
            
            if not clip_results:
                print(f"警告：CLIP检索未返回结果")
                continue
                
            # 第二阶段：ResNet重排序
            final_results = resnet_second_stage_rerank(image_path, clip_results, resnet_model, device)
            
            if not final_results:
                print(f"警告：ResNet重排序未返回结果")
                continue
            
            # 保存单个查询的结果
            result_file = os.path.join(RESULTS_DIR, f"{dataset_name}_{shot_count}_shot_{category_name}_{sample_id}_retrieval_results.json")
            
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            print(f"已保存样本 {sample_id} 的检索结果到 {result_file}")
            
            # 可视化前10个结果
            visualize_results(image_path, [r["image_path"] for r in final_results[:10]], 
                             os.path.join(RESULTS_DIR, f"{dataset_name}_{shot_count}_shot_{category_name}_{sample_id}_visual.jpg"))
            
            # 添加到类别结果
            query_result = {
                "sample_id": sample_id,
                "image_path": image_path,
                "category": category_name,
                "similar_images": final_results
            }
            category_results.append(query_result)
        
        # 将此类别的结果添加到all_results
        if category_results:
            if category_name not in all_results:
                all_results[category_name] = []
            all_results[category_name].extend(category_results)
    
    # 保存所有类别的结果
    all_results_file = os.path.join(RESULTS_DIR, f"{dataset_name}_{shot_count}_shot_retrieval_results.json")
    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"{dataset_name} {shot_count}_shot 所有类别的检索结果已合并保存到 {all_results_file}")
    return all_results 

# 计算Mini ImageNet图像的CLIP特征
def compute_mini_imagenet_clip_features(mini_imagenet_dir, model, preprocess, device):
    """计算Mini ImageNet数据集的CLIP特征"""
    global TERMINATE
    
    # 确定Mini ImageNet图像目录
    mini_imagenet_images_dir = os.path.join(mini_imagenet_dir, "train")
    if not os.path.exists(mini_imagenet_images_dir) or not os.path.isdir(mini_imagenet_images_dir):
        print(f"错误：找不到Mini ImageNet图像目录，已检查 {mini_imagenet_images_dir}")
        return [], []
    
    print(f"使用Mini ImageNet图像目录: {mini_imagenet_images_dir}")
    
    # 获取所有图像文件
    image_paths = []
    # 遍历所有子文件夹
    for class_dir in os.listdir(mini_imagenet_images_dir):
        class_path = os.path.join(mini_imagenet_images_dir, class_dir)
        if os.path.isdir(class_path):
            # 获取该类别下的所有jpg文件
            for img_ext in ["*.jpg", "*.jpeg", "*.png"]:
                found_files = list(Path(class_path).glob(img_ext))
                image_paths.extend([str(p) for p in found_files])
    
    if not image_paths:
        print(f"错误：在 {mini_imagenet_images_dir} 中找不到任何图像")
        return [], []
    
    print(f"找到 {len(image_paths)} 张Mini ImageNet图像")
    
    # 计算特征
    features = []
    valid_paths = []
    
    for i, img_path in enumerate(tqdm(image_paths, desc="提取Mini ImageNet CLIP特征")):
        # 检查是否应该终止
        if TERMINATE or (i > 0 and i % 10 == 0 and TERMINATE):
            print(f"正在终止Mini ImageNet特征提取，已完成 {i}/{len(image_paths)} 张图像")
            break
            
        try:
            # 清理图像路径
            img_path = clean_image_path(img_path)
            
            image = Image.open(img_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model.encode_image(image_tensor)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                
            features.append(feature.cpu().numpy()[0])
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"处理图像时出错 {img_path}: {e}")
            continue
        
        # 每1000张图像保存一次进度
        if i > 0 and i % 1000 == 0:
            print(f"已处理 {i}/{len(image_paths)} 张图像")
    
    return np.array(features), valid_paths

# 主函数
def main():
    parser = argparse.ArgumentParser(description="CLIP+ResNet图像检索 - 多shot版本")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ArTaxOr", "DIOR", "FISH", "NEU-DET", "UODD", "clipart1k"],
                        help="要处理的数据集列表")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10],
                        help="要处理的shot数量列表")
    parser.add_argument("--coco-dir", type=str, default="./coco",
                        help="COCO数据集路径")
    parser.add_argument("--mini-imagenet-dir", type=str, default="./miniimagenet",
                        help="Mini ImageNet数据集路径")
    parser.add_argument("--dataset-source", type=str, choices=["coco", "mini-imagenet", "both"], default="coco",
                        help="用于检索的数据集源 (coco, mini-imagenet, both)")
    parser.add_argument("--clip-top-k", type=int, default=100,
                        help="CLIP第一阶段检索的图像数量")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果保存目录")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="可见GPU设备的ID (当使用CUDA_VISIBLE_DEVICES时通常为0)")
    parser.add_argument("--pretrained-coco-features", type=str, default="./coco_embeddings_global.pt",
                        help="预提取的COCO特征文件路径(.pt或.npy文件)")
    parser.add_argument("--pretrained-coco-paths", type=str, default=None,
                        help="预提取的COCO图像路径文件(.json文件)")
    parser.add_argument("--pretrained-mini-imagenet-features", type=str, default=None,
                        help="预提取的Mini ImageNet特征文件路径(.pt或.npy文件)")
    parser.add_argument("--pretrained-mini-imagenet-paths", type=str, default=None,
                        help="预提取的Mini ImageNet图像路径文件(.json文件)")
    parser.add_argument("--global-features", action="store_true",
                        help="使用全局预提取特征文件")
    parser.add_argument("--force-recompute", action="store_true", 
                        help="强制重新计算数据集特征，忽略缓存")
    parser.add_argument("--lamainpaint-dir", type=str, default=None,
                        help="lamainpaint目录路径，不指定则使用默认路径")
    parser.add_argument("--force-recompute-inpainted", action="store_true",
                        help="强制重新计算inpainted图像特征，忽略缓存")
    
    args = parser.parse_args()
    
    # 更新结果目录
    global RESULTS_DIR, LAMAINPAINT_DIR
    if args.output_dir:
        RESULTS_DIR = args.output_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 更新lamainpaint目录
    if args.lamainpaint_dir:
        LAMAINPAINT_DIR = args.lamainpaint_dir
        print(f"使用指定的lamainpaint目录: {LAMAINPAINT_DIR}")
    
    # 设置设备
    device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"使用设备: {device}")
    
    # 初始化模型
    clip_model, clip_preprocess = init_clip_model(device)
    resnet_model = init_resnet_model(device)
    
    # 初始化数据集特征和路径字典
    dataset_features = {}
    dataset_paths = {}
    
    # 根据dataset_source决定加载哪些数据集
    load_coco = args.dataset_source in ["coco", "both"]
    load_mini_imagenet = args.dataset_source in ["mini-imagenet", "both"]
    
    # 加载COCO特征（如果需要）
    if load_coco:
        coco_features, coco_paths = load_or_compute_coco_features(args, device, clip_model, clip_preprocess)
        if coco_features is not None and len(coco_features) > 0:
            dataset_features["coco"] = coco_features
            dataset_paths["coco"] = coco_paths
    
    # 加载Mini ImageNet特征（如果需要）
    if load_mini_imagenet:
        mini_imagenet_features, mini_imagenet_paths = load_or_compute_mini_imagenet_features(args, device, clip_model, clip_preprocess)
        if mini_imagenet_features is not None and len(mini_imagenet_features) > 0:
            dataset_features["mini-imagenet"] = mini_imagenet_features
            dataset_paths["mini-imagenet"] = mini_imagenet_paths
    
    # 检查是否有可用的数据集特征
    if not dataset_features:
        print("错误：没有可用的数据集特征，无法进行检索")
        return
    
    # 所有shot和数据集的汇总结果
    all_shots_results = {}
    
    # 处理每个数据集
    for dataset_name in args.datasets:
        all_shots_results[dataset_name] = {}
        
        # 处理每个shot
        for shot_count in args.shots:
            print(f"\n====== 处理数据集: {dataset_name}, {shot_count}_shot ======")
            
            # 检查shot目录是否存在
            shot_dir = os.path.join(LAMAINPAINT_DIR, dataset_name, f"{shot_count}_shot")
            if not os.path.exists(shot_dir) or not os.path.isdir(shot_dir):
                print(f"警告：找不到数据集 {dataset_name} 的 {shot_count}_shot 目录: {shot_dir}")
                print(f"跳过数据集 {dataset_name} 的 {shot_count}_shot")
                continue
            
            # 检查shot目录中是否有jpg文件
            jpg_files = glob.glob(os.path.join(shot_dir, "*.jpg"))
            if not jpg_files:
                print(f"警告：在 {shot_dir} 中找不到任何jpg图像")
                print(f"跳过数据集 {dataset_name} 的 {shot_count}_shot")
                continue
            
            # 按类别进行检索
            dataset_results = retrieve_by_category_multi_source(
                dataset_name, 
                shot_count,
                clip_model, 
                clip_preprocess, 
                resnet_model, 
                dataset_features,
                dataset_paths,
                device,
                args.force_recompute_inpainted
            )
            
            if dataset_results:
                all_shots_results[dataset_name][f"{shot_count}_shot"] = dataset_results
                print(f"完成数据集 {dataset_name} 的 {shot_count}_shot 检索")
            else:
                print(f"数据集 {dataset_name} 的 {shot_count}_shot 检索失败")
    
    # 保存所有数据集和所有shot的汇总结果
    if any(all_shots_results.values()):
        all_shots_results_file = os.path.join(RESULTS_DIR, "all_shots_retrieval_results.json")
        with open(all_shots_results_file, "w", encoding="utf-8") as f:
            json.dump(all_shots_results, f, indent=2, ensure_ascii=False)
        print(f"所有数据集和所有shot的检索结果已合并保存到 {all_shots_results_file}")
        print("=============================================================")
        print("图像检索完成！结果包括:")
        print(f"1. 使用以下数据源: {args.dataset_source}")
        print(f"2. 使用CLIP特征进行第一阶段检索 (top {args.clip_top_k})")
        print(f"3. 使用ResNet风格特征进行第二阶段重排序")
        print(f"4. 处理了以下shot数量: {args.shots}")
        print(f"所有检索结果已保存到: {RESULTS_DIR}")
        print("=============================================================")
    else:
        print("没有成功检索任何数据集")

if __name__ == "__main__":
    main() 