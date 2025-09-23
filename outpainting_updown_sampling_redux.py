import os
import sys
import json
import glob
import shutil
from datetime import datetime
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import FluxPriorReduxPipeline, FluxFillPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.utils import load_image
from tqdm import tqdm
import random
import uuid
import socket
import argparse
import math
import time
import traceback
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch.multiprocessing as mp
from multiprocessing import Queue, Manager
import threading

# 设置设备和数据类型
device = "cuda"
dtype = torch.bfloat16

# 为每个数据集设置strength参数
strength_params = {
    "FISH": 0.8,
    "DIOR": 0.8,
    "ArTaxOr": 0.9,
    "UODD": 0.4,
    "NEU-DET": 0.3,
    # "NEU-DET": 0.8,
    "clipart1k": 0.9,
    "NWPU_VHR-10": 0.8,  # 添加NWPU_VHR-10，与DIOR相同的参数
    "Camouflage": 0.6,  # 可自定义提示词
    "coco": 0.8,  # 添加COCO数据集参数
}

# 为每个数据集设置guidance_scale参数
guidance_scale_params = {
    "FISH": 35.0,
    "DIOR": 30.0,
    "ArTaxOr": 30.0,
    "UODD": 30.0,
    # "NEU-DET": 20.0,
    "NEU-DET": 30.0,
    "clipart1k": 40.0,
    "NWPU_VHR-10": 30.0,  # 添加NWPU_VHR-10，与DIOR相同的参数
    "Camouflage": 30.0,  # 可自定义提示词
    "coco": 30.0,  # 添加COCO数据集参数
}

# 为每个数据集设置image_prompt_scale参数
image_prompt_scale_params = {
    "FISH": 1.2,
    "DIOR": 1.0,
    "ArTaxOr": 1.0,
    "UODD": 1,
    "NEU-DET": 1.0,
    "clipart1k": 1.0,
    "NWPU_VHR-10": 1.0,
    "Camouflage": 1.0,  # 可自定义提示词
    "coco": 1.0,  # 添加COCO数据集参数
}

# 为每个数据集设置上采样目标维度参数
upscale_dimension_params = {
    "FISH": 1024,
    "DIOR": 1024,
    "ArTaxOr": 1024,
    "UODD": 2048,  # UODD数据集上采样到至少2048像素
    "NEU-DET": 1024,
    "clipart1k": 1024,
    "NWPU_VHR-10": 1024,
    "Camouflage": 1024,  # 可自定义上采样维度
    "coco": 1024,  # 添加COCO数据集参数
}

# 为每个数据集设置redux提示词参数
redux_prompt_params = {
    "FISH": "wihout fish, A crystal-clear underwater environment, crisp and in sharp focus, foreground clarity is high; natural lighting and color continuity.",
    "DIOR": "",
    "ArTaxOr": "",
    "UODD": "",
    "NEU-DET": "",
    "clipart1k": "",
    "NWPU_VHR-10": "",
    "Camouflage": "",  # 可自定义提示词
    "coco": "",  # 添加COCO数据集参数
}

# 默认strength值
default_strength = 0.75

# 默认guidance_scale值
default_guidance_scale = 30.0

# 高分辨率处理参数
MIN_DIMENSION = 1024  # 默认最小目标维度
MAX_DIMENSION = 2800  # 最大目标维度
HR_MIN_DIMENSION = 1024  # 高分辨率处理的默认最小目标维度
UPSCALE_METHOD = Image.BICUBIC  # 上采样方法
DOWNSCALE_METHOD = Image.BICUBIC  # 下采样方法

# 路径配置
local_path = "./model"
repo_redux = local_path + "/FLUX.1-Redux-dev"
repo_base_fill = local_path + "/FLUX.1-Fill-dev"
result_dir = "./result"
datasets_dir = "./datasets"  # 数据集根目录

# 数据集列表
datasets_1 = [
    "clipart1k",
    "NEU-DET",
    "ArTaxOr",
    "coco",  # 添加COCO数据集
]

datasets_2 = [
    "FISH",
    "UODD",
    "DIOR",
    # "NWPU_VHR-10",  # 添加NWPU_VHR-10到datasets_2组
    # "Camouflage",   # 添加Camouflage到datasets_2组
]

# 要处理的数据集
datasets_to_process = datasets_1 + datasets_2

# 所有数据集
all_datasets = datasets_1 + datasets_2

# 生成唯一的进程ID
def generate_process_id():
    """生成唯一的进程ID，使用主机名、时间戳和随机字符串"""
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = str(uuid.uuid4())[:8]
    return f"{hostname}_{timestamp}_{random_str}"

# 设置全局进程ID
PROCESS_ID = generate_process_id()

# 多GPU并行相关配置
def get_available_gpus():
    """获取可用的GPU数量"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def split_samples_for_gpus(sample_list, num_gpus):
    """将样本列表分配给多个GPU，支持不均匀分配"""
    if num_gpus <= 1:
        return [sample_list]
    
    total_samples = len(sample_list)
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus
    
    gpu_samples = []
    start_idx = 0
    
    for gpu_id in range(num_gpus):
        # 前remainder个GPU多分配一个样本
        current_batch_size = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        gpu_samples.append(sample_list[start_idx:end_idx])
        start_idx = end_idx
    
    return gpu_samples

def create_gpu_process_id(base_process_id, gpu_id):
    """为每个GPU创建独立的进程ID"""
    return f"{base_process_id}_gpu{gpu_id}"

def gpu_worker_process(gpu_id, dataset_name, sample_ids, shot_number, progress_queue, result_queue, base_process_id):
    """GPU工作进程函数"""
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device_name = f"cuda:{gpu_id}"
        
        # 为当前GPU创建独立的进程ID
        gpu_process_id = create_gpu_process_id(base_process_id, gpu_id)
        
        # 更新全局进程ID
        global PROCESS_ID
        original_process_id = PROCESS_ID
        PROCESS_ID = gpu_process_id
        
        print(f"[GPU {gpu_id}] 开始处理 {len(sample_ids)} 个样本")
        print(f"[GPU {gpu_id}] 使用设备: {device_name}")
        print(f"[GPU {gpu_id}] 进程ID: {gpu_process_id}")
        
        # 处理分配给当前GPU的样本
        gpu_logs = []
        processed_count = 0
        failed_count = 0
        
        for i, sample_id in enumerate(sample_ids):
            try:
                # 发送进度更新
                progress_info = {
                    'gpu_id': gpu_id,
                    'current': i + 1,
                    'total': len(sample_ids),
                    'sample_id': sample_id,
                    'status': 'processing'
                }
                progress_queue.put(progress_info)
                
                # 查找对应的样本目录
                sample_dir = None
                result_folders = get_dataset_results(dataset_name, shot_number)
                
                for result_folder in result_folders:
                    try:
                        for sample_dir_name in os.listdir(result_folder):
                            sample_dir_path = os.path.join(result_folder, sample_dir_name)
                            if (os.path.isdir(sample_dir_path) and 
                                sample_dir_name not in ["__pycache__", "common_imgs"] and
                                (sample_dir_name == sample_id or sample_id in sample_dir_name)):
                                # 检查是否有必要的文件
                                if (os.path.exists(os.path.join(sample_dir_path, "target_input.png")) and
                                    glob.glob(os.path.join(sample_dir_path, "generated_image*.png"))):
                                    sample_dir = sample_dir_path
                                    break
                    except Exception as e:
                        continue
                    
                    if sample_dir:
                        break
                
                # 处理样本
                if sample_dir:
                    log_info = process_sample_hires(dataset_name, sample_id, sample_dir=sample_dir, shot_number=shot_number)
                else:
                    log_info = process_sample_hires(dataset_name, sample_id, shot_number=shot_number)
                
                gpu_logs.append(log_info)
                
                if log_info["status"] == "completed":
                    processed_count += 1
                    progress_info['status'] = 'completed'
                else:
                    failed_count += 1
                    progress_info['status'] = 'failed'
                    progress_info['error'] = log_info.get('error', 'Unknown error')
                
                progress_queue.put(progress_info)
                
            except Exception as e:
                failed_count += 1
                error_msg = f"处理样本 {sample_id} 时出错: {str(e)}"
                print(f"[GPU {gpu_id}] {error_msg}")
                
                # 创建错误日志记录
                error_log = {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "shot_number": shot_number,
                    "status": "error",
                    "error": error_msg,
                    "gpu_id": gpu_id
                }
                gpu_logs.append(error_log)
                
                # 发送错误进度更新
                progress_info = {
                    'gpu_id': gpu_id,
                    'current': i + 1,
                    'total': len(sample_ids),
                    'sample_id': sample_id,
                    'status': 'failed',
                    'error': error_msg
                }
                progress_queue.put(progress_info)
        
        # 发送完成信号
        completion_info = {
            'gpu_id': gpu_id,
            'status': 'gpu_completed',
            'processed_count': processed_count,
            'failed_count': failed_count,
            'total_samples': len(sample_ids)
        }
        progress_queue.put(completion_info)
        
        # 生成GPU特定的结果JSON
        result_json = generate_formatted_result_json(dataset_name, gpu_logs, shot_number)
        result_json['gpu_id'] = gpu_id
        result_json['gpu_process_id'] = gpu_process_id
        
        # 保存GPU特定的结果文件
        save_formatted_result_json(dataset_name, result_json, shot_number)
        
        # 将结果发送回主进程
        result_queue.put({
            'gpu_id': gpu_id,
            'logs': gpu_logs,
            'result_json': result_json,
            'processed_count': processed_count,
            'failed_count': failed_count
        })
        
        print(f"[GPU {gpu_id}] 处理完成: 成功 {processed_count}, 失败 {failed_count}")
        
    except Exception as e:
        error_msg = f"GPU {gpu_id} 工作进程出现严重错误: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # 发送错误信息
        result_queue.put({
            'gpu_id': gpu_id,
            'error': error_msg,
            'logs': [],
            'processed_count': 0,
            'failed_count': len(sample_ids) if 'sample_ids' in locals() else 0
        })
    finally:
        # 恢复原始进程ID
        if 'original_process_id' in locals():
            PROCESS_ID = original_process_id

def progress_monitor_thread(progress_queue, num_gpus, total_samples):
    """进度监控线程函数"""
    gpu_progress = {i: {'current': 0, 'total': 0, 'status': 'waiting'} for i in range(num_gpus)}
    completed_gpus = set()
    
    print(f"\n开始监控 {num_gpus} 个GPU的处理进度，总样本数: {total_samples}")
    print("=" * 80)
    
    while len(completed_gpus) < num_gpus:
        try:
            # 获取进度更新，设置超时避免阻塞
            progress_info = progress_queue.get(timeout=1.0)
            
            gpu_id = progress_info['gpu_id']
            
            if progress_info['status'] == 'gpu_completed':
                completed_gpus.add(gpu_id)
                gpu_progress[gpu_id]['status'] = 'completed'
                print(f"\n[GPU {gpu_id}] 完成处理: 成功 {progress_info['processed_count']}, 失败 {progress_info['failed_count']}")
            else:
                # 更新GPU进度
                gpu_progress[gpu_id].update({
                    'current': progress_info['current'],
                    'total': progress_info['total'],
                    'status': progress_info['status'],
                    'sample_id': progress_info.get('sample_id', ''),
                })
                
                # 显示当前进度
                if progress_info['status'] in ['completed', 'failed']:
                    status_symbol = "✓" if progress_info['status'] == 'completed' else "✗"
                    sample_id = progress_info.get('sample_id', '')
                    print(f"[GPU {gpu_id}] {status_symbol} {sample_id} ({progress_info['current']}/{progress_info['total']})")
            
            # 每10个样本显示一次总体进度
            if progress_info.get('current', 0) % 10 == 0 or progress_info['status'] == 'gpu_completed':
                display_overall_progress(gpu_progress, completed_gpus, num_gpus)
                
        except:
            # 超时或其他异常，继续监控
            continue
    
    print("\n" + "=" * 80)
    print("所有GPU处理完成！")

def display_overall_progress(gpu_progress, completed_gpus, num_gpus):
    """显示总体进度"""
    total_processed = sum(gpu_progress[i]['current'] for i in range(num_gpus))
    total_samples = sum(gpu_progress[i]['total'] for i in range(num_gpus))
    
    if total_samples > 0:
        overall_percentage = (total_processed / total_samples) * 100
        print(f"\n总体进度: {total_processed}/{total_samples} ({overall_percentage:.1f}%)")
        
        # 显示各GPU状态
        gpu_status_line = "GPU状态: "
        for i in range(num_gpus):
            if i in completed_gpus:
                status = "完成"
            else:
                current = gpu_progress[i]['current']
                total = gpu_progress[i]['total']
                if total > 0:
                    percentage = (current / total) * 100
                    status = f"{percentage:.0f}%"
                else:
                    status = "等待"
            gpu_status_line += f"GPU{i}:{status} "
        print(gpu_status_line)

def process_image_resolution(image, min_dimension=MIN_DIMENSION, max_dimension=MAX_DIMENSION):
    """
    处理图像分辨率，根据需要进行上采样或下采样
    
    参数:
    image: PIL图像对象
    min_dimension: 最小目标维度
    max_dimension: 最大目标维度（防止CUDA内存溢出）
    
    返回:
    processed_image: 处理后的图像
    up_scale_factor: 上采样比例（如果进行了上采样）
    down_scale_factor: 下采样比例（如果进行了下采样）
    需要上采样: 布尔值，表示是否进行了上采样
    需要下采样: 布尔值，表示是否进行了下采样
    """
    width, height = image.size
    max_size = max(width, height)
    min_size = min(width, height)
    
    # 同时需要上采样和下采样的情况（一个维度小于min_dimension，另一个维度大于max_dimension）
    if min_size < min_dimension and max_size > max_dimension:
        error_msg = f"图像既需要上采样又需要下采样：尺寸 {width}x{height}，最小维度 {min_size}，最大维度 {max_size}"
        print(f"错误：{error_msg}")
        raise ValueError(error_msg)
    
    # 检查是否需要上采样（任一维度小于min_dimension）
    if min_size < min_dimension:
        # 计算需要的缩放比例
        scale_w = min_dimension / width if width < min_dimension else 1.0
        scale_h = min_dimension / height if height < min_dimension else 1.0
        up_scale_factor = max(scale_w, scale_h)
        
        # 计算新尺寸
        new_width = int(width * up_scale_factor)
        new_height = int(height * up_scale_factor)
        
        # 执行上采样
        upscaled_image = image.resize((new_width, new_height), UPSCALE_METHOD)
        return upscaled_image, up_scale_factor, 1.0, True, False
    
    # 检查是否需要下采样（任一维度大于max_dimension）
    elif max_size > max_dimension:
        # 计算需要的缩放比例
        down_scale_factor = max_dimension / max_size
        
        # 计算新尺寸
        new_width = int(width * down_scale_factor)
        new_height = int(height * down_scale_factor)
        
        # 执行下采样
        downscaled_image = image.resize((new_width, new_height), DOWNSCALE_METHOD)
        return downscaled_image, 1.0, down_scale_factor, False, True
    
    # 图像尺寸已在合适范围内，不需要处理
    return image, 1.0, 1.0, False, False

def downscale_image(image, scale_factor):
    """
    按比例下采样图像
    
    参数:
    image: PIL图像对象
    scale_factor: 原始上采样使用的缩放比例
    
    返回:
    downscaled_image: 下采样后的图像
    """
    if scale_factor <= 1.0:
        return image  # 如果没有上采样，则返回原图
    
    width, height = image.size
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)
    
    return image.resize((new_width, new_height), DOWNSCALE_METHOD)

def upscale_image(image, scale_factor):
    """
    按比例上采样图像
    
    参数:
    image: PIL图像对象
    scale_factor: 上采样比例
    
    返回:
    upscaled_image: 上采样后的图像
    """
    if scale_factor <= 1.0:
        return image  # 如果不需要上采样，则返回原图
    
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return image.resize((new_width, new_height), UPSCALE_METHOD)

def load_model():
    """加载模型"""
    print("正在加载模型...")
    
    # 加载文本编码器模型
    text_encoder = CLIPTextModel.from_pretrained(
        local_path + "/FLUX.1-dev",
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        local_path + "/FLUX.1-dev",
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        local_path + "/FLUX.1-dev",
        subfolder="tokenizer",
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        local_path + "/FLUX.1-dev",
        subfolder="tokenizer_2",
    )

    # 加载FluxPriorRedux和FluxFill模型
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
        repo_redux,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=dtype
    ).to(device)

    pipe_fill = FluxFillPipeline.from_pretrained(
        repo_base_fill,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=dtype
    ).to(device)
    
    return pipe_prior_redux, pipe_fill

def load_annotation_file(dataset_name, shot_number=1):
    """
    加载数据集的注释文件
    
    参数:
    dataset_name: 数据集名称
    shot_number: shot数（1, 5或10）
    
    返回:
    annotations_data: 注释数据（JSON格式）
    """
    annotation_file = os.path.join(datasets_dir, dataset_name, "annotations", f"{shot_number}_shot.json")
    
    if not os.path.exists(annotation_file):
        print(f"警告：找不到注释文件: {annotation_file}")
        return None
    
    try:
        with open(annotation_file, 'r') as f:
            annotations_data = json.load(f)
        return annotations_data
    except Exception as e:
        print(f"读取注释文件出错: {str(e)}")
        return None

def get_bbox_and_original_image(dataset_name, sample_id, shot_number=1):
    """
    根据样本ID获取原始图像和所有bbox信息
    
    参数:
    dataset_name: 数据集名称
    sample_id: 样本ID（通常是目录名，对应图像文件名不含扩展名）
    shot_number: shot数（1, 5或10）
    
    返回:
    original_image: 原始图像
    bbox_images: 裁剪出的所有bbox图像列表
    bbox_coords_list: 所有bbox坐标列表，每个元素为 (x, y, width, height)
    image_id: 图像ID
    categories: 各个bbox对应的类别名称列表
    """
    # 加载注释文件
    annotations_data = load_annotation_file(dataset_name, shot_number)
    if not annotations_data:
        return None, None, None, None, None
    
    # 构建文件名到图像信息的映射
    filename_to_image = {}
    for img in annotations_data["images"]:
        # 去除扩展名，便于匹配
        filename_no_ext = os.path.splitext(img["file_name"])[0]
        filename_to_image[filename_no_ext] = img
    
    # 构建图像ID到图像信息的映射（用于后续查找）
    image_id_to_info = {}
    for img in annotations_data["images"]:
        image_id_to_info[img["id"]] = img
    
    # 构建类别ID到类别名称的映射
    category_id_to_name = {cat["id"]: cat["name"] for cat in annotations_data["categories"]}
    
    # 首先，尝试直接通过文件名匹配
    matched_image_info = filename_to_image.get(sample_id)
    
    if not matched_image_info:
        # 如果没有精确匹配，尝试查找包含sample_id的文件名
        for filename, info in filename_to_image.items():
            if sample_id in filename or filename in sample_id:
                matched_image_info = info
                break
                
    if not matched_image_info:
        print(f"警告：找不到与样本ID {sample_id} 匹配的图像")
        return None, None, None, None, None
    
    # 获取匹配图像的ID
    image_id = matched_image_info["id"]
    
    # 查找所有对应的注释（可能有多个bbox）
    matched_annotations = []
    for annotation in annotations_data["annotations"]:
        # 注意：image_id可能是整数或字符串，需要进行类型转换比较
        if str(annotation["image_id"]) == str(image_id):
            matched_annotations.append(annotation)
    
    if not matched_annotations:
        print(f"警告：找不到图像ID {image_id} 的注释")
        return None, None, None, None, None
    
    # 获取图像文件路径
    image_file_name = matched_image_info["file_name"]
    image_path = os.path.join(datasets_dir, dataset_name, "train", image_file_name)
    
    if not os.path.exists(image_path):
        print(f"警告：找不到图像文件: {image_path}")
        return None, None, None, None, None
    
    # 加载原始图像
    try:
        original_image = load_image_safe(image_path)
    except Exception as e:
        print(f"加载原始图像失败: {str(e)}")
        return None, None, None, None, None
    
    # 为每个bbox创建裁剪图像和类别信息
    bbox_images = []
    bbox_coords_list = []
    categories = []
    
    for annotation in matched_annotations:
        # 获取bbox坐标
        bbox_coords = annotation["bbox"]
        bbox_coords_list.append(bbox_coords)
        
        # 获取类别信息
        category_id = annotation["category_id"]
        category = category_id_to_name.get(category_id, "unknown")
        categories.append(category)
        
        # 裁剪bbox图像
        x, y, width, height = [float(coord) if isinstance(coord, str) else coord for coord in bbox_coords]
        x, y, width, height = int(x), int(y), int(width), int(height)
        
        try:
            # 确保坐标在图像范围内
            x = max(0, min(x, original_image.width - 1))
            y = max(0, min(y, original_image.height - 1))
            width = max(1, min(width, original_image.width - x))
            height = max(1, min(height, original_image.height - y))
            
            # 裁剪bbox区域
            bbox_image = original_image.crop((x, y, x + width, y + height))
            bbox_images.append(bbox_image)
        except Exception as e:
            print(f"裁剪bbox图像失败: {str(e)}")
            bbox_images.append(None)
    
    return original_image, bbox_images, bbox_coords_list, image_id, categories

def get_bbox_image_path_and_category(dataset_name, sample_id):
    """获取bbox图像路径和对应的类别名称列表（作为回退方法）"""
    # 不同数据集的类别名称
    category_map = {
        "FISH": "fish",
        "DIOR": ["Expressway-Service-area", "airplane", "airport", "baseballfield", 
                "basketballcourt", "bridge", "chimney", "dam", "golffield", 
                "groundtrackfield", "harbor", "overpass", "ship", "stadium", 
                "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"],
        "ArTaxOr": ["Araneae", "Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata"],
        "UODD": ["seacucumber", "scallop", "seaurchin"],
        "NEU-DET": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        "clipart1k": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                      "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "NWPU_VHR-10": ["airplane", "ship", "storage-tank", "baseball-diamond", "tennis-court", 
                     "basketball-court", "ground-track-field", "harbor", "bridge", "vehicle"],
        "Camouflage": ["Bat", "Bear", "Bird", "Body_Painting", "Camel", "Cat", "Crab", "Crocodile", 
                       "Deer", "Dog", "Dolphin", "Elephant", "Fish", "Fox", "Frog", "Giraffe", 
                       "Goat", "Hedgehog", "Horse", "Insect", "Kangaroo", "Leopard", "Lion", 
                       "Turtle", "Weasel", "Worm"],  # 更新了Camouflage的类别列表
        "coco": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                 "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                 "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                 "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                 "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                 "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                 "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                 "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                 "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                 "toothbrush"],  # COCO数据集的80个类别
    }
    
    # 公共路径
    bbox_crops_dir = "./bbox_crops"
    
    # 获取该数据集的类别列表
    categories = category_map.get(dataset_name, [])
    if not categories:
        print(f"警告：未知数据集 {dataset_name}，无法确定类别")
        return None, None
    
    # 如果categories是字符串（单一类别），转为列表
    if isinstance(categories, str):
        categories = [categories]
    
    # 存储找到的所有匹配的bbox路径和对应类别
    found_bbox_paths = []
    found_categories = []
    
    # 遍历所有可能的类别，查找匹配的bbox图像
    for category in categories:
        category_dir = os.path.join(bbox_crops_dir, dataset_name, category)
        if not os.path.exists(category_dir):
            continue
        
        # 查找以sample_id开头的图像文件
        bbox_files = glob.glob(os.path.join(category_dir, f"{sample_id}*.*"))
        if bbox_files:
            found_bbox_paths.extend(bbox_files)
            found_categories.extend([category] * len(bbox_files))
            continue  # 找到匹配文件后继续查找其他类别
        
        # 如果没有找到以sample_id开头的文件，尝试查找包含sample_id的文件
        bbox_files = glob.glob(os.path.join(category_dir, f"*{sample_id}*.*"))
        if bbox_files:
            found_bbox_paths.extend(bbox_files)
            found_categories.extend([category] * len(bbox_files))
    
    # 如果找到多个bbox路径，返回所有路径和类别
    if found_bbox_paths:
        return found_bbox_paths, found_categories
    
    # 如果没有找到任何匹配的bbox文件
    print(f"警告：找不到样本 {sample_id} 的bbox图像")
    return None, None

def extract_sample_prefix(bbox_path):
    """从bbox文件路径中提取样本ID前缀"""
    # 获取文件名（不含路径）
    filename = os.path.basename(bbox_path)
    # 获取第一个下划线前的部分作为样本ID前缀
    parts = filename.split('_')
    if len(parts) > 0:
        return parts[0]
    else:
        # 如果没有下划线，则使用无扩展名的文件名
        return os.path.splitext(filename)[0]

def load_image_safe(image_path):
    """安全加载图像，处理各种可能的异常情况"""
    try:
        # 使用diffusers的load_image函数加载图像
        image = load_image(image_path)
        return image
    except Exception as e:
        # 如果diffusers的load_image失败，尝试使用PIL直接加载
        try:
            print(f"diffusers.load_image失败，尝试使用PIL直接加载: {str(e)}")
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e2:
            print(f"使用PIL加载图像失败: {str(e2)}")
            raise ValueError(f"无法加载图像 {image_path}: {str(e2)}")

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_dataset_results(dataset_name, shot_number=1):
    """获取数据集的所有结果目录，根据shot数筛选"""
    # 构建对应shot数的数据集路径
    # 处理NWPU_VHR-10数据集的特殊情况（目录名中使用下划线而非连字符）
    if dataset_name == "NWPU_VHR-10":
        dataset_path_name = "NWPU_VHR_10"
    else:
        dataset_path_name = dataset_name
        
    dataset_path = os.path.join(result_dir, f"{dataset_path_name}_{shot_number}shot_retrieval")
    
    # 确认目录存在
    if not os.path.exists(dataset_path):
        print(f"警告：找不到数据集 {dataset_name} 的 {shot_number}shot 结果目录: {dataset_path}")
        return []
    
    # 查找所有结果目录（以results_开头的文件夹）
    result_folders = []
    try:
        for f in os.listdir(dataset_path):
            full_path = os.path.join(dataset_path, f)
            if os.path.isdir(full_path) and f.startswith("results_"):
                # 验证目录确实存在且可访问
                if os.path.exists(full_path) and os.access(full_path, os.R_OK):
                    result_folders.append(full_path)
                else:
                    print(f"警告：目录 {full_path} 存在问题，无法访问")
    except Exception as e:
        print(f"获取数据集 {dataset_name} 的 {shot_number}shot 结果目录时出错: {str(e)}")
    
    return result_folders

def get_outpaint_directory_path(result_path, dataset_name, sample_id, shot_number=1):
    """获取对应的outpaint目录路径，根据shot数创建不同目录"""
    # 构建新的outpaint目录路径，包含shot信息和样本级别的目录
    # 移除result_folder_name这一层
    outpaint_root = f"./outpaint_hires/process_{PROCESS_ID}"
    outpaint_path = os.path.join(outpaint_root, dataset_name, f"{shot_number}_shot", sample_id)
    
    return outpaint_path

def generate_outpaint_mask(original_image, bbox_coords_list):
    """
    生成outpaint的mask图像，黑色区域为保留（所有bbox区域），白色区域为重绘（非bbox区域）
    
    参数:
    original_image: 原始图像
    bbox_coords_list: bbox坐标列表，每个元素格式为(x, y, width, height)
    
    返回:
    mask: 生成的mask图像
    bbox_coords_list: 原始bbox坐标列表
    """
    width, height = original_image.size
    mask = Image.new("L", (width, height), 255)  # 创建白色背景图像（需要重绘的区域）
    draw = ImageDraw.Draw(mask)
    
    # 遍历所有bbox坐标
    for bbox_coords in bbox_coords_list:
        # 解析bbox坐标，注释文件中bbox格式为xywh
        x, y, w, h = bbox_coords
        
        # 计算右下角坐标
        x2 = x + w
        y2 = y + h
        
        # 确保坐标在图像范围内
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 绘制黑色矩形作为保留区域（bbox区域）
        draw.rectangle([x, y, x2, y2], fill=0)
    
    return mask, bbox_coords_list  # 返回mask和原始bbox坐标列表

def process_sample_hires(dataset_name, sample_id, category_name=None, sample_dir=None, shot_number=1):
    """
    处理单个样本，实现高分辨率版本的outpainting流程
    
    参数:
    dataset_name: 数据集名称
    sample_id: 样本ID
    category_name: 类别名称（可选）
    sample_dir: 样本目录路径（可选）
    shot_number: shot数
    
    返回:
    log_info: 日志信息，包含处理结果
    """
    # 打印处理信息
    print(f"处理样本 {sample_id} 从数据集 {dataset_name}，shot数: {shot_number}")
    start_time = time.time()
    
    # 创建样本前缀（用于文件命名），包含shot信息
    sample_prefix = f"{dataset_name}_{sample_id}_{shot_number}shot"
    
    # 日志信息初始化
    log_info = {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "sample_prefix": sample_prefix,
        "category": category_name,
        "shot_number": shot_number,
        "status": "started",
        "error": None,
        "original_resolution": None,
        "upscaled_resolution": None,
        "downscaled_resolution": None,  # 新增下采样分辨率信息
        "up_scale_factor": None,
        "down_scale_factor": None,      # 新增下采样比例信息
        "was_upscaled": False,
        "was_downscaled": False,        # 新增下采样标志
        "image_id": None,
        "original_image_size": None,
        "bbox_image_sizes": None,
        "bbox_coords_list": None,
        "outpainted_images": [],
        "process_time_seconds": 0
    }
    
    try:
        # 首先尝试从注释文件获取原始图像和bbox
        original_image, bbox_images, bbox_coords_list, image_id, categories = get_bbox_and_original_image(
            dataset_name, sample_id, shot_number
        )
        
        if original_image is None and sample_dir is not None:
            # 如果通过注释文件未找到且提供了样本目录，从样本目录获取
            print(f"无法通过注释文件获取样本 {sample_id} 的信息，尝试从样本目录加载...")
            
            # 尝试从样本目录加载原始图像
            original_image_path = os.path.join(sample_dir, "target_input.png")
            if not os.path.exists(original_image_path):
                raise ValueError(f"找不到原始图像文件: {original_image_path}")
            
            try:
                original_image = load_image_safe(original_image_path)
            except Exception as e:
                raise ValueError(f"无法加载原始图像: {str(e)}")
            
            # 尝试从样本目录获取所有bbox图像和类别
            bbox_paths, category_names = get_bbox_image_path_and_category(dataset_name, sample_id)
            if bbox_paths is None or len(bbox_paths) == 0:
                # 如果找不到bbox图像，为了演示，使用一个假设的bbox坐标（中心区域的30%）
                print(f"警告：找不到样本 {sample_id} 的bbox图像，使用默认bbox坐标")
                img_width, img_height = original_image.size
                bbox_width = int(img_width * 0.3)
                bbox_height = int(img_height * 0.3)
                bbox_x = (img_width - bbox_width) // 2
                bbox_y = (img_height - bbox_height) // 2
                bbox_coords = [bbox_x, bbox_y, bbox_width, bbox_height]
                # 裁剪默认bbox区域作为bbox图像
                bbox_image = original_image.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))
                
                # 将单个bbox转换为列表形式
                bbox_images = [bbox_image]
                bbox_coords_list = [bbox_coords]
                categories = [category_name] if category_name else ["unknown"]
            else:
                # 加载所有找到的bbox图像
                bbox_images = []
                bbox_coords_list = []
                categories = category_names or []
                
                for i, bbox_path in enumerate(bbox_paths):
                    try:
                        # 加载bbox图像
                        bbox_image = load_image_safe(bbox_path)
                        bbox_images.append(bbox_image)
                        
                        # 为了简化，这里使用图像不同位置来估计bbox坐标
                        img_width, img_height = original_image.size
                        bbox_width, bbox_height = bbox_image.size
                        
                        # 对多个bbox进行简单的排列，避免完全重叠
                        offset_x = (i % 3) * (img_width // 6)  # 水平方向上的偏移
                        offset_y = (i // 3) * (img_height // 6)  # 垂直方向上的偏移
                        
                        # 计算bbox的位置，使其分布在图像的不同区域
                        center_x = (img_width // 2 - bbox_width // 2) + offset_x
                        center_y = (img_height // 2 - bbox_height // 2) + offset_y
                        
                        # 确保bbox坐标在图像范围内
                        center_x = max(0, min(center_x, img_width - bbox_width))
                        center_y = max(0, min(center_y, img_height - bbox_height))
                        
                        bbox_coords = [center_x, center_y, bbox_width, bbox_height]
                        bbox_coords_list.append(bbox_coords)
                        
                        # 确保类别列表长度与bbox列表一致
                        if i >= len(categories):
                            categories.append(category_name if category_name else "unknown")
                    
                    except Exception as e:
                        print(f"加载bbox图像 {bbox_path} 失败: {str(e)}")
                        # 跳过这个bbox
                        continue
        elif original_image is None:
            # 如果无法从注释文件获取且没有提供样本目录，则尝试备选方法
            print(f"无法通过注释文件获取样本 {sample_id} 的信息，尝试备选方法...")
            bbox_paths, category_names = get_bbox_image_path_and_category(dataset_name, sample_id)
            
            if bbox_paths is None or len(bbox_paths) == 0:
                raise ValueError(f"找不到样本 {sample_id} 的bbox图像")
            
            # 从第一个bbox路径提取sample_id前缀（用于查找背景图像）
            sample_prefix_extracted = extract_sample_prefix(bbox_paths[0])
            
            # 查找对应的结果目录
            result_folders = get_dataset_results(dataset_name, shot_number)
            if not result_folders:
                raise ValueError(f"找不到数据集 {dataset_name} 的 {shot_number}shot 结果目录")
            
            # 加载所有bbox图像
            bbox_images = []
            for bbox_path in bbox_paths:
                try:
                    bbox_image = load_image_safe(bbox_path)
                    bbox_images.append(bbox_image)
                except Exception as e:
                    print(f"加载bbox图像 {bbox_path} 失败: {str(e)}")
                    # 跳过这个bbox
            
            if not bbox_images:
                raise ValueError(f"无法加载样本 {sample_id} 的任何bbox图像")
            
            # 在结果目录中查找原始背景图像
            found_bg_image = False
            for result_folder in result_folders:
                # 查找匹配的样本目录
                for sample_dir_name in os.listdir(result_folder):
                    sample_dir_path = os.path.join(result_folder, sample_dir_name)
                    if not os.path.isdir(sample_dir_path) or sample_dir_name in ["__pycache__", "common_imgs"]:
                        continue
                    
                    # 检查是否有匹配的样本ID
                    if sample_dir_name == sample_id or sample_prefix_extracted in sample_dir_name:
                        # 检查是否有target_input.png文件
                        bg_image_path = os.path.join(sample_dir_path, "target_input.png")
                        if os.path.exists(bg_image_path):
                            try:
                                original_image = load_image_safe(bg_image_path)
                                sample_dir = sample_dir_path  # 设置样本目录
                                found_bg_image = True
                                break
                            except Exception as e:
                                print(f"加载背景图像失败: {str(e)}")
                                continue
                
                if found_bg_image:
                    break
            
            if not found_bg_image or original_image is None:
                raise ValueError(f"找不到样本 {sample_id} 的背景图像")
            
            # 为每个bbox创建坐标（使用简单排列避免重叠）
            bbox_coords_list = []
            img_width, img_height = original_image.size
            
            for i, bbox_img in enumerate(bbox_images):
                bbox_width, bbox_height = bbox_img.size
                
                # 对多个bbox进行简单的排列，避免完全重叠
                offset_x = (i % 3) * (img_width // 6)  # 水平方向上的偏移
                offset_y = (i // 3) * (img_height // 6)  # 垂直方向上的偏移
                
                # 计算bbox的位置，使其分布在图像的不同区域
                center_x = (img_width // 2 - bbox_width // 2) + offset_x
                center_y = (img_height // 2 - bbox_height // 2) + offset_y
                
                # 确保bbox坐标在图像范围内
                center_x = max(0, min(center_x, img_width - bbox_width))
                center_y = max(0, min(center_y, img_height - bbox_height))
                
                bbox_coords = [center_x, center_y, bbox_width, bbox_height]
                bbox_coords_list.append(bbox_coords)
            
            # 确保类别列表长度与bbox列表一致
            categories = category_names[:len(bbox_images)]
            while len(categories) < len(bbox_images):
                categories.append(category_names[0] if category_names else "unknown")
        
        # 查找生成的图像文件（这些将作为背景图）
        if sample_dir is None:
            raise ValueError("未提供样本目录，无法查找生成的背景图像")
            
        bg_images = glob.glob(os.path.join(sample_dir, "generated_image*png"))
        if not bg_images:
            raise ValueError(f"样本 {sample_id} 没有生成图像作为背景")
        
        # 更新日志信息 - 适应多bbox情况
        log_info["category"] = categories[0] if categories else (category_name or "unknown")  # 使用第一个类别作为主类别
        log_info["categories"] = categories  # 存储所有类别
        log_info["original_resolution"] = original_image.size
        log_info["image_id"] = image_id if image_id else "unknown"
        log_info["original_image_size"] = (original_image.width, original_image.height)
        
        # 保存所有bbox的尺寸信息
        bbox_image_sizes = []
        for bbox_img in bbox_images:
            if bbox_img:
                bbox_image_sizes.append((bbox_img.width, bbox_img.height))
            else:
                bbox_image_sizes.append(None)
        log_info["bbox_image_sizes"] = bbox_image_sizes
        log_info["bbox_coords_list"] = bbox_coords_list
        
        # 获取样本对应的结果目录（如果有指定样本目录，使用其父目录）
        result_folder = os.path.dirname(sample_dir) if sample_dir else get_dataset_results(dataset_name, shot_number)[0]
        
        # 获取对应的outpaint目录（已修改为包含样本ID）
        outpaint_path = get_outpaint_directory_path(result_folder, dataset_name, sample_id, shot_number)
        # 确保目录存在
        ensure_dir_exists(outpaint_path)
        
        # 保存原始背景图像 - 使用带有样本前缀的文件名
        orig_bg_path = os.path.join(outpaint_path, f"{sample_prefix}_original.png")
        original_image.save(orig_bg_path)
        
        # 保存所有裁剪的bbox图像到outpaint目录
        bbox_saved_paths = []
        for i, bbox_image in enumerate(bbox_images):
            if bbox_image:
                bbox_saved_filename = f"{sample_prefix}_bbox{i+1}_original.jpg"
                bbox_saved_path = os.path.join(outpaint_path, bbox_saved_filename)
                try:
                    bbox_image.save(bbox_saved_path)
                    print(f"已将bbox图像{i+1}保存至: {bbox_saved_path}")
                    bbox_saved_paths.append(bbox_saved_path)
                except Exception as e:
                    print(f"保存bbox图像{i+1}失败: {str(e)}")
                    bbox_saved_paths.append(None)
            else:
                bbox_saved_paths.append(None)
        
        log_info["bbox_saved_paths"] = bbox_saved_paths
        
        # 获取数据集特定的上采样目标维度
        min_dimension = get_upscale_dimension_param(dataset_name)
        
        # 处理图像分辨率（上采样和下采样）
        try:
            processed_image, up_scale_factor, down_scale_factor, was_upscaled, was_downscaled = process_image_resolution(
                original_image, 
                min_dimension=min_dimension,  # 使用数据集特定的上采样目标维度
                max_dimension=MAX_DIMENSION
            )
        except ValueError as e:
            # 捕获图像既需要上采样又需要下采样的错误
            raise ValueError(f"样本 {sample_id} 处理失败: {str(e)}")
        
        # 更新日志信息
        log_info["upscaled_resolution"] = processed_image.size
        log_info["up_scale_factor"] = up_scale_factor
        log_info["down_scale_factor"] = down_scale_factor
        log_info["was_upscaled"] = was_upscaled
        log_info["was_downscaled"] = was_downscaled
        log_info["min_dimension_used"] = min_dimension  # 记录使用的上采样目标维度
        
        if was_downscaled:
            # 保存下采样后的背景图像
            downscaled_bg_path = os.path.join(outpaint_path, f"{sample_prefix}_downscaled_bg.png")
            processed_image.save(downscaled_bg_path)
            log_info["downscaled_resolution"] = processed_image.size
        
        if was_upscaled:
            # 保存上采样后的背景图像
            upscaled_bg_path = os.path.join(outpaint_path, f"{sample_prefix}_upscaled_bg.png")
            processed_image.save(upscaled_bg_path)
        
        # 调整所有bbox坐标以匹配处理后的图像
        processed_bbox_coords_list = []
        for bbox_coords in bbox_coords_list:
            if was_upscaled:
                # 上采样时放大坐标
                processed_coords = [int(coord * up_scale_factor) for coord in bbox_coords]
            elif was_downscaled:
                # 下采样时缩小坐标
                processed_coords = [int(coord * down_scale_factor) for coord in bbox_coords]
            else:
                # 未处理时保持原坐标
                processed_coords = bbox_coords
            processed_bbox_coords_list.append(processed_coords)
        
        # 生成outpaint的mask，现在支持多个bbox
        mask_image, adjusted_bbox_list = generate_outpaint_mask(processed_image, processed_bbox_coords_list)
        
        # 加载模型（如果尚未加载）
        pipe_prior_redux, pipe_fill = load_model()
        
        # 获取数据集对应的强度参数
        strength = get_strength_param(dataset_name)
        
        # 获取数据集对应的guidance_scale参数
        guidance_scale = get_guidance_scale_param(dataset_name)
        
        # 获取数据集对应的image_prompt_scale参数
        image_prompt_scale = get_image_prompt_scale_param(dataset_name)
        
        # 对每个背景图像进行处理
        for bg_idx, bg_path in enumerate(bg_images):
            # 提取rank信息
            bg_filename = os.path.basename(bg_path)
            rank_suffix = ""
            if "rank" in bg_filename:
                rank_suffix = f"_{bg_filename.split('rank')[1].split('.')[0]}"
            else:
                # 如果没有rank信息，使用索引
                rank_suffix = f"_{bg_idx+1}"
            
            # 保存mask图像 - 使用带有样本前缀的文件名
            mask_filename = f"{sample_prefix}_mask{rank_suffix}.png"
            mask_path = os.path.join(outpaint_path, mask_filename)
            mask_image.save(mask_path)
            
            # 加载背景图像
            try:
                bg_image = load_image_safe(bg_path)
            except Exception as e:
                print(f"加载背景图像 {bg_path} 失败: {str(e)}")
                continue
                
            # 复制背景图到outpaint目录 - 使用带有样本前缀的文件名
            bg_saved_filename = f"{sample_prefix}_bg{rank_suffix}_original.png"
            bg_saved_path = os.path.join(outpaint_path, bg_saved_filename)
            try:
                shutil.copy(bg_path, bg_saved_path)
                print(f"已将背景图像保存至: {bg_saved_path}")
            except Exception as e:
                print(f"保存背景图像失败: {str(e)}")
                # 继续执行，不中断流程
            
            # 使用当前seed
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator("cpu").manual_seed(seed)
            
            # 获取数据集对应的redux提示词
            redux_prompt = get_redux_prompt(dataset_name)
            
            # 使用FluxPriorRedux从背景图生成提示嵌入
            pipe_prior_output = pipe_prior_redux(
                [bg_image],  # 使用当前背景图像生成提示嵌入
                prompt=redux_prompt,  # 使用数据集特定的redux提示词
                prompt_2="",
                prompt_embeds_scale=[image_prompt_scale],  # 使用数据集特定的image_prompt_scale
                pooled_prompt_embeds_scale=[1.0],  # 使用默认值1.0
            )
            
            # 执行outpainting（填充）
            result_image = pipe_fill(
                image=processed_image,
                mask_image=mask_image,
                height=processed_image.height,
                width=processed_image.width,
                guidance_scale=guidance_scale,
                num_inference_steps=50,
                prompt_embeds=pipe_prior_output.prompt_embeds,
                pooled_prompt_embeds=pipe_prior_output.pooled_prompt_embeds,
                generator=generator,
                strength=strength,
            ).images[0]
            
            # 保存高分辨率结果 - 使用带有样本前缀的文件名
            hires_result_filename = f"{sample_prefix}_hires_result{rank_suffix}.png"
            hires_result_path = os.path.join(outpaint_path, hires_result_filename)
            result_image.save(hires_result_path)
            
            # 处理结果恢复回原始分辨率
            final_result = result_image
            
            # 如果进行了上采样或下采样，需要恢复到原始分辨率
            if was_upscaled:
                # 上采样后需要下采样回原始分辨率
                final_result = downscale_image(result_image, up_scale_factor)
            elif was_downscaled:
                # 下采样后需要上采样回原始分辨率
                final_result = upscale_image(result_image, 1.0 / down_scale_factor)
            
            # 保存恢复到原始分辨率的最终结果
            final_filename = f"{sample_prefix}_final_result{rank_suffix}.png"
            final_path = os.path.join(outpaint_path, final_filename)
            final_result.save(final_path)
            
            # 保存参数信息到JSON文件
            params_filename = f"{sample_prefix}_params{rank_suffix}.json"
            params_path = os.path.join(outpaint_path, params_filename)
            
            # 创建参数记录 - 更新为支持多个bbox和分辨率调整
            params_record = {
                "categories": categories,
                "image_scale": 1.0,  # 使用默认值
                "prompt_scale": 1.0,  # 使用默认值
                "image_prompt_scale": image_prompt_scale,  # 记录image_prompt_scale参数
                "guidance_scale": guidance_scale,
                "num_inference_steps": 50,
                "strength": strength,  # 记录strength参数
                "redux_prompt": redux_prompt,  # 记录redux提示词
                "seed": seed,
                "process_id": PROCESS_ID,
                "shot_number": shot_number,  # 确保包含shot信息
                "bg_index": bg_idx,
                "bg_filename": bg_filename,
                "original_bg_path": bg_path,
                "copied_bg_path": bg_saved_path,
                "original_resolution": {
                    "width": original_image.width,
                    "height": original_image.height
                },
                "processed_resolution": {
                    "width": processed_image.width,
                    "height": processed_image.height
                },
                "min_dimension_used": min_dimension,  # 记录使用的上采样目标维度
                "up_scale_factor": up_scale_factor,
                "down_scale_factor": down_scale_factor,
                "was_upscaled": was_upscaled,
                "was_downscaled": was_downscaled,
                "bbox_coords_list": bbox_coords_list,
                "processed_bbox_coords_list": processed_bbox_coords_list,
                "image_id": image_id if image_id else "unknown",
                "num_bbox": len(bbox_coords_list)  # 添加bbox数量信息
            }
            
            # 保存参数JSON
            with open(params_path, "w") as f:
                json.dump(params_record, f, indent=2)
                
            # 记录生成图像信息（添加到log_info）
            image_record = {
                "original_bg_path": bg_path,
                "copied_bg_path": bg_saved_path,
                "hires_result_path": hires_result_path,
                "final_result_path": final_path,
                "mask_path": mask_path,
                "params_path": params_path,
                "bbox_coords_list": bbox_coords_list,
                "processed_bbox_coords_list": processed_bbox_coords_list,
                "params": params_record
            }
            log_info["outpainted_images"].append(image_record)
        
        # 更新日志信息
        log_info["original_saved_path"] = orig_bg_path
        log_info["status"] = "completed"
    
    except Exception as e:
        # 记录错误信息
        log_info["status"] = "error"
        log_info["error"] = str(e)
        print(f"处理样本 {sample_id} 时出错: {str(e)}")
        traceback.print_exc()
    finally:
        # 计算处理时间
        end_time = time.time()
        process_time = end_time - start_time
        log_info["process_time_seconds"] = process_time
        
        # 打印处理结果
        if log_info["status"] == "completed":
            print(f"样本 {sample_id} 处理完成，耗时 {process_time:.2f} 秒")
        else:
            print(f"样本 {sample_id} 处理失败，耗时 {process_time:.2f} 秒")
        
        # 返回日志信息
        return log_info

def get_strength_param(dataset_name):
    """获取数据集对应的强度参数"""
    return strength_params.get(dataset_name, 0.75)  # 默认值为0.75

def get_guidance_scale_param(dataset_name):
    """获取数据集对应的guidance_scale参数"""
    return guidance_scale_params.get(dataset_name, 30.0)  # 默认值为30.0

def get_image_prompt_scale_param(dataset_name):
    """获取数据集对应的image_prompt_scale参数"""
    return image_prompt_scale_params.get(dataset_name, 1.0)  # 默认值为1.0

def get_upscale_dimension_param(dataset_name):
    """获取数据集对应的上采样目标维度参数"""
    return upscale_dimension_params.get(dataset_name, 1024)  # 默认值为1024

def get_redux_prompt(dataset_name):
    """获取数据集对应的redux提示词"""
    return redux_prompt_params.get(dataset_name, "")  # 默认值为空字符串

def generate_formatted_result_json(dataset_name, logs, shot_number=1):
    """
    生成与redux_outpaint.py类似格式的JSON结果文件
    
    参数:
    dataset_name: 数据集名称
    logs: 样本日志列表
    shot_number: shot数
    
    返回:
    result_json: 格式化的JSON结果
    """
    # 创建顶层数据结构
    result_json = {
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "process_id": PROCESS_ID,
        "shot_number": shot_number,
        "samples": []  # 直接在顶层放置samples
    }
    
    # 处理所有样本日志
    for log in logs:
        if log["status"] != "completed":
            continue
            
        # 获取第一个样本的结果目录路径（如果有）
        if not log["outpainted_images"]:
            continue
        
        # 创建样本记录
        sample_record = {
            "sample_id": log["sample_id"],
            "category": log.get("category", "unknown"),
            "categories": log.get("categories", [log.get("category", "unknown")]),  # 获取所有类别
            "sample_prefix": log["sample_prefix"],
            "process_id": PROCESS_ID,
            "shot_number": shot_number,  # 确保包含shot信息
            "image_id": log["image_id"],
            "original_image_size": log["original_image_size"],
            "bbox_image_sizes": log.get("bbox_image_sizes", []),  # 获取所有bbox尺寸
            "bbox_coords_list": log.get("bbox_coords_list", []),  # 获取所有bbox坐标
            "num_bbox": len(log.get("bbox_coords_list", [])),  # 添加bbox数量信息
            "outpainted_images": []
        }
        
        # 添加原始图像和bbox图像路径
        if "original_saved_path" in log:
            sample_record["original_saved_path"] = log["original_saved_path"]
        
        # 添加所有bbox图像路径
        if "bbox_saved_paths" in log:
            sample_record["bbox_saved_paths"] = log["bbox_saved_paths"]
        
        # 处理所有outpainted图像
        for img_record in log["outpainted_images"]:
            # 创建outpainted图像记录
            outpainted_image = {
                "original_bg_path": img_record["original_bg_path"],
                "copied_bg_path": img_record["copied_bg_path"],
                "outpainted_image_path": img_record["hires_result_path"],
                "final_result_path": img_record["final_result_path"],
                "mask_path": img_record["mask_path"],
                "params_path": img_record["params_path"],
                "bbox_coords_list": img_record.get("bbox_coords_list", []),  # 获取所有bbox坐标
                "shot_number": shot_number,  # 添加shot信息
                "params": img_record["params"]
            }
            sample_record["outpainted_images"].append(outpainted_image)
        
        # 直接添加样本到顶层samples数组
        result_json["samples"].append(sample_record)
    
    return result_json

def process_dataset_samples(dataset_name, sample_ids=None, shot_number=1, skip_processed=None):
    """
    处理指定数据集的样本
    
    参数:
    dataset_name: 数据集名称
    sample_ids: 要处理的样本ID列表（如果为None，则处理所有找到的样本）
    shot_number: shot数
    skip_processed: 已处理的样本ID集合，这些样本将被跳过
    
    返回:
    logs: 所有样本的日志信息列表
    """
    logs = []
    skip_processed = skip_processed or set()
    
    if sample_ids:
        # 处理指定的样本
        for sample_id in sample_ids:
            # 如果样本已处理成功，则跳过
            if sample_id in skip_processed:
                print(f"样本 {sample_id} 已成功处理过，跳过")
                continue
                
            # 处理样本
            log_info = process_sample_hires(dataset_name, sample_id, shot_number=shot_number)
            logs.append(log_info)
    else:
        # 查找并处理所有样本
        # 首先获取该数据集的结果目录
        result_folders = get_dataset_results(dataset_name, shot_number)
        if not result_folders:
            print(f"警告：找不到数据集 {dataset_name} 的 {shot_number}shot 结果目录")
            return logs
        
        # 另一种方式：从注释文件中获取样本ID列表
        annotation_file = os.path.join(datasets_dir, dataset_name, "annotations", f"{shot_number}_shot.json")
        sample_ids_from_annotations = set()
        
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                
                # 从注释文件中获取所有图像ID
                for image in annotation_data.get("images", []):
                    image_id = image.get("id")
                    if image_id:
                        file_name = image.get("file_name", "")
                        # 使用无扩展名的文件名作为样本ID
                        sample_id = os.path.splitext(file_name)[0]
                        sample_ids_from_annotations.add(sample_id)
                
                print(f"从注释文件中获取到 {len(sample_ids_from_annotations)} 个样本ID")
            except Exception as e:
                print(f"读取注释文件获取样本ID时出错: {str(e)}")
        
        # 遍历每个结果目录，查找样本
        for result_folder in result_folders:
            print(f"处理结果目录: {result_folder}")
            
            # 获取所有样本目录
            try:
                all_items = os.listdir(result_folder)
                sample_dirs = [d for d in all_items 
                              if os.path.isdir(os.path.join(result_folder, d)) and 
                              d not in ["__pycache__", "common_imgs"]]
            except Exception as e:
                print(f"获取样本目录时出错: {str(e)}")
                continue
            
            # 处理每个样本
            for sample_dir_name in sample_dirs:
                sample_id = sample_dir_name  # 使用目录名作为样本ID
                
                # 如果样本已处理成功，则跳过
                if sample_id in skip_processed:
                    print(f"样本 {sample_id} 已成功处理过，跳过")
                    continue
                    
                full_sample_path = os.path.join(result_folder, sample_dir_name)
                
                # 检查样本目录是否存在必要的文件
                if not os.path.exists(os.path.join(full_sample_path, "target_input.png")):
                    print(f"跳过样本 {sample_id}，因为找不到target_input.png文件")
                    continue
                
                if not glob.glob(os.path.join(full_sample_path, "generated_image*png")):
                    print(f"跳过样本 {sample_id}，因为找不到generated_image*.png文件")
                    continue
                
                # 如果样本ID尚未处理，则进行处理
                if sample_id not in {log["sample_id"] for log in logs}:
                    log_info = process_sample_hires(dataset_name, sample_id, sample_dir=full_sample_path, shot_number=shot_number)
                    logs.append(log_info)
            
            # 处理从注释文件中获取的样本ID（可能没有对应的结果目录）
            for sample_id in sample_ids_from_annotations:
                # 如果样本已处理成功，则跳过
                if sample_id in skip_processed:
                    print(f"样本 {sample_id} 已成功处理过，跳过")
                    continue
                    
                if sample_id not in {log["sample_id"] for log in logs}:
                    # 尝试直接使用注释文件信息处理样本
                    try:
                        log_info = process_sample_hires(dataset_name, sample_id, shot_number=shot_number)
                        if log_info["status"] == "completed":
                            logs.append(log_info)
                    except Exception as e:
                        print(f"处理注释文件中的样本 {sample_id} 时出错: {str(e)}")
    
    # 创建格式化的结果JSON
    result_json = generate_formatted_result_json(dataset_name, logs, shot_number)
    
    # 保存结果JSON
    save_formatted_result_json(dataset_name, result_json, shot_number)
    
    # 返回所有样本的日志信息列表
    return logs

def save_formatted_result_json(dataset_name, result_json, shot_number):
    """
    保存格式化的结果JSON文件
    
    参数:
    dataset_name: 数据集名称
    result_json: 结果JSON对象
    shot_number: shot数
    """
    # 确保outpaint_hires目录存在
    outpaint_root = f"./outpaint_hires/process_{PROCESS_ID}"
    # 创建包含shot信息的目录结构
    dataset_dir = os.path.join(outpaint_root, dataset_name, f"{shot_number}_shot")
    ensure_dir_exists(dataset_dir)
    
    # 创建文件名（移除时间戳）
    filename = f"outpaint_results_{shot_number}shot.json"
    file_path = os.path.join(dataset_dir, filename)
    
    # 保存JSON文件
    with open(file_path, "w") as f:
        json.dump(result_json, f, indent=2)
    
    print(f"已保存格式化的结果JSON文件: {file_path}")
    return file_path

def process_dataset_samples_multi_gpu(dataset_name, sample_ids=None, shot_number=1, num_gpus=1):
    """
    使用多GPU并行处理数据集样本
    
    参数:
    dataset_name: 数据集名称
    sample_ids: 要处理的样本ID列表
    shot_number: shot数
    num_gpus: 使用的GPU数量
    
    返回:
    all_logs: 合并后的所有样本日志信息列表
    """
    if num_gpus <= 1:
        # 单GPU模式，使用原有函数
        return process_dataset_samples(dataset_name, sample_ids, shot_number)
    
    # 获取样本列表
    if sample_ids is None:
        # 如果没有指定样本ID，从结果目录和注释文件中获取
        sample_ids = get_all_sample_ids(dataset_name, shot_number)
    
    if not sample_ids:
        print(f"警告：数据集 {dataset_name} 没有找到任何样本")
        return []
    
    print(f"开始多GPU并行处理数据集 {dataset_name}，使用 {num_gpus} 个GPU")
    print(f"总样本数: {len(sample_ids)}")
    
    # 分配样本到各个GPU
    gpu_sample_lists = split_samples_for_gpus(sample_ids, num_gpus)
    
    # 显示分配情况
    for i, samples in enumerate(gpu_sample_lists):
        print(f"GPU {i}: {len(samples)} 个样本")
    
    # 创建进程间通信队列
    progress_queue = Queue()
    result_queue = Queue()
    
    # 启动进度监控线程
    monitor_thread = threading.Thread(
        target=progress_monitor_thread,
        args=(progress_queue, num_gpus, len(sample_ids))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 启动GPU工作进程
    processes = []
    for gpu_id in range(num_gpus):
        if gpu_id < len(gpu_sample_lists) and gpu_sample_lists[gpu_id]:
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, dataset_name, gpu_sample_lists[gpu_id], shot_number, 
                      progress_queue, result_queue, PROCESS_ID)
            )
            process.start()
            processes.append(process)
    
    # 等待所有进程完成
    for process in processes:
        process.join()
    
    # 等待监控线程完成
    monitor_thread.join(timeout=5)
    
    # 收集所有GPU的结果
    all_logs = []
    all_result_jsons = []
    total_processed = 0
    total_failed = 0
    
    print("\n收集各GPU处理结果...")
    
    # 从结果队列中获取所有结果
    gpu_results = {}
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            gpu_id = result['gpu_id']
            gpu_results[gpu_id] = result
        except:
            break
    
    # 按GPU ID顺序处理结果
    for gpu_id in range(num_gpus):
        if gpu_id in gpu_results:
            result = gpu_results[gpu_id]
            
            if 'error' in result:
                print(f"GPU {gpu_id} 出现错误: {result['error']}")
                total_failed += result.get('failed_count', 0)
            else:
                all_logs.extend(result['logs'])
                all_result_jsons.append(result['result_json'])
                total_processed += result.get('processed_count', 0)
                total_failed += result.get('failed_count', 0)
                
                print(f"GPU {gpu_id}: 成功 {result.get('processed_count', 0)}, 失败 {result.get('failed_count', 0)}")
    
    print(f"\n多GPU处理完成:")
    print(f"总计 - 成功: {total_processed}, 失败: {total_failed}, 总样本数: {len(sample_ids)}")
    
    # 合并所有GPU的结果为统一的JSON
    merged_result_json = merge_gpu_results(dataset_name, all_result_jsons, shot_number)
    
    # 保存合并后的结果
    save_formatted_result_json(dataset_name, merged_result_json, shot_number)
    
    return all_logs

def get_all_sample_ids(dataset_name, shot_number):
    """获取数据集的所有样本ID"""
    sample_ids = set()
    
    # 从结果目录获取样本ID
    result_folders = get_dataset_results(dataset_name, shot_number)
    for result_folder in result_folders:
        try:
            for sample_dir_name in os.listdir(result_folder):
                sample_dir_path = os.path.join(result_folder, sample_dir_name)
                if (os.path.isdir(sample_dir_path) and 
                    sample_dir_name not in ["__pycache__", "common_imgs"] and
                    os.path.exists(os.path.join(sample_dir_path, "target_input.png"))):
                    sample_ids.add(sample_dir_name)
        except Exception as e:
            print(f"获取样本ID时出错: {str(e)}")
    
    # 从注释文件获取样本ID
    annotation_file = os.path.join(datasets_dir, dataset_name, "annotations", f"{shot_number}_shot.json")
    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            for image in annotation_data.get("images", []):
                file_name = image.get("file_name", "")
                sample_id = os.path.splitext(file_name)[0]
                sample_ids.add(sample_id)
        except Exception as e:
            print(f"从注释文件获取样本ID时出错: {str(e)}")
    
    return list(sample_ids)

def merge_gpu_results(dataset_name, gpu_result_jsons, shot_number):
    """合并多个GPU的结果JSON"""
    if not gpu_result_jsons:
        return generate_formatted_result_json(dataset_name, [], shot_number)
    
    # 使用第一个GPU的结果作为基础
    merged_result = gpu_result_jsons[0].copy()
    merged_result['samples'] = []
    merged_result['multi_gpu'] = True
    merged_result['num_gpus'] = len(gpu_result_jsons)
    merged_result['gpu_process_ids'] = []
    
    # 合并所有GPU的样本结果
    for gpu_result in gpu_result_jsons:
        merged_result['samples'].extend(gpu_result.get('samples', []))
        merged_result['gpu_process_ids'].append(gpu_result.get('gpu_process_id', ''))
    
    return merged_result

def process_all_datasets(shot_number=1):
    """
    处理所有数据集
    
    参数:
    shot_number: shot数
    
    返回:
    all_logs: 所有数据集的日志信息字典
    """
    all_logs = {}
    
    for dataset_name in datasets_to_process:
        print(f"正在处理数据集: {dataset_name}, shot数: {shot_number}")
        logs = process_dataset_samples(dataset_name, shot_number=shot_number)
        all_logs[dataset_name] = logs
    
    # 创建合并的日志文件（用于内部参考）
    save_logs(all_logs)
    
    return all_logs

def save_logs(logs, filename=None):
    """保存日志信息到文件（内部使用，作为完整记录）"""
    if filename is None:
        # 使用时间戳和进程ID作为默认文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"outpaint_hires_logs_{timestamp}.json"
    
    # 确保outpaint_hires目录存在
    outpaint_root = f"./outpaint_hires/process_{PROCESS_ID}"
    ensure_dir_exists(outpaint_root)
    
    # 完整的日志文件路径
    log_path = os.path.join(outpaint_root, filename)
    
    # 保存日志
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    
    print(f"内部日志已保存到: {log_path}")
    
    return log_path

def copy_final_results_to_collection(process_id=None, shot_number=None):
    """
    将所有数据集的最终结果复制到一个集合目录中
    
    参数:
    process_id: 要收集的特定进程ID，默认为当前进程ID
    shot_number: 指定要收集的shot数，如果为None则收集所有shot
    
    返回:
    collection_dir: 集合目录路径
    """
    if process_id is None:
        process_id = PROCESS_ID
    
    # 创建集合目录，添加shot信息到路径（如果指定）
    collection_base = f"./final_results/process_{process_id}"
    if shot_number is not None:
        collection_dir = f"{collection_base}/{shot_number}_shot"
    else:
        collection_dir = collection_base
    ensure_dir_exists(collection_dir)
    
    # 获取outpaint_hires目录
    outpaint_root = f"./outpaint_hires/process_{process_id}"
    if not os.path.exists(outpaint_root):
        print(f"警告：找不到outpaint目录: {outpaint_root}")
        return collection_dir
    
    # 遍历所有数据集目录
    copied_count = 0
    for dataset_name in os.listdir(outpaint_root):
        dataset_path = os.path.join(outpaint_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        
        # 创建数据集集合目录
        dataset_collection_dir = os.path.join(collection_dir, dataset_name)
        ensure_dir_exists(dataset_collection_dir)
        
        # 遍历shot目录（如1_shot, 5_shot, 10_shot）
        for shot_dir in os.listdir(dataset_path):
            # 如果指定了shot_number，则只处理对应的shot目录
            if shot_number is not None and not shot_dir.startswith(f"{shot_number}_shot"):
                continue
                
            shot_path = os.path.join(dataset_path, shot_dir)
            if not os.path.isdir(shot_path):
                continue
            
            # 创建shot集合目录
            shot_collection_dir = os.path.join(dataset_collection_dir, shot_dir)
            ensure_dir_exists(shot_collection_dir)
            
            # 直接遍历样本目录（移除results层）
            for sample_id in os.listdir(shot_path):
                sample_path = os.path.join(shot_path, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                
                # 查找所有final_result文件
                final_result_files = glob.glob(os.path.join(sample_path, "*_final_result*.png"))
                for final_result in final_result_files:
                    # 仅复制文件名中包含final_result的文件
                    final_result_filename = os.path.basename(final_result)
                    target_path = os.path.join(shot_collection_dir, final_result_filename)
                    
                    try:
                        shutil.copy(final_result, target_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"复制文件 {final_result} 到 {target_path} 失败: {str(e)}")
    
    print(f"已将 {copied_count} 个最终结果文件复制到集合目录: {collection_dir}")
    return collection_dir

def main():
    """主函数"""
    # 首先声明要使用的全局变量
    global HR_MIN_DIMENSION, PROCESS_ID
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="高分辨率Outpainting处理脚本")
    parser.add_argument("--dataset", type=str, help="要处理的数据集名称，如不指定则处理所有配置的数据集")
    parser.add_argument("--dataset_group", type=str, choices=['1', '2', 'all'], help="选择要处理的数据集组（1、2或all）", default='all')
    parser.add_argument("--sample_id", type=str, help="要处理的样本ID，如不指定则处理所有样本")
    parser.add_argument("--shot", type=int, default=1, choices=[1, 2, 3, 5, 10, 20], help="使用的shot数（1, 2, 3, 5, 10或20）")
    parser.add_argument("--min_dimension", type=int, default=HR_MIN_DIMENSION, help="默认上采样的最小目标尺寸（默认1024），将被数据集特定设置覆盖")
    parser.add_argument("--custom_upscale", type=str, help="自定义上采样维度，格式为 '数据集名称:维度值'，如 'UODD:2048'")
    parser.add_argument("--process_id", type=str, help="自定义进程ID，如不指定则自动生成")
    parser.add_argument("--collect_only", action="store_true", help="仅收集已有的最终结果，不进行处理")
    # 添加新的命令行参数，用于启用多bbox支持
    parser.add_argument("--multi_bbox", action="store_true", help="启用多bbox支持，处理同一图像上的多个物体")
    # 添加断点续传相关参数
    parser.add_argument("--resume", action="store_true", help="启用断点续传，只处理之前失败的图片")
    parser.add_argument("--log_file", type=str, help="用于断点续传的日志文件路径")
    parser.add_argument("--failed_only", action="store_true", help="只处理失败的样本")
    # 添加多GPU并行参数
    parser.add_argument("--multi_gpu", action="store_true", help="启用多GPU并行处理")
    parser.add_argument("--num_gpus", type=int, help="使用的GPU数量，如不指定则自动检测")
    args = parser.parse_args()
    
    # 更新全局变量
    if args.min_dimension and args.min_dimension != HR_MIN_DIMENSION:
        HR_MIN_DIMENSION = args.min_dimension
        print(f"已设置默认最小目标尺寸为: {HR_MIN_DIMENSION}")
    
    # 处理自定义上采样维度
    if args.custom_upscale:
        try:
            # 格式为 "数据集名称:维度值"，如 "UODD:2048"
            dataset_name, dimension_str = args.custom_upscale.split(":")
            dimension = int(dimension_str)
            if dataset_name in upscale_dimension_params:
                upscale_dimension_params[dataset_name] = dimension
                print(f"已为数据集 {dataset_name} 设置自定义上采样维度: {dimension}")
            else:
                print(f"警告：未知数据集 {dataset_name}，无法设置自定义上采样维度")
        except Exception as e:
            print(f"解析自定义上采样维度时出错: {str(e)}")
            print("正确格式为 '数据集名称:维度值'，如 'UODD:2048'")
    
    if args.process_id:
        PROCESS_ID = args.process_id
        print(f"使用自定义进程ID: {PROCESS_ID}")
    
    # 如果只收集结果
    if args.collect_only:
        print(f"只收集最终结果模式，shot数: {args.shot if args.shot else '所有'}")
        collection_dir = copy_final_results_to_collection(PROCESS_ID, args.shot)
        print(f"所有最终结果已收集到: {collection_dir}")
        return
    
    # 打印多bbox支持状态
    if args.multi_bbox:
        print("已启用多bbox支持，将处理同一图像上的多个物体")
    
    # 断点续传功能
    processed_samples = set()
    failed_samples = set()
    
    if args.resume or args.failed_only:
        log_file = args.log_file
        if not log_file:
            # 如果未指定日志文件，尝试查找匹配的日志文件
            dataset_str = args.dataset if args.dataset else "all"
            shot_str = f"shot{args.shot}"
            log_pattern = f"*{dataset_str}*{shot_str}*.log"
            log_files = glob.glob(log_pattern)
            
            if log_files:
                log_file = log_files[0]  # 使用找到的第一个匹配的日志文件
                print(f"使用日志文件: {log_file}")
            else:
                print(f"警告：未找到匹配的日志文件，断点续传功能将无法使用")
        
        if log_file and os.path.exists(log_file):
            print(f"从日志文件 {log_file} 加载处理记录，用于断点续传...")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 检查成功完成的样本
                        if "处理样本" in line and "处理完成" in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "样本" and i+1 < len(parts):
                                    sample_id = parts[i+1]
                                    processed_samples.add(sample_id)
                                    break
                        # 检查处理失败的样本
                        elif "处理样本" in line and "处理失败" in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "样本" and i+1 < len(parts):
                                    sample_id = parts[i+1]
                                    failed_samples.add(sample_id)
                                    break
                
                print(f"从日志文件中找到 {len(processed_samples)} 个已处理成功的样本")
                print(f"从日志文件中找到 {len(failed_samples)} 个处理失败的样本")
            except Exception as e:
                print(f"读取日志文件时出错: {str(e)}")
    
    # 打印当前上采样维度设置
    print("\n数据集上采样目标维度设置:")
    for dataset, dimension in sorted(upscale_dimension_params.items()):
        print(f"  {dataset}: {dimension}像素")
    print("")
    
    # 多GPU并行设置
    use_multi_gpu = args.multi_gpu
    num_gpus = 1
    
    if use_multi_gpu:
        available_gpus = get_available_gpus()
        if available_gpus == 0:
            print("警告：未检测到可用GPU，将使用单GPU模式")
            use_multi_gpu = False
        else:
            if args.num_gpus:
                num_gpus = min(args.num_gpus, available_gpus)
            else:
                num_gpus = available_gpus
            
            print(f"启用多GPU并行模式，使用 {num_gpus} 个GPU (可用GPU数: {available_gpus})")
            
            # 设置multiprocessing启动方法
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
    
    # 打印处理信息
    processing_mode = f"多GPU并行 ({num_gpus} GPU)" if use_multi_gpu else "单GPU"
    print(f"开始高分辨率Outpainting处理，模式: {processing_mode}, 进程ID: {PROCESS_ID}, shot数: {args.shot}")
    
    # 确定要处理的数据集
    process_datasets = []
    if args.dataset:
        # 如果指定了单个数据集，则只处理该数据集
        process_datasets = [args.dataset]
    else:
        # 根据dataset_group参数选择数据集组
        if args.dataset_group == '1':
            process_datasets = datasets_1
        elif args.dataset_group == '2':
            process_datasets = datasets_2
        else:  # 'all'
            process_datasets = datasets_to_process
    
    # 处理指定数据集或所有数据集
    all_logs = {}
    for dataset_name in process_datasets:
        if dataset_name in all_datasets:
            print(f"正在处理数据集: {dataset_name}, shot数: {args.shot}")
            if args.sample_id:
                print(f"处理样本: {args.sample_id}")
                # 检查此样本是否已成功处理过
                if args.resume and args.sample_id in processed_samples:
                    print(f"样本 {args.sample_id} 已成功处理过，跳过")
                    continue
                # 如果只处理失败的样本，检查此样本是否在失败列表中
                if args.failed_only and args.sample_id not in failed_samples:
                    print(f"样本 {args.sample_id} 不在失败列表中，跳过")
                    continue
                
                log_info = process_sample_hires(dataset_name, args.sample_id, shot_number=args.shot)
                all_logs[dataset_name] = [log_info]
                
                # 为单个样本创建格式化的结果JSON
                result_json = generate_formatted_result_json(dataset_name, [log_info], args.shot)
                save_formatted_result_json(dataset_name, result_json, args.shot)
            else:
                # 处理所有样本或只处理失败的样本
                if args.failed_only:
                    print(f"只处理失败的样本，共 {len(failed_samples)} 个")
                    # 将失败样本ID列表传递给处理函数
                    if use_multi_gpu:
                        logs = process_dataset_samples_multi_gpu(
                            dataset_name, 
                            sample_ids=list(failed_samples), 
                            shot_number=args.shot,
                            num_gpus=num_gpus
                        )
                    else:
                        logs = process_dataset_samples(
                            dataset_name, 
                            sample_ids=list(failed_samples), 
                            shot_number=args.shot
                        )
                else:
                    # 处理所有样本
                    if use_multi_gpu:
                        logs = process_dataset_samples_multi_gpu(
                            dataset_name, 
                            shot_number=args.shot,
                            num_gpus=num_gpus
                        )
                    else:
                        # 将已处理的样本ID传递给process_dataset_samples函数，以便跳过
                        logs = process_dataset_samples(
                            dataset_name, 
                            shot_number=args.shot,
                            skip_processed=processed_samples if args.resume else set()
                        )
                all_logs[dataset_name] = logs
        else:
            print(f"错误：未知数据集 {dataset_name}，可用的数据集: {', '.join(all_datasets)}")
            continue
    
    # 收集最终结果到单一目录
    collection_dir = copy_final_results_to_collection(PROCESS_ID, args.shot)
    
    # 保存内部日志（完整记录）
    if all_logs:
        log_path = save_logs(all_logs)
        print(f"所有处理完成，内部日志已保存到: {log_path}")
        print(f"所有最终结果已收集到: {collection_dir}")
    else:
        print("没有成功处理任何数据集")

if __name__ == "__main__":
    main()
