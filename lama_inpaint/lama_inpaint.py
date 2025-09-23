import os
import json
import numpy as np
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
import argparse
import logging
import time
from datetime import datetime
from collections import defaultdict

# 设置日志记录
def setup_logger():
    """设置日志记录器"""
    log_dir = "../lamainpaint/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lama_inpaint_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def create_mask_from_bbox(image_width, image_height, bbox):
    """创建一个基于bbox的mask图像，确保坐标在图像边界内"""
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    # bbox格式: [x, y, width, height]
    x, y, width, height = bbox
    
    # 确保坐标在图像边界内
    x = max(0, x)
    y = max(0, y)
    right = min(image_width, x + width)
    bottom = min(image_height, y + height)
    
    # 只有当区域有效时才绘制
    if right > x and bottom > y:
        draw.rectangle([x, y, right, bottom], fill=255)
    
    return mask

def create_mask_from_multiple_bboxes(image_width, image_height, bboxes):
    """创建一个基于多个bbox的mask图像，所有bbox区域都标记为需要inpaint"""
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    for bbox in bboxes:
        # bbox格式: [x, y, width, height]
        x, y, width, height = bbox
        
        # 确保坐标在图像边界内
        x = max(0, x)
        y = max(0, y)
        right = min(image_width, x + width)
        bottom = min(image_height, y + height)
        
        # 只有当区域有效时才绘制
        if right > x and bottom > y:
            draw.rectangle([x, y, right, bottom], fill=255)
    
    return mask

def ensure_rgb(image):
    """确保图像是RGB模式（3通道）"""
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def process_dataset(dataset_name, shot_count, logger):
    """处理特定数据集的特定shot数量"""
    logger.info(f"开始处理数据集: {dataset_name}, {shot_count}-shot")
    
    # 设置路径
    dataset_path = os.path.join("../datasets", dataset_name)
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return 0, 0
        
    train_images_dir = os.path.join(dataset_path, "train")
    if not os.path.exists(train_images_dir):
        logger.error(f"训练图像目录不存在: {train_images_dir}")
        return 0, 0
        
    annotation_file = os.path.join(dataset_path, "annotations", f"{shot_count}_shot.json")
    
    # 确保输出目录存在，使用安全的路径名
    safe_dataset_name = dataset_name.replace(" ", "_").replace("-", "_")
    output_dir = os.path.join("../lamainpaint", safe_dataset_name, f"{shot_count}_shot")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 初始化LaMa模型
    simple_lama = SimpleLama()
    
    # 加载注释文件
    try:
        with open(annotation_file, 'r') as f:
            annotations_data = json.load(f)
        logger.info(f"成功加载注释文件 {annotation_file}")
    except Exception as e:
        logger.error(f"无法加载注释文件 {annotation_file}: {e}")
        return 0, 0
    
    # 创建图像ID到文件名的映射
    image_id_to_filename = {}
    for image_info in annotations_data.get('images', []):
        image_id_to_filename[image_info['id']] = {
            'file_name': image_info['file_name'],
            'width': image_info['width'],
            'height': image_info['height']
        }
    
    # 按image_id对所有annotations分组
    image_id_to_annotations = defaultdict(list)
    for annotation in annotations_data.get('annotations', []):
        image_id_to_annotations[annotation['image_id']].append(annotation)
    
    # 获取类别ID到名称的映射（如果有）
    category_mapping = {}
    if 'categories' in annotations_data:
        for category in annotations_data['categories']:
            category_mapping[category['id']] = category['name']
    
    logger.info(f"找到 {len(image_id_to_filename)} 个图像和 {len(annotations_data.get('annotations', []))} 个注释")
    logger.info(f"共有 {len(image_id_to_annotations)} 个不同的图像需要处理")
    
    # 统计处理的图像和错误的图像
    processed_count = 0
    error_count = 0
    multi_bbox_count = 0
    
    # 处理每个图像
    for image_id, annotations_list in tqdm(image_id_to_annotations.items()):
        if image_id not in image_id_to_filename:
            logger.warning(f"警告: 找不到图像ID {image_id} 的信息")
            continue
        
        image_info = image_id_to_filename[image_id]
        image_path = os.path.join(train_images_dir, image_info['file_name'])
        
        # 检查这个图像是否有多个bbox
        has_multiple_bboxes = len(annotations_list) > 1
        if has_multiple_bboxes:
            multi_bbox_count += 1
            # 记录多bbox信息
            bbox_category_info = []
            for ann in annotations_list:
                category_id = ann['category_id']
                category_name = category_mapping.get(category_id, f"未知类别 {category_id}")
                bbox_category_info.append(f"{category_name}(ID:{category_id})")
            
            logger.info(f"处理多bbox图像: {image_info['file_name']}, bbox数量: {len(annotations_list)}, "
                       f"类别: {', '.join(bbox_category_info)}")
        
        try:
            # 加载图像
            image = Image.open(image_path)
            
            # 确保图像是RGB模式
            image = ensure_rgb(image)
            
            # 确保图像尺寸与注释中的尺寸一致，如果不一致，则调整图像尺寸
            if image.width != image_info['width'] or image.height != image_info['height']:
                logger.debug(f"调整图像 {image_info['file_name']} 的尺寸从 {image.width}x{image.height} 到 {image_info['width']}x{image_info['height']}")
                image = image.resize((image_info['width'], image_info['height']))
            
            # 收集该图像的所有bbox
            bboxes = [ann['bbox'] for ann in annotations_list]
            
            # 创建包含所有bbox的mask
            mask = create_mask_from_multiple_bboxes(image_info['width'], image_info['height'], bboxes)
            
            try:
                # 进行修复
                result = simple_lama(image, mask)
            except RuntimeError as e:
                # 如果出现通道错误，尝试使用numpy数组直接处理
                if "expected input" in str(e) and "channels" in str(e):
                    logger.warning(f"尝试使用替代方法处理图像 {image_path}")
                    
                    # 将图像和掩码转换为numpy数组
                    img_np = np.array(image)
                    mask_np = np.array(mask)
                    
                    # 确保掩码是二维的
                    if len(mask_np.shape) > 2:
                        mask_np = mask_np[:, :, 0]
                    
                    # 将掩码转换为二值掩码
                    mask_np = (mask_np > 127).astype(np.uint8) * 255
                    
                    # 直接使用模型的内部方法，跳过通道检查
                    result_array = simple_lama.model(img_np, mask_np)
                    result = Image.fromarray(result_array)
                else:
                    raise e
            
            # 保存结果
            output_filename = os.path.join(output_dir, image_info['file_name'])
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            result.save(output_filename)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"处理图像 {image_path} 时出错: {e}")
            error_count += 1
    
    logger.info(f"完成数据集 {dataset_name} {shot_count}-shot 的处理: 成功处理 {processed_count} 个图像, 错误 {error_count} 个")
    logger.info(f"其中处理了 {multi_bbox_count} 个有多个bbox的图像")
    return processed_count, error_count

def main():
    """主函数,处理所有数据集和shot配置"""
    # 设置日志记录
    logger = setup_logger()
    logger.info("LaMa Inpainting开始执行")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LaMa Inpainting for Multiple Datasets')
    parser.add_argument('--datasets', nargs='+', default=['ArTaxOr', 'clipart1k', 'DIOR', 'FISH', 'NEU-DET'],
                        help='要处理的数据集列表')
    parser.add_argument('--shots', nargs='+', default=['1', '2', '3', '5', '10'], 
                        help='每个数据集要处理的shot数量')
    parser.add_argument('--fix-channels', action='store_true',
                        help='修复通道不匹配问题')
    args = parser.parse_args()
    
    logger.info(f"将处理以下数据集: {', '.join(args.datasets)}")
    logger.info(f"将处理以下shot数量: {', '.join(args.shots)}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 统计处理的总数据
    total_processed = 0
    total_errors = 0
    
    # 处理所有指定的数据集和shot配置
    for dataset_name in args.datasets:
        for shot_count in args.shots:
            try:
                processed, errors = process_dataset(dataset_name, shot_count, logger)
                total_processed += processed
                total_errors += errors
            except Exception as e:
                logger.error(f"处理数据集 {dataset_name} {shot_count}-shot 时发生错误: {e}")
    
    # 记录结束时间和总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"所有数据集处理完成: 成功处理 {total_processed} 个图像, 错误 {total_errors} 个")
    logger.info(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

if __name__ == "__main__":
    main()