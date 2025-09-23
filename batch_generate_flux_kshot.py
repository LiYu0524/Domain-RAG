import os
import json
import glob
from datetime import datetime
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.utils import load_image
from tqdm import tqdm
import shutil
import random  # 添加random模块用于随机选择COCO图像
import argparse  # 导入argparse模块用于解析命令行参数

# 全局变量配置
OUTPUT_BASE_DIR = "result"  # 可以修改此变量来改变保存的目录

# 数据库选择配置
DATABASE_TYPE = "coco"  # 可选值: "coco" 或 "miniimagenet"

# 路径配置
local_path = "./model"
repo_redux = local_path + "/FLUX.1-Redux-dev"
repo_base = local_path + "/FLUX.1-dev"
inpainted_result_dir = "./lamainpaint"
retrieval_results_dir = "/nvme/liyu/Flux/retrieval/retrieval_results_cocominiimagenet"
all_datasets_retrieval_file = os.path.join(retrieval_results_dir, "all_shots_retrieval_results.json")

# 数据库路径配置
coco_dataset_dir = "./retrieval/coco/train2017"
miniimagenet_dataset_dir = "./retrieval/miniimagenet"  # miniImageNet数据集路径

# 添加命令行参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description='批量生成Flux k-shot图像')
    parser.add_argument('--dataset', type=str, default=None, help='要处理的数据集名称，例如 NWPU_VHR_10')
    parser.add_argument('--shots', nargs='+', type=int, default=None, help='要处理的shot数量，例如 3 5 10 20')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_DIR, help='结果保存的目录路径')
    parser.add_argument('--database', type=str, default=DATABASE_TYPE, choices=['coco', 'miniimagenet'], 
                        help='使用的参考数据库类型，可选值: coco 或 miniimagenet')
    parser.add_argument('--retrieval_results_dir', type=str, default="/nvme/liyu/Flux/retrieval/retrieval_results_cocominiimagenet", 
                        help='检索结果文件所在目录路径')
    parser.add_argument('--dataset_group', type=str, default=None, 
                        choices=['dataset1', 'dataset2', 'dataset3', 'dataset4'], 
                        help='要处理的数据集分组，可选值: dataset1, dataset2, dataset3, dataset4')
    return parser.parse_args()

# 固定参数配置
device = "cuda"
dtype = torch.bfloat16

# 权重配置，保留原始代码中的参数值
target_image_scale = 1.0
# target_text_scale = 1.0

coco_image_scale = 0.8
# coco_image_scale = 0.8

target_text_scale = 1.0
# coco_image_scale = 1.0

coco_text_scale = 1.0

# 提示词
prompt_retrieval = ""

# 数据集配置 - 可以配置不同的数据集
# 将默认数据集改为可配置项
kshot_dataset_name = "ArTaxOr"
# 可选值: "ArTaxOr", "UODD", "FISH", "DIOR", "NEU-DET", "clipart1k", "NWPU_VHR_10", "coco"

# 数据集对应的样本名称参考:
# - ArTaxOr: 蜘蛛标本名，如 "Alopecosa_albofasciata"
# - UODD: 水下物体样本，如 "seacucumber_001", "scallop_003", "seaurchin_002"
# - FISH: 鱼类样本，如 "Acipenser_gueldenstaedtii_LC"
# - DIOR: 遥感物体样本，如 "airport_00015"
# - NEU-DET: 钢铁表面缺陷样本，如 "crazing_001"
# - clipart1k: 卡通图像样本，如 "aeroplane_001"
# - NWPU_VHR_10: 遥感卫星图像样本，如 "171"

# 数据集分组配置 - 将数据集分为4个组
dataset_groups = {
    # "dataset1": ["ArTaxOr"],
    # "dataset2": ["UODD", "FISH"],
    # "dataset3": ["DIOR", "NEU-DET"],
    # "dataset4": ["clipart1k", "NWPU_VHR_10"]
    "dataset1": ["UODD", "ArTaxOr", "FISH", "coco"],
    "dataset2": ["DIOR", "NEU-DET", "clipart1k"],
}

# 新增k-shot路径配置
lamainpaint_dir = "./lamainpaint"
kshot_dir = os.path.join(lamainpaint_dir, kshot_dataset_name, "5_shot")

# 数据集列表
datasets = ["FISH", "DIOR", "ArTaxOr", "UODD", "NEU-DET", "clipart1k", "coco"]
def load_all_datasets_retrieval_results():
    """加载所有数据集的检索结果"""
    if os.path.exists(all_datasets_retrieval_file):
        with open(all_datasets_retrieval_file, 'r') as f:
            results = json.load(f)
            
            # 调试信息：打印JSON结构
            print(f"成功加载检索结果文件: {all_datasets_retrieval_file}")
            print(f"检索结果中的顶级键数量: {len(results.keys())}")
            print(f"检索结果中的部分顶级键: {list(results.keys())[:5]}")  # 只打印前5个键
            
            # 检查是否有指定数据集的键
            if kshot_dataset_name in results:
                print(f"{kshot_dataset_name}下的键数量: {len(results[kshot_dataset_name].keys())}")
                print(f"{kshot_dataset_name}下的部分键: {list(results[kshot_dataset_name].keys())[:5]}")
            
            return results
    else:
        print(f"警告：找不到所有数据集的检索结果文件 {all_datasets_retrieval_file}")
        return None

def load_model():
    """加载模型"""
    print("正在加载模型...")
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

    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
        repo_redux,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=dtype
    ).to(device)

    pipe = FluxPipeline.from_pretrained(
        repo_base,
        torch_dtype=dtype
    ).to(device)
    
    return pipe_prior_redux, pipe

def get_retrieval_results(dataset_name):
    """获取检索结果文件"""
    retrieval_file = os.path.join(retrieval_results_dir, f"{dataset_name}_all_categories_retrieval_results.json")
    if os.path.exists(retrieval_file):
        with open(retrieval_file, 'r') as f:
            return json.load(f)
    else:
        print(f"警告：找不到数据集 {dataset_name} 的检索结果文件")
        return None

def get_sample_folders(dataset_name):
    """获取数据集中所有样本文件夹"""
    dataset_path = os.path.join(inpainted_result_dir, dataset_name, "inpainted_images")
    if not os.path.exists(dataset_path):
        print(f"警告：找不到数据集 {dataset_name} 的样本文件夹")
        return []
    
    # 获取所有样本文件夹
    sample_folders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f)) and 
                     f not in ["__pycache__"]]
    
    return sample_folders

def get_kshot_samples():
    """获取k-shot样本列表"""
    if not os.path.exists(kshot_dir):
        print(f"警告：找不到k-shot目录 {kshot_dir}")
        return []
    
    # 获取所有jpg文件
    sample_files = [f for f in os.listdir(kshot_dir) 
                   if f.endswith('.jpg') and 
                   os.path.isfile(os.path.join(kshot_dir, f))]
    
    # 提取文件名（不含扩展名）作为样本名
    sample_names = [os.path.splitext(f)[0] for f in sample_files]
    
    print(f"从k-shot目录找到 {len(sample_names)} 个样本")
    return sample_names

def get_random_coco_images(num_images=10):
    """随机选择COCO数据集中的图像"""
    if not os.path.exists(coco_dataset_dir):
        print(f"警告：找不到COCO数据集目录 {coco_dataset_dir}")
        return []
    
    # 获取所有COCO图像
    coco_images = [f for f in os.listdir(coco_dataset_dir) 
                  if f.endswith(('.jpg', '.png')) and 
                  os.path.isfile(os.path.join(coco_dataset_dir, f))]
    
    if not coco_images:
        print("警告：COCO数据集目录中没有找到图像")
        return []
    
    # 随机选择指定数量的图像
    if len(coco_images) <= num_images:
        selected_images = coco_images
    else:
        selected_images = random.sample(coco_images, num_images)
    
    # 返回完整路径
    selected_paths = [os.path.join(coco_dataset_dir, img) for img in selected_images]
    print(f"随机选择了 {len(selected_paths)} 个COCO图像")
    return selected_paths

def get_random_miniimagenet_images(num_images=10):
    """随机选择miniImageNet数据集中的图像"""
    if not os.path.exists(miniimagenet_dataset_dir):
        print(f"警告：找不到miniImageNet数据集目录 {miniimagenet_dataset_dir}")
        return []
    
    # 获取所有miniImageNet图像
    all_images = []
    
    # 遍历miniImageNet目录，查找所有图像文件
    for root, dirs, files in os.walk(os.path.join(miniimagenet_dataset_dir, "train")):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print("警告：miniImageNet数据集目录中没有找到图像")
        return []
    
    # 随机选择指定数量的图像
    if len(all_images) <= num_images:
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_images)
    
    print(f"随机选择了 {len(selected_images)} 个miniImageNet图像")
    return selected_images

def find_similar_image(retrieval_results, sample_name, categories):
    """从检索结果中找到最相似的图片"""
    # 如果categories是单个字符串，转为列表以统一处理
    if isinstance(categories, str):
        categories = [categories]
    
    # 对于每个类别尝试查找
    for category in categories:
        if category not in retrieval_results:
            continue
            
        for item in retrieval_results[category]:
            # 检查原始文件名是否匹配
            original_filename = item.get("original_filename", "")
            if sample_name in original_filename:
                similar_images = item.get("similar_images", [])
                if not similar_images:
                    continue
                
                # 首先尝试找到没有blur的图片
                non_blurred_images = []
                blurred_images = []
                
                for similar in similar_images:
                    image_path = similar.get("image_path", "")
                    if not image_path:
                        continue
                    
                    # 使用get_correct_image_path获取正确的图像路径
                    correct_path = get_correct_image_path(image_path)
                    
                    if correct_path and os.path.exists(correct_path):
                        # 检查是否是模糊图像
                        if "_blurred" in os.path.basename(correct_path):
                            blurred_images.append((similar["similarity"], correct_path))
                        else:
                            non_blurred_images.append((similar["similarity"], correct_path))
                
                # 优先选择非模糊图像，按相似度排序
                if non_blurred_images:
                    non_blurred_images.sort(key=lambda x: x[0], reverse=True)
                    print(f"找到样本 {sample_name} 的非模糊相似图像: {os.path.basename(non_blurred_images[0][1])}")
                    return non_blurred_images[0][1]
                
                # 如果没有找到非模糊图像，则使用模糊图像
                if blurred_images:
                    blurred_images.sort(key=lambda x: x[0], reverse=True)
                    print(f"找到样本 {sample_name} 的模糊相似图像: {os.path.basename(blurred_images[0][1])}")
                    return blurred_images[0][1]
    
    return None

def find_top5_similar_images(all_datasets_results, dataset_name, sample_name):
    """从all_datasets_retrieval_results中找到top5相似的COCO图像"""
    if not all_datasets_results:
        print(f"警告：all_datasets_results为空")
        return []
    
    # 尝试不同的大小写形式
    dataset_variants = [
        dataset_name,  # 原始形式
        dataset_name.upper(),  # 全大写
        dataset_name.lower(),  # 全小写
        dataset_name.capitalize(),  # 首字母大写
    ]
    
    # 处理特殊情况
    if dataset_name.lower() == "neu-det":
        dataset_variants.extend(["NEU-DET", "neu-det", "Neu-Det"])
    elif dataset_name.lower() == "artaxor":
        dataset_variants.extend(["ArTaxOr", "ARTAXOR"])
    elif dataset_name.lower() == "uodd":
        dataset_variants.extend(["UODD", "uodd", "Uodd"])
    
    # 查找匹配的数据集名称
    matched_dataset = None
    for variant in dataset_variants:
        if variant in all_datasets_results:
            matched_dataset = variant
            break
    
    if not matched_dataset:
        print(f"警告：在all_datasets_retrieval_results中找不到数据集 {dataset_name}，尝试了以下变体：{dataset_variants}")
        return []
    
    # 获取样本的检索结果
    dataset_results = all_datasets_results[matched_dataset]
    
    # 尝试不同的样本名称格式
    sample_variants = [
        sample_name,  # 原始形式
        sample_name.upper(),  # 全大写
        sample_name.lower(),  # 全小写
    ]
    
    # 查找匹配的样本名称
    matched_sample = None
    for variant in sample_variants:
        if variant in dataset_results:
            matched_sample = variant
            break
    
    if not matched_sample:
        print(f"警告：在all_datasets_retrieval_results中找不到样本 {sample_name}，尝试了以下变体：{sample_variants}")
        return []
    
    sample_results = dataset_results[matched_sample]
    
    # 收集所有可能的相似COCO图像
    coco_images = []
    
    # 处理多种可能的JSON结构
    if isinstance(sample_results, list):
        # 如果样本结果是列表形式
        for item in sample_results:
            process_similar_images_item(item, coco_images)
    elif isinstance(sample_results, dict):
        # 如果样本结果是字典形式
        process_similar_images_item(sample_results, coco_images)
    
    # 按相似度排序
    coco_images.sort(key=lambda x: x[0], reverse=True)
    
    # 返回top5结果（去重）
    top5_paths = []
    seen_paths = set()
    
    for similarity, path, rank in coco_images:
        if path not in seen_paths:
            seen_paths.add(path)
            top5_paths.append((similarity, path, rank))
            if len(top5_paths) >= 5:
                break
    
    if top5_paths:
        print(f"为样本 {sample_name} 找到 {len(top5_paths)} 个相似COCO图像")
        return top5_paths
    else:
        print(f"警告：为样本 {sample_name} 找不到相似COCO图像")
        return []

def process_similar_images_item(item, coco_images):
    """处理单个检索结果项，提取相似图像"""
    # 处理字典里可能直接包含similar_images的情况
    similar_images = item.get("similar_images", [])
    process_similar_images_list(similar_images, coco_images)
    
    # 处理字典里可能包含子列表的情况
    for key, value in item.items():
        if isinstance(value, list) and key != "similar_images":
            for sub_item in value:
                if isinstance(sub_item, dict):
                    similar_images = sub_item.get("similar_images", [])
                    process_similar_images_list(similar_images, coco_images)

def process_similar_images_list(similar_images, coco_images):
    """处理相似图像列表，提取COCO图像路径"""
    for similar in similar_images:
        # 跳过非字典项
        if not isinstance(similar, dict):
            continue
            
        image_path = similar.get("image_path", "")
        similarity = similar.get("similarity", 0)
        rank = similar.get("rank", 0)
        
        if not image_path or not isinstance(image_path, str):
            continue
        
        # 使用get_correct_image_path函数获取正确的图像路径
        correct_path = get_correct_image_path(image_path)
        
        if correct_path and os.path.exists(correct_path):
            coco_images.append((similarity, correct_path, rank))
        else:
            # 如果找不到图像，尝试处理blurred版本
            if "_blurred" in image_path:
                # 尝试非模糊版本
                non_blurred_path = image_path.replace("_blurred", "")
                correct_non_blurred_path = get_correct_image_path(non_blurred_path)
                
                if correct_non_blurred_path and os.path.exists(correct_non_blurred_path):
                    coco_images.append((similarity, correct_non_blurred_path, rank))
                else:
                    # 尝试模糊版本
                    correct_blurred_path = get_correct_image_path(image_path)
                    if correct_blurred_path and os.path.exists(correct_blurred_path):
                        coco_images.append((similarity, correct_blurred_path, rank))

def generate_image(pipe_prior_redux, pipe, coco_image_path, target_image_path, output_path, rank=None, similarity=None):
    """使用模型生成图像"""
    try:
        # 加载输入图像
        coco_image = load_image(coco_image_path)
        target_image = load_image(target_image_path)
        
        # 获取目标图像的尺寸
        width, height = target_image.size
        
        # 确保尺寸是16的倍数（vae_scale_factor * 2 = 8 * 2 = 16）
        height = (height // 16) * 16
        width = (width // 16) * 16
        
        # 如果尺寸太小，设置最小尺寸
        min_size = 64  # 最小尺寸，确保是16的倍数
        height = max(height, min_size)
        width = max(width, min_size)
        
        # 生成图像
        pipe_prior_output = pipe_prior_redux(
            [coco_image, target_image],
            prompt=["", prompt_retrieval],
            prompt_2=["", prompt_retrieval],
            prompt_embeds_scale=[coco_image_scale, target_image_scale],
            pooled_prompt_embeds_scale=[coco_text_scale, target_text_scale],
        )
        
        images = pipe(
            guidance_scale=2.5,
            num_inference_steps=50,
            height=1024,  # 使用获取的高度
            width=1024,    # 使用获取的宽度
            generator=torch.Generator("cpu").manual_seed(0),
            **pipe_prior_output,
        ).images
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存生成的图像
        images[0].save(output_path)
        
        # 保存参数信息
        params_file = os.path.join(os.path.dirname(output_path), "params.txt")
        if not os.path.exists(params_file):
            with open(params_file, "w") as f:
                f.write(f"数据库类型: {DATABASE_TYPE}\n")  # 添加数据库类型信息
                f.write(f"参考图像权重: {coco_image_scale}\n")
                f.write(f"目标图像权重: {target_image_scale}\n")
                f.write(f"参考文本权重: {coco_text_scale}\n")
                f.write(f"目标文本权重: {target_text_scale}\n")
                f.write(f"提示词: {prompt_retrieval}\n")
                f.write(f"指导比例: 2.5\n")
                f.write(f"推理步数: 50\n")
                f.write(f"生成图像尺寸: {width}x{height}\n")  # 添加尺寸信息
                f.write(f"原始图像尺寸: {target_image.size[0]}x{target_image.size[1]}\n")  # 添加原始尺寸信息
        
        # 为每个rank保存参考输入信息
        rank_str = f"rank{rank}" if rank is not None else ""
        similarity_str = f"_sim{similarity:.4f}" if similarity is not None else ""
        ref_info_file = os.path.join(os.path.dirname(output_path), f"ref_info{rank_str}{similarity_str}.txt")
        with open(ref_info_file, "w") as f:
            f.write(f"数据库类型: {DATABASE_TYPE}\n")  # 添加数据库类型信息
            f.write(f"参考图像: {coco_image_path}\n")
            f.write(f"目标图像: {target_image_path}\n")
            f.write(f"生成图像尺寸: {width}x{height}\n")  # 添加尺寸信息
            f.write(f"原始图像尺寸: {target_image.size[0]}x{target_image.size[1]}\n")  # 添加原始尺寸信息
            if rank is not None:
                f.write(f"排名: {rank}\n")
            if similarity is not None:
                f.write(f"相似度: {similarity}\n")
        
        # 复制输入图像 (只复制一次目标图像，每个rank复制对应的参考图像)
        ref_image_output = os.path.join(os.path.dirname(output_path), f"ref_input{rank_str}.jpg")
        target_image_output = os.path.join(os.path.dirname(output_path), "target_input.png")
        
        if not os.path.exists(target_image_output):
            shutil.copy(target_image_path, target_image_output)
        
        shutil.copy(coco_image_path, ref_image_output)
        
        return True
    except Exception as e:
        print(f"生成图像时出错: {str(e)}")
        return False

def process_dataset(dataset_name, pipe_prior_redux, pipe, all_datasets_results=None, output_dir=OUTPUT_BASE_DIR):
    """处理单个数据集"""
    print(f"正在处理数据集: {dataset_name}")
    
    # 获取检索结果
    retrieval_results = None
    # 如果没有提供all_datasets_results，使用旧的方法
    if all_datasets_results is None:
        retrieval_results = get_retrieval_results(dataset_name)
        if not retrieval_results:
            print(f"跳过数据集 {dataset_name}，因为找不到检索结果")
            return
    
    # 获取样本文件夹
    sample_folders = get_sample_folders(dataset_name)
    if not sample_folders:
        print(f"跳过数据集 {dataset_name}，因为找不到样本文件夹")
        return
    
    # 创建输出目录
    result_dir = f"{output_dir}/{dataset_name}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = os.path.join(result_dir, f"results_coco_{coco_image_scale}_target_{target_image_scale}_cocotext_{coco_text_scale}_targettext_{target_text_scale}_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 创建参数摘要文件
    with open(os.path.join(output_base_dir, "batch_params.txt"), "w") as f:
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"COCO图像权重: {coco_image_scale}\n")
        f.write(f"目标图像权重: {target_image_scale}\n")
        f.write(f"COCO文本权重: {coco_text_scale}\n")
        f.write(f"目标文本权重: {target_text_scale}\n")
        f.write(f"提示词: {prompt_retrieval}\n")
        f.write(f"指导比例: 2.5\n")
        f.write(f"推理步数: 50\n")
        f.write(f"处理样本数: {len(sample_folders)}\n")
        f.write(f"为每个样本生成: 最多5张图像 (基于相似度最高的COCO图像)\n")
        f.write(f"图像尺寸: 动态调整至与目标图像匹配 (保证是16的倍数)\n")
    
    # 处理日志
    success_count = 0
    failed_count = 0
    total_generated_images = 0
    image_sizes = {}  # 用于记录生成的不同图像尺寸
    
    # 遍历所有样本文件夹
    for sample_name in tqdm(sample_folders, desc=f"处理 {dataset_name} 样本"):
        # 获取目标图像路径
        target_image_path = os.path.join(inpainted_result_dir, dataset_name, "inpainted_images", sample_name, "1_inpainted.png")
        if not os.path.exists(target_image_path):
            print(f"跳过样本 {sample_name}，因为找不到目标图像")
            failed_count += 1
            continue
        
        # 创建样本输出目录
        sample_output_dir = os.path.join(output_base_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # 使用新方法查找top5相似COCO图像
        if all_datasets_results is not None:
            top5_coco_images = find_top5_similar_images(all_datasets_results, dataset_name, sample_name)
            if not top5_coco_images:
                # 如果新方法失败，尝试旧方法
                categories = get_category_for_dataset(dataset_name)
                if not categories:
                    print(f"跳过样本 {sample_name}，因为无法确定数据集类别")
                    failed_count += 1
                    continue
                
                # 确保旧方法有retrieval_results可用
                if retrieval_results is None:
                    retrieval_results = get_retrieval_results(dataset_name)
                    if not retrieval_results:
                        print(f"跳过样本 {sample_name}，因为找不到检索结果")
                        failed_count += 1
                        continue
                
                # 使用旧方法找到最相似的COCO图像
                coco_image_path = find_similar_image(retrieval_results, sample_name, categories)
                if not coco_image_path:
                    print(f"跳过样本 {sample_name}，因为找不到匹配的COCO图像")
                    failed_count += 1
                    continue
                
                # 使用旧方法只生成一张图像
                output_path = os.path.join(sample_output_dir, "generated_image.png")
                success = generate_image(pipe_prior_redux, pipe, coco_image_path, target_image_path, output_path)
                
                if success:
                    print(f"成功生成样本 {sample_name} 的图像")
                    success_count += 1
                    total_generated_images += 1
                    
                    # 记录图像尺寸
                    try:
                        with open(os.path.join(sample_output_dir, "params.txt"), "r") as size_f:
                            for line in size_f:
                                if "生成图像尺寸:" in line:
                                    size_str = line.strip().split(": ")[1]
                                    if size_str not in image_sizes:
                                        image_sizes[size_str] = 1
                                    else:
                                        image_sizes[size_str] += 1
                                    break
                    except Exception as e:
                        print(f"记录图像尺寸时出错: {str(e)}")
                else:
                    print(f"生成样本 {sample_name} 的图像失败")
                    failed_count += 1
            else:
                # 使用top5相似COCO图像生成多张图像
                sample_success = False
                for i, (similarity, coco_path, rank) in enumerate(top5_coco_images):
                    rank = i + 1  # 确保rank从1开始
                    output_path = os.path.join(sample_output_dir, f"generated_image_rank{rank}.png")
                    success = generate_image(
                        pipe_prior_redux, pipe, coco_path, target_image_path, 
                        output_path, rank=rank, similarity=similarity
                    )
                    
                    if success:
                        print(f"成功生成样本 {sample_name} 的图像 (rank {rank})")
                        total_generated_images += 1
                        sample_success = True
                        
                        # 记录图像尺寸
                        try:
                            size_file = os.path.join(sample_output_dir, f"ref_info{rank}.txt")
                            if os.path.exists(size_file):
                                with open(size_file, "r") as size_f:
                                    for line in size_f:
                                        if "生成图像尺寸:" in line:
                                            size_str = line.strip().split(": ")[1]
                                            if size_str not in image_sizes:
                                                image_sizes[size_str] = 1
                                            else:
                                                image_sizes[size_str] += 1
                                            break
                        except Exception as e:
                            print(f"记录图像尺寸时出错: {str(e)}")
                    else:
                        print(f"生成样本 {sample_name} 的图像失败 (rank {rank})")
                
                if sample_success:
                    success_count += 1
                else:
                    failed_count += 1
        else:
            # 使用旧方法
            categories = get_category_for_dataset(dataset_name)
            if not categories:
                print(f"跳过样本 {sample_name}，因为无法确定数据集类别")
                failed_count += 1
                continue
            
            # 确保有retrieval_results可用
            if retrieval_results is None:
                retrieval_results = get_retrieval_results(dataset_name)
                if not retrieval_results:
                    print(f"跳过样本 {sample_name}，因为找不到检索结果")
                    failed_count += 1
                    continue
            
            # 使用旧方法找到最相似的COCO图像
            coco_image_path = find_similar_image(retrieval_results, sample_name, categories)
            if not coco_image_path:
                print(f"跳过样本 {sample_name}，因为找不到匹配的COCO图像")
                failed_count += 1
                continue
            
            # 生成图像
            output_path = os.path.join(sample_output_dir, "generated_image.png")
            success = generate_image(pipe_prior_redux, pipe, coco_image_path, target_image_path, output_path)
            
            if success:
                print(f"成功生成样本 {sample_name} 的图像")
                success_count += 1
                total_generated_images += 1
                
                # 记录图像尺寸
                try:
                    with open(os.path.join(sample_output_dir, "params.txt"), "r") as size_f:
                        for line in size_f:
                            if "生成图像尺寸:" in line:
                                size_str = line.strip().split(": ")[1]
                                if size_str not in image_sizes:
                                    image_sizes[size_str] = 1
                                else:
                                    image_sizes[size_str] += 1
                                    break
                except Exception as e:
                    print(f"记录图像尺寸时出错: {str(e)}")
            else:
                print(f"生成样本 {sample_name} 的图像失败")
                failed_count += 1
    
    # 更新处理日志
    with open(os.path.join(output_base_dir, "batch_params.txt"), "a") as f:
        f.write(f"成功处理样本数: {success_count}\n")
        f.write(f"失败处理样本数: {failed_count}\n")
        f.write(f"总共生成图像数: {total_generated_images}\n")
        
        # 记录各种图像尺寸及其数量
        f.write(f"\n生成图像尺寸统计:\n")
        for size_str, count in sorted(image_sizes.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {size_str}: {count}张图像\n")
            
        f.write(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"数据集 {dataset_name} 处理完成：成功 {success_count} 个样本，失败 {failed_count} 个样本，总共生成 {total_generated_images} 张图像")

def get_category_for_dataset(dataset_name):
    """根据数据集名称获取相应的类别名称"""
    dataset_name = dataset_name.lower()
    if dataset_name == "fish":
        return "fish"
    elif dataset_name == "dior":
        return ["Expressway-Service-area", "airplane", "airport", "baseballfield", 
                "basketballcourt", "bridge", "chimney", "dam", "golffield", 
                "groundtrackfield", "harbor", "overpass", "ship", "stadium", 
                "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"]
    elif dataset_name == "artaxor":
        return "Araneae"
    elif dataset_name == "uodd":
        return ["seacucumber", "scallop", "seaurchin"]
    elif dataset_name == "neu-det":
        return ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    elif dataset_name == "clipart1k":
        return ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    elif dataset_name == "nwpu_vhr_10":
        return "NWPU_VHR_10"
    elif dataset_name == "coco":
        return "coco"
    else:
        print(f"警告：未知数据集 {dataset_name}，无法确定类别")
        return None

def process_kshot_dataset_with_retrieval(dataset_name, pipe_prior_redux, pipe, all_datasets_results, shot_number=5, output_dir=OUTPUT_BASE_DIR):
    """使用检索结果处理k-shot数据集"""
    print(f"正在处理k-shot数据集: {dataset_name} {shot_number}-shot (使用检索结果)")
    
    # 检查检索结果是否存在
    if not all_datasets_results:
        print(f"错误：没有提供检索结果")
        return
    
    # 更新k-shot目录路径以使用指定的shot数量
    shot_dir = os.path.join(lamainpaint_dir, f"{dataset_name}", f"{shot_number}_shot")
    
    # 检查目录是否存在
    if not os.path.exists(shot_dir):
        print(f"错误：找不到k-shot目录 {shot_dir}")
        return
    
    # 获取k-shot样本列表
    sample_files = [f for f in os.listdir(shot_dir) 
                   if f.endswith('.jpg') and 
                   os.path.isfile(os.path.join(shot_dir, f))]
    
    # 提取文件名（不含扩展名）作为样本名
    sample_names = [os.path.splitext(f)[0] for f in sample_files]
    
    if not sample_names:
        print(f"跳过数据集 {dataset_name} {shot_number}-shot，因为找不到样本")
        return
    
    print(f"从{shot_number}-shot目录找到 {len(sample_names)} 个样本")
    
    # 创建输出目录 - 使用新的命名规则
    result_dir = f"{output_dir}/{dataset_name}_{shot_number}shot_retrieval"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = os.path.join(result_dir, f"results_coco_{coco_image_scale}_target_{target_image_scale}_cocotext_{coco_text_scale}_targettext_{target_text_scale}_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 创建参数摘要文件
    with open(os.path.join(output_base_dir, "batch_params.txt"), "w") as f:
        f.write(f"数据集: {dataset_name} ({shot_number}-shot，使用检索结果)\n")
        f.write(f"COCO图像权重: {coco_image_scale}\n")
        f.write(f"目标图像权重: {target_image_scale}\n")
        f.write(f"COCO文本权重: {coco_text_scale}\n")
        f.write(f"目标文本权重: {target_text_scale}\n")
        f.write(f"提示词: {prompt_retrieval}\n")
        f.write(f"指导比例: 2.5\n")
        f.write(f"推理步数: 50\n")
        f.write(f"处理样本数: {len(sample_names)}\n")
        f.write(f"为每个样本生成: 最多10张图像 (基于相似度最高的COCO图像)\n")
        f.write(f"图像尺寸: 动态调整至与目标图像匹配 (保证是16的倍数)\n")
    
    # 处理日志
    success_count = 0
    failed_count = 0
    total_generated_images = 0
    image_sizes = {}  # 用于记录生成的不同图像尺寸
    
    # 对NEU-DET数据集执行特殊检查
    is_neudet_special_case = False
    neudet_available_categories = []
    if dataset_name == "NEU-DET" and "NEU-DET" in all_datasets_results:
        shot_key = f"{shot_number}_shot"
        # 检查是否是直接包含样本的结构
        if shot_key not in all_datasets_results["NEU-DET"] and any(sample_name in all_datasets_results["NEU-DET"] for sample_name in sample_names[:3]):
            is_neudet_special_case = True
            print(f"检测到NEU-DET特殊结构：直接包含样本，没有{shot_key}子目录")
        
        # 获取可用的NEU-DET类别
        if shot_key in all_datasets_results["NEU-DET"]:
            neudet_available_categories = list(all_datasets_results["NEU-DET"][shot_key].keys())
            print(f"\n[NEU-DET] 可用类别: {neudet_available_categories}")
            
            # 检查是否包含特殊类别
            special_categories = ["pitted_surface", "rolled-in_scale", "crazing", "inclusion", "patches", "scratches"]
            for cat in special_categories:
                if cat in neudet_available_categories:
                    print(f"[NEU-DET] 找到特殊类别: {cat}")
                    # 打印该类别下的样本数量
                    if all_datasets_results["NEU-DET"][shot_key][cat]:
                        sample_count = len(all_datasets_results["NEU-DET"][shot_key][cat])
                        print(f"[NEU-DET] 类别 {cat} 下有 {sample_count} 个样本")
                        # 打印前几个样本ID
                        for i, sample_item in enumerate(all_datasets_results["NEU-DET"][shot_key][cat][:3]):
                            if "sample_id" in sample_item:
                                print(f"[NEU-DET] - 样本 {i+1}: {sample_item['sample_id']}")
                            else:
                                print(f"[NEU-DET] - 样本 {i+1}: {sample_item.keys()}")
    
    # 找出所有特殊样本（pitted_surface_和rolled-in_scale_开头）
    special_samples = []
    for sample_name in sample_names:
        if sample_name.startswith("pitted_surface_") or sample_name.startswith("rolled-in_scale_"):
            special_samples.append(sample_name)
    
    if special_samples:
        print(f"\n[NEU-DET] 找到 {len(special_samples)} 个特殊样本:")
        for sample in special_samples[:10]:  # 只打印前10个
            print(f"[NEU-DET] - 特殊样本: {sample}")
    
    # 遍历所有k-shot样本
    for sample_name in tqdm(sample_names, desc=f"处理 {dataset_name} {shot_number}-shot样本"):
        # 获取目标图像路径 - 使用指定的shot_dir
        target_image_path = os.path.join(shot_dir, f"{sample_name}.jpg")
        if not os.path.exists(target_image_path):
            print(f"跳过样本 {sample_name}，因为找不到目标图像")
            failed_count += 1
            continue
        
        # 创建样本输出目录
        sample_output_dir = os.path.join(output_base_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # 特殊处理NEU-DET的pitted_surface和rolled-in_scale样本
        is_special_sample = False
        if dataset_name == "NEU-DET" and (sample_name.startswith("pitted_surface_") or sample_name.startswith("rolled-in_scale_")):
            is_special_sample = True
            print(f"\n[NEU-DET特殊样本] 开始处理: {sample_name}")
            
            # 确定类别
            if sample_name.startswith("pitted_surface_"):
                category = "pitted_surface"
            else:
                category = "rolled-in_scale"
                
            print(f"[NEU-DET特殊样本] 样本 {sample_name} 属于类别 {category}")
            
            # 检查类别是否存在于检索结果中
            shot_key = f"{shot_number}_shot"
            if "NEU-DET" in all_datasets_results and shot_key in all_datasets_results["NEU-DET"]:
                if category in all_datasets_results["NEU-DET"][shot_key]:
                    print(f"[NEU-DET特殊样本] 类别 {category} 存在于检索结果中")
                    # 打印类别下的样本数量
                    sample_count = len(all_datasets_results["NEU-DET"][shot_key][category])
                    print(f"[NEU-DET特殊样本] 类别 {category} 下有 {sample_count} 个样本")
                else:
                    print(f"[NEU-DET特殊样本] 警告: 类别 {category} 不存在于检索结果中")
                    print(f"[NEU-DET特殊样本] 可用类别: {list(all_datasets_results['NEU-DET'][shot_key].keys())}")
        
        # 获取top5相似图像
        try:
            # 打印详细调试信息，帮助诊断NEU-DET问题
            if dataset_name == "NEU-DET":
                print(f"\n[调试] 开始处理NEU-DET样本: {sample_name}")
                print(f"[调试] 当前shot_number: {shot_number}")
                shot_key = f"{shot_number}_shot"
                if "NEU-DET" in all_datasets_results and shot_key in all_datasets_results["NEU-DET"]:
                    print(f"[调试] NEU-DET/{shot_key} 中的类别: {list(all_datasets_results['NEU-DET'][shot_key].keys())}")
                    
                    # 分析样本名称以提取可能的类别
                    possible_categories = []
                    if "-" in sample_name:
                        # 处理带连字符的情况，如"rolled-in_scale_14"
                        hyphen_parts = sample_name.split("-")
                        possible_categories.append(hyphen_parts[0])  # 第一部分
                        if "_" in sample_name:
                            underscore_pos = sample_name.rfind("_")
                            if underscore_pos > 0:
                                possible_categories.append(sample_name[:underscore_pos])  # 数字前的部分
                    elif "_" in sample_name:
                        # 处理标准情况，如"inclusion_106"
                        underscore_parts = sample_name.split("_")
                        possible_categories.append(underscore_parts[0])  # 第一部分
                    
                    print(f"[调试] 从样本名称'{sample_name}'提取的可能类别: {possible_categories}")
                    
                    # 检查这些可能的类别是否存在于检索结果中
                    for cat in possible_categories:
                        if cat in all_datasets_results["NEU-DET"][shot_key]:
                            print(f"[调试] 类别'{cat}'存在于检索结果中")
                        else:
                            print(f"[调试] 类别'{cat}'不存在于检索结果中")
                            
                            # 尝试部分匹配
                            for existing_cat in all_datasets_results["NEU-DET"][shot_key].keys():
                                if cat.lower() in existing_cat.lower() or existing_cat.lower() in cat.lower():
                                    print(f"[调试] 类别'{cat}'与现有类别'{existing_cat}'部分匹配")
            
            top5_images = get_top5_similar_images_from_json(all_datasets_results, sample_name, dataset_name, shot_number)
            if not top5_images:
                print(f"跳过样本 {sample_name}，因为找不到相似图像")
                failed_count += 1
                
                # 记录错误信息到文件
                error_file = os.path.join(sample_output_dir, "error.txt")
                with open(error_file, "w") as f:
                    f.write(f"处理样本 {sample_name} 时出错: 找不到相似图像\n")
                    f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                continue
        except ValueError as e:
            # 捕获NEU-DET特殊处理的错误
            print(f"处理样本 {sample_name} 时出错: {str(e)}")
            
            # 记录错误信息到文件
            error_file = os.path.join(sample_output_dir, "error.txt")
            with open(error_file, "w") as f:
                f.write(f"处理样本 {sample_name} 时出错: {str(e)}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # 额外记录特殊样本信息
                if is_special_sample:
                    f.write(f"\n这是一个特殊样本: {sample_name}\n")
                    if "NEU-DET" in all_datasets_results and shot_key in all_datasets_results["NEU-DET"]:
                        f.write(f"可用类别: {list(all_datasets_results['NEU-DET'][shot_key].keys())}\n")
            
            failed_count += 1
            continue
        except Exception as e:
            # 捕获其他所有异常
            error_message = f"处理样本 {sample_name} 时发生未知错误: {str(e)}"
            print(error_message)
            
            # 记录错误信息到文件
            error_file = os.path.join(sample_output_dir, "error.txt")
            with open(error_file, "w") as f:
                f.write(f"{error_message}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # 记录更详细的异常信息
                import traceback
                traceback.print_exc()
                f.write("\n详细错误信息:\n")
                f.write(traceback.format_exc())
            
            failed_count += 1
            continue
        
        # 使用top5相似图像生成
        sample_success = False
        for i, (similarity, ref_image_path, rank) in enumerate(top5_images):
            # 确保只处理rank 1-5的图像
            if rank > 5:
                continue
                
            output_path = os.path.join(sample_output_dir, f"generated_image_rank{rank}.png")
            success = generate_image(
                pipe_prior_redux, pipe, ref_image_path, target_image_path, 
                output_path, rank=rank, similarity=similarity
            )
            
            if success:
                print(f"成功生成样本 {sample_name} 的图像 (rank {rank})")
                total_generated_images += 1
                sample_success = True
                
                # 记录图像尺寸
                try:
                    size_file = os.path.join(sample_output_dir, f"ref_info{rank}.txt")
                    if os.path.exists(size_file):
                        with open(size_file, "r") as size_f:
                            for line in size_f:
                                if "生成图像尺寸:" in line:
                                    size_str = line.strip().split(": ")[1]
                                    if size_str not in image_sizes:
                                        image_sizes[size_str] = 1
                                    else:
                                        image_sizes[size_str] += 1
                                    break
                except Exception as e:
                    print(f"记录图像尺寸时出错: {str(e)}")
            else:
                print(f"生成样本 {sample_name} 的图像失败 (rank {rank})")
        
        if sample_success:
            success_count += 1
        else:
            failed_count += 1
            
            # 记录生成失败信息
            error_file = os.path.join(sample_output_dir, "generation_failed.txt")
            with open(error_file, "w") as f:
                f.write(f"生成样本 {sample_name} 的图像失败\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if top5_images:
                    f.write(f"找到了 {len(top5_images)} 个相似图像，但生成全部失败\n")
                    for i, (sim, path, rank) in enumerate(top5_images):
                        f.write(f"  - Rank {rank}: {path} (相似度: {sim:.4f})\n")
    
    # 更新处理日志
    with open(os.path.join(output_base_dir, "batch_params.txt"), "a") as f:
        f.write(f"成功处理样本数: {success_count}\n")
        f.write(f"失败处理样本数: {failed_count}\n")
        f.write(f"总共生成图像数: {total_generated_images}\n")
        
        # 记录各种图像尺寸及其数量
        f.write(f"\n生成图像尺寸统计:\n")
        for size_str, count in sorted(image_sizes.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {size_str}: {count}张图像\n")
            
        f.write(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"数据集 {dataset_name} {shot_number}-shot处理完成：成功 {success_count} 个样本，失败 {failed_count} 个样本，总共生成 {total_generated_images} 张图像")

def find_matching_key(all_datasets_results, sample_name):
    """尝试在检索结果中找到匹配的键，处理可能的键名不匹配问题"""
    # 检查当前数据集/5_shot下
    if kshot_dataset_name in all_datasets_results and "5_shot" in all_datasets_results[kshot_dataset_name]:
        if sample_name in all_datasets_results[kshot_dataset_name]["5_shot"]:
            print(f"在{kshot_dataset_name}/5_shot下找到样本 {sample_name}")
            return f"{kshot_dataset_name}.5_shot.{sample_name}"
    
    # 直接匹配
    if sample_name in all_datasets_results:
        return sample_name
    
    # 检查当前数据集下
    if kshot_dataset_name in all_datasets_results and sample_name in all_datasets_results[kshot_dataset_name]:
        return f"{kshot_dataset_name}.{sample_name}"
    
    # 尝试不同的大小写形式
    variants = [
        sample_name.upper(),
        sample_name.lower(),
        sample_name.capitalize()
    ]
    
    # 检查所有可能的路径
    for variant in variants:
        # 直接在顶级
        if variant in all_datasets_results:
            print(f"找到样本 {sample_name} 的变体: {variant}")
            return variant
            
        # 在当前数据集下
        if kshot_dataset_name in all_datasets_results:
            if variant in all_datasets_results[kshot_dataset_name]:
                print(f"在{kshot_dataset_name}下找到样本 {sample_name} 的变体: {variant}")
                return f"{kshot_dataset_name}.{variant}"
                
            # 在当前数据集/5_shot下
            if "5_shot" in all_datasets_results[kshot_dataset_name]:
                if variant in all_datasets_results[kshot_dataset_name]["5_shot"]:
                    print(f"在{kshot_dataset_name}/5_shot下找到样本 {sample_name} 的变体: {variant}")
                    return f"{kshot_dataset_name}.5_shot.{variant}"
    
    # 没有找到匹配
    return None

def get_top5_similar_images_from_json(all_datasets_results, sample_name, dataset_name=None, shot_number=5):
    """从JSON检索结果中获取top5相似图像"""
    if dataset_name is None:
        dataset_name = kshot_dataset_name
        
    shot_key = f"{shot_number}_shot"
    sample_results = None
    
    # 特殊处理COCO数据集
    if dataset_name == "coco":
        print(f"检测到COCO数据集，使用特殊处理")
        try:
            sample_results = find_coco_sample(all_datasets_results, sample_name, shot_number)
            if sample_results is None:
                # 提供更详细的错误信息
                error_msg = f"无法在COCO数据集中找到样本 {sample_name}"
                if "coco" in all_datasets_results and shot_key in all_datasets_results["coco"]:
                    available_samples = list(all_datasets_results["coco"][shot_key].keys())[:10]
                    error_msg += f"，可用样本示例: {available_samples}"
                print(f"错误: {error_msg}")
                raise ValueError(error_msg)
        except Exception as e:
            # 捕获所有异常，提供详细错误信息
            error_msg = f"无法在COCO数据集中找到样本 {sample_name}"
            print(f"错误: {error_msg} ({str(e)})")
            raise ValueError(error_msg)
    # 特殊处理NEU-DET数据集
    elif dataset_name == "NEU-DET":
        print(f"检测到NEU-DET数据集，使用特殊处理")
        try:
            sample_results = find_neudet_sample(all_datasets_results, sample_name, shot_number)
            if sample_results is None:
                # 提供更详细的错误信息
                error_msg = f"无法在NEU-DET数据集中找到样本 {sample_name}"
                if "NEU-DET" in all_datasets_results and shot_key in all_datasets_results["NEU-DET"]:
                    available_categories = list(all_datasets_results["NEU-DET"][shot_key].keys())
                    error_msg += f"，可用类别: {available_categories}"
                print(f"错误: {error_msg}")
                raise ValueError(error_msg)
        except Exception as e:
            # 捕获所有异常，提供详细错误信息
            error_msg = f"无法在NEU-DET数据集中找到样本 {sample_name}"
            print(f"错误: {error_msg} ({str(e)})")
            raise ValueError(error_msg)
    else:
        # 标准处理其他数据集
        # 尝试找到匹配的样本
        if dataset_name in all_datasets_results:
            # 首先检查数据集是否有shot_key子目录
            if shot_key in all_datasets_results[dataset_name]:
                # 标准结构: dataset -> shot_key -> sample
                if sample_name in all_datasets_results[dataset_name][shot_key]:
                    sample_results = all_datasets_results[dataset_name][shot_key][sample_name]
                    print(f"在{dataset_name}/{shot_key}下找到样本 {sample_name}")
                else:
                    # 打印部分键以便调试
                    print(f"警告：在{dataset_name}/{shot_key}下找不到样本 {sample_name}")
                    if len(all_datasets_results[dataset_name][shot_key]) > 0:
                        print(f"{dataset_name}/{shot_key}下的一些键：", list(all_datasets_results[dataset_name][shot_key].keys())[:10])
            
            # 如果在shot_key子目录下没找到，尝试直接在数据集下查找样本
            if sample_results is None and sample_name in all_datasets_results[dataset_name]:
                # 替代结构: dataset -> sample (NEU-DET这样的数据集)
                sample_results = all_datasets_results[dataset_name][sample_name]
                print(f"在{dataset_name}直接找到样本 {sample_name} (无shot子目录)")
            
            # 如果仍然没找到，尝试查找不同版本的样本名称
            if sample_results is None:
                # 尝试不同的样本名称格式
                sample_variants = [
                    sample_name.replace("-", "_"),  # 处理可能的连字符/下划线差异
                    sample_name.replace("_", "-"),
                    sample_name.lower(),            # 尝试不同的大小写
                    sample_name.upper(),
                    sample_name.capitalize()
                ]
                
                # 在shot_key子目录中查找变体
                if shot_key in all_datasets_results[dataset_name]:
                    for variant in sample_variants:
                        if variant in all_datasets_results[dataset_name][shot_key]:
                            sample_results = all_datasets_results[dataset_name][shot_key][variant]
                            print(f"在{dataset_name}/{shot_key}下找到样本变体 {variant}")
                            break
                
                # 直接在数据集下查找变体
                if sample_results is None:
                    for variant in sample_variants:
                        if variant in all_datasets_results[dataset_name]:
                            sample_results = all_datasets_results[dataset_name][variant]
                            print(f"在{dataset_name}下直接找到样本变体 {variant}")
                            break
    
    # 如果所有尝试都失败，处理失败情况
    if sample_results is None:
        if dataset_name in all_datasets_results:
            print(f"警告：在{dataset_name}下找不到样本 {sample_name} 或其变体")
            keys_info = f"数据集顶级键: {list(all_datasets_results[dataset_name].keys())[:10]}"
            print(keys_info)
        else:
            print(f"警告：在检索结果中找不到数据集 {dataset_name}")
            
        # 对于NEU-DET数据集，如果特殊处理失败则报错
        if dataset_name == "NEU-DET":
            error_msg = f"无法在NEU-DET数据集中找到样本 {sample_name} 或其类别"
            print(f"错误: {error_msg}")
            raise ValueError(error_msg)
        
        # 根据当前数据库类型选择随机图像
        if DATABASE_TYPE == "coco":
            print(f"将为样本 {sample_name} 使用随机COCO图像")
            random_images = get_random_coco_images(5)  # 获取5张随机COCO图片
        else:
            print(f"将为样本 {sample_name} 使用随机miniImageNet图像")
            random_images = get_random_miniimagenet_images(5)  # 需要实现此函数
            
        if random_images:
            top5_images = []
            for i, path in enumerate(random_images):
                # 只使用rank 1-5的图像
                if i < 5:
                    top5_images.append((1.0 - i*0.1, path, i+1))  # 假设相似度递减
            return top5_images
        return []
    
    # 检查是否是列表形式
    if isinstance(sample_results, list) and len(sample_results) > 0:
        # 使用第一个结果
        sample_result = sample_results[0]
    else:
        # 如果不是列表，直接使用
        sample_result = sample_results
    
    # 获取相似图像列表
    similar_images = []
    
    # 处理sample_result可能是字典或嵌套结构的情况
    if isinstance(sample_result, dict):
        # 尝试直接获取similar_images
        if "similar_images" in sample_result:
            similar_images = sample_result.get("similar_images", [])
        
        # 如果没有找到similar_images，尝试检查是否有其他包含similar_images的键
        if not similar_images:
            for key, value in sample_result.items():
                if isinstance(value, dict) and "similar_images" in value:
                    similar_images = value.get("similar_images", [])
                    if similar_images:
                        break
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "similar_images" in item:
                            similar_images = item.get("similar_images", [])
                            if similar_images:
                                break
                    if similar_images:
                        break
    
    if not similar_images:
        print(f"警告：样本 {sample_name} 没有相似图像")
        # 打印样本结果的键，帮助调试
        if isinstance(sample_result, dict):
            print(f"样本结果键: {list(sample_result.keys())}")
        
        # 对于NEU-DET数据集，如果没有找到相似图像则报错
        if dataset_name == "NEU-DET":
            error_msg = f"在NEU-DET数据集的样本 {sample_name} 中找不到相似图像"
            print(f"错误: {error_msg}")
            # 添加更多调试信息
            if isinstance(sample_result, dict):
                print(f"样本结果: {sample_result}")
            raise ValueError(error_msg)
        
        return []
    
    # 提取top5图像路径和相似度
    top5_images = []
    for img in similar_images:
        if not isinstance(img, dict):
            continue
            
        rank = img.get("rank", 0)
        
        # 只处理rank为1-5的图像
        if rank > 5:
            continue
            
        similarity = img.get("similarity", 0)
        image_path = img.get("image_path", "")
        
        if not image_path:
            continue
        
        # 使用get_correct_image_path获取正确的图像路径
        correct_path = get_correct_image_path(image_path)
        
        # 检查文件是否存在
        if correct_path and os.path.exists(correct_path):
            top5_images.append((similarity, correct_path, rank))
        else:
            print(f"警告：找不到图像文件 {image_path}，修正后的路径: {correct_path}")
    
    # 按rank排序 (确保rank从小到大排序)
    top5_images.sort(key=lambda x: x[2])
    
    if top5_images:
        print(f"为样本 {sample_name} 找到 {len(top5_images)} 个相似图像 (仅rank 1-5)")
        for i, (sim, path, rank) in enumerate(top5_images):
            print(f"  - Rank {rank}: {os.path.basename(path)} (相似度: {sim:.4f})")
        return top5_images
    else:
        error_msg = f"警告：为样本 {sample_name} 找不到有效的相似图像 (rank 1-5)"
        print(error_msg)
        
        # 对于NEU-DET数据集，如果没有找到有效相似图像则报错
        if dataset_name == "NEU-DET":
            error_msg = f"在NEU-DET数据集的样本 {sample_name} 中找不到有效的相似图像 (rank 1-5)"
            print(f"错误: {error_msg}")
            # 添加更多调试信息
            if similar_images:
                print(f"找到的相似图像数量: {len(similar_images)}")
                for i, img in enumerate(similar_images[:3]):  # 只打印前3个
                    print(f"  相似图像 {i+1}: {img}")
            raise ValueError(error_msg)
        
        return []

def get_correct_image_path(image_path):
    """根据数据库类型获取正确的图像路径"""
    # 如果路径为空，返回None
    if not image_path:
        return None
    
    # 首先尝试原始路径
    if os.path.exists(image_path):
        return image_path
    
    # 处理 /DATA_HDD/ly/Flux/retrieval/datasets/ 路径转换
    if "/DATA_HDD/ly/Flux/retrieval/datasets/" in image_path:
        datasets_path = image_path.replace("/DATA_HDD/ly/Flux/retrieval/datasets/", "./datasets/")
        if os.path.exists(datasets_path):
            return datasets_path
        # 如果转换后的路径也不存在，继续后续处理
    
    # 获取文件名
    filename = os.path.basename(image_path)
    
    # 处理COCO路径
    if "coco/train2017" in image_path:
        if DATABASE_TYPE == "coco":
            # 使用COCO数据库
            return os.path.join(coco_dataset_dir, filename)
        else:
            # 尝试查找对应的miniImageNet图像（这可能需要基于检索结果进行更复杂的映射）
            print(f"警告：当前使用miniImageNet数据库，但检索结果包含COCO路径: {image_path}")
            # 如果有明确的COCO到miniImageNet映射，可以在这里实现
            return None
    
    # 处理miniImageNet路径
    if "miniimagenet/train" in image_path:
        if DATABASE_TYPE == "miniimagenet":
            # 使用fix_miniimagenet_path函数处理路径
            return fix_miniimagenet_path(image_path)
        else:
            # 尝试查找对应的COCO图像
            print(f"警告：当前使用COCO数据库，但检索结果包含miniImageNet路径: {image_path}")
            # 如果有明确的miniImageNet到COCO映射，可以在这里实现
            return None
    
    # 处理以../或./开头的路径
    if image_path.startswith("../") or image_path.startswith("./"):
        # 可能是相对于工作目录的路径
        rel_path = image_path
        if image_path.startswith("../"):
            # 将../替换为./
            rel_path = "./" + image_path[3:]
        
        if os.path.exists(rel_path):
            return rel_path
    
    # 原始路径检查已在函数开头处理，这里不再重复检查
    
    # 尝试在当前数据库目录下查找
    if DATABASE_TYPE == "coco":
        coco_path = os.path.join(coco_dataset_dir, filename)
        if os.path.exists(coco_path):
            return coco_path
    else:
        # 对于miniImageNet，可能需要更复杂的处理
        # 尝试使用fix_miniimagenet_path函数修复路径
        fixed_path = fix_miniimagenet_path(image_path)
        if fixed_path and os.path.exists(fixed_path):
            return fixed_path
            
        # 搜索整个目录树
        for root, dirs, files in os.walk(miniimagenet_dataset_dir):
            if filename in files:
                return os.path.join(root, filename)
    
    # 最后，如果找不到，返回None
    print(f"警告：找不到图像文件 {filename}（原路径：{image_path}）")
    return None

def ensure_miniimagenet_directory():
    """确保miniImageNet目录结构存在，如果不存在则创建"""
    # 检查miniImageNet目录是否存在
    if not os.path.exists(miniimagenet_dataset_dir):
        print(f"警告：miniImageNet目录不存在，正在创建: {miniimagenet_dataset_dir}")
        os.makedirs(miniimagenet_dataset_dir, exist_ok=True)
    
    # 检查train子目录是否存在
    train_dir = os.path.join(miniimagenet_dataset_dir, "train")
    if not os.path.exists(train_dir):
        print(f"警告：miniImageNet训练目录不存在，正在创建: {train_dir}")
        os.makedirs(train_dir, exist_ok=True)
    
    print(f"miniImageNet目录结构已检查/创建")
    return True

def fix_miniimagenet_path(image_path):
    """修复miniImageNet图像路径，确保可以找到图像文件"""
    # 确保目录结构存在
    ensure_miniimagenet_directory()
    
    # 如果路径为空，返回None
    if not image_path:
        return None
    
    # 获取文件名
    filename = os.path.basename(image_path)
    
    # 可能的路径前缀列表
    possible_prefixes = [
        "./retrieval/miniimagenet/train/",
        "retrieval/miniimagenet/train/",
        "./miniimagenet/train/",
        "miniimagenet/train/",
        "./retrieval/train/",
        "retrieval/train/"
    ]
    
    # 处理miniImageNet路径
    if "miniimagenet/train" in image_path or "train" in image_path:
        # 尝试直接使用路径
        if os.path.exists(image_path):
            return image_path
            
        # 如果image_path以任何前缀开头，提取类别和文件名部分
        class_name = None
        for prefix in possible_prefixes:
            if image_path.startswith(prefix):
                path_without_prefix = image_path[len(prefix):]
                parts = path_without_prefix.split('/')
                if parts:
                    class_name = parts[0]
                    if len(parts) > 1:
                        filename = parts[1]
                    break
        
        # 如果通过前缀方式没找到，通过train关键字定位
        if class_name is None:
            parts = image_path.split('/')
            for i, part in enumerate(parts):
                if part == "train" and i+1 < len(parts):
                    class_name = parts[i+1]
                    if i+2 < len(parts):
                        filename = parts[i+2]
                    break
        
        # 如果找到了类别和文件名
        if class_name:
            # 确保类别目录存在
            class_dir = os.path.join(miniimagenet_dataset_dir, "train", class_name)
            if not os.path.exists(class_dir):
                print(f"正在创建类别目录: {class_dir}")
                os.makedirs(class_dir, exist_ok=True)
            
            # 构建目标路径
            target_path = os.path.join(class_dir, filename)
            
            # 检查文件是否存在
            if os.path.exists(target_path):
                return target_path
            else:
                # 文件不存在，尝试其他位置
                print(f"警告：找不到miniImageNet图像文件: {target_path}")
                
                # 尝试在当前工作目录下查找
                for prefix in possible_prefixes:
                    test_path = os.path.join(".", prefix, class_name, filename)
                    if os.path.exists(test_path):
                        print(f"在路径 {test_path} 找到图像")
                        return test_path
                
                # 尝试不同的路径组合
                absolute_path = os.path.abspath(image_path)
                if os.path.exists(absolute_path):
                    print(f"在绝对路径找到图像: {absolute_path}")
                    return absolute_path
                
                # 尝试直接查找文件名
                for root, dirs, files in os.walk(miniimagenet_dataset_dir):
                    if filename in files:
                        found_path = os.path.join(root, filename)
                        print(f"通过文件名在 {found_path} 找到图像")
                        return found_path
                
                # 如果以上都失败，尝试类似文件名的文件
                for root, dirs, files in os.walk(miniimagenet_dataset_dir):
                    for file in files:
                        # 检查文件名是否相似
                        if filename.split('.')[0] in file:
                            found_path = os.path.join(root, file)
                            print(f"找到类似文件名的图像: {found_path}")
                            return found_path
                
                # 如果所有尝试都失败，返回None
                print(f"无法找到任何匹配的图像文件: {filename}")
                return None
    
    # 如果不是miniImageNet路径，直接返回原始路径
    return image_path

def extract_miniimagenet_classes_from_retrieval(all_datasets_results):
    """从检索结果中提取miniImageNet类别信息"""
    classes = set()
    
    # 遍历检索结果
    for dataset_key, dataset_value in all_datasets_results.items():
        if isinstance(dataset_value, dict):
            for shot_key, shot_value in dataset_value.items():
                if isinstance(shot_value, dict):
                    for sample_key, sample_value in shot_value.items():
                        # 处理各种可能的结构
                        if isinstance(sample_value, dict):
                            process_sample_value(sample_value, classes)
                        elif isinstance(sample_value, list):
                            for item in sample_value:
                                if isinstance(item, dict):
                                    process_sample_value(item, classes)
    
    print(f"从检索结果中提取到 {len(classes)} 个miniImageNet类别")
    # 创建这些类别的目录
    for class_name in classes:
        class_dir = os.path.join(miniimagenet_dataset_dir, "train", class_name)
        if not os.path.exists(class_dir):
            print(f"创建miniImageNet类别目录: {class_dir}")
            os.makedirs(class_dir, exist_ok=True)
    
    return classes

def process_sample_value(sample_dict, classes):
    """处理样本值，提取miniImageNet类别"""
    # 直接查找similar_images
    if "similar_images" in sample_dict:
        process_similar_images(sample_dict["similar_images"], classes)
    
    # 遍历其他键值对
    for key, value in sample_dict.items():
        if isinstance(value, dict) and "similar_images" in value:
            process_similar_images(value["similar_images"], classes)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "similar_images" in item:
                    process_similar_images(item["similar_images"], classes)

def process_similar_images(similar_images, classes):
    """处理相似图像列表，提取miniImageNet类别"""
    if not isinstance(similar_images, list):
        return
    
    for img in similar_images:
        if not isinstance(img, dict) or "image_path" not in img:
            continue
        
        image_path = img["image_path"]
        if "miniimagenet/train" in image_path:
            # 尝试提取类别
            parts = image_path.split("/")
            for i, part in enumerate(parts):
                if part == "train" and i+1 < len(parts):
                    class_name = parts[i+1]
                    classes.add(class_name)
                    break

def find_coco_sample(all_datasets_results, sample_name, shot_number):
    """专门用于处理COCO数据集的样本检索
    
    Args:
        all_datasets_results: 所有数据集的检索结果
        sample_name: 样本名称，如 "000000382438"
        shot_number: shot数量，如 1, 5, 10
        
    Returns:
        找到的样本结果，如果找不到则返回None
    """
    shot_key = f"{shot_number}_shot"
    
    # 检查COCO数据集是否存在
    if "coco" not in all_datasets_results:
        print(f"错误：在检索结果中找不到coco数据集")
        return None
    
    # 检查shot_key是否存在
    if shot_key not in all_datasets_results["coco"]:
        print(f"错误：在coco数据集中找不到{shot_key}子目录")
        return None
    
    # COCO样本直接查找
    if sample_name in all_datasets_results["coco"][shot_key]:
        sample_data = all_datasets_results["coco"][shot_key][sample_name]
        print(f"在coco/{shot_key}下找到样本 {sample_name}")
        
        # COCO检索结果是列表形式，取第一个元素
        if isinstance(sample_data, list) and len(sample_data) > 0:
            return sample_data[0]
        else:
            return sample_data
    
    # 如果直接查找失败，尝试不同的样本名称格式
    sample_variants = [
        sample_name.zfill(12),  # 补齐到12位，如 "000000382438"
        sample_name.lstrip('0'),  # 去除前导零，如 "382438"
        f"00000{sample_name}",  # 添加前导零
        sample_name.upper(),
        sample_name.lower()
    ]
    
    for variant in sample_variants:
        if variant in all_datasets_results["coco"][shot_key]:
            sample_data = all_datasets_results["coco"][shot_key][variant]
            print(f"在coco/{shot_key}下找到样本变体 {variant} (原样本名: {sample_name})")
            
            # COCO检索结果是列表形式，取第一个元素
            if isinstance(sample_data, list) and len(sample_data) > 0:
                return sample_data[0]
            else:
                return sample_data
    
    print(f"错误：在coco/{shot_key}下没有找到样本 {sample_name} 或其任何变体")
    return None

def find_neudet_sample(all_datasets_results, sample_name, shot_number):
    """专门用于处理NEU-DET数据集的样本检索
    
    Args:
        all_datasets_results: 所有数据集的检索结果
        sample_name: 样本名称，如 "inclusion_106" 或 "rolled-in_scale_14"
        shot_number: shot数量，如 1, 5, 10
        
    Returns:
        找到的样本结果，如果找不到则返回None
    """
    shot_key = f"{shot_number}_shot"
    
    # 检查NEU-DET数据集是否存在
    if "NEU-DET" not in all_datasets_results:
        print(f"错误：在检索结果中找不到NEU-DET数据集")
        return None
    
    # 检查shot_key是否存在
    if shot_key not in all_datasets_results["NEU-DET"]:
        print(f"错误：在NEU-DET数据集中找不到{shot_key}子目录")
        return None
    
    # NEU-DET特殊类别处理 - 直接判断样本名称的前缀
    # 处理确切的类别名称
    if sample_name.startswith("pitted_surface_"):
        # pitted_surface类别
        category = "pitted_surface"
        sample_id = sample_name
        print(f"识别到pitted_surface类别样本: {sample_name}")
    elif sample_name.startswith("rolled-in_scale_"):
        # rolled-in_scale类别
        category = "rolled-in_scale"
        sample_id = sample_name
        print(f"识别到rolled-in_scale类别样本: {sample_name}")
    else:
        # 从样本名称提取类别（处理复杂格式，包括带连字符的名称）
        if "-" in sample_name and "_" in sample_name:
            # 先尝试找到最后一个下划线
            last_underscore_pos = sample_name.rfind('_')
            if last_underscore_pos > 0:
                category = sample_name[:last_underscore_pos]
                number = sample_name[last_underscore_pos+1:]
                print(f"从复杂样本名称中提取：类别={category}, 编号={number}")
            else:
                parts = sample_name.split('_')
                if len(parts) < 2:
                    print(f"错误：样本名称 {sample_name} 格式不正确，无法解析")
                    return None
                category = parts[0]
        else:
            # 标准处理（用下划线分割）
            parts = sample_name.split('_')
            if len(parts) < 2:
                print(f"错误：样本名称 {sample_name} 格式不正确，应为'类别_编号'格式")
                return None
            category = parts[0]
    
    # 特殊处理：直接检查确切的类别名称
    special_categories = ["pitted_surface", "rolled-in_scale", "crazing", "inclusion", "patches", "scratches"]
    if category in special_categories:
        print(f"使用精确匹配的类别名称: {category}")
    else:
        # 尝试多种类别名称变体，处理类别名可能的不同表示形式
        category_variants = [
            category,
            category.replace("-", "_"),
            category.replace("_", "-"),
            category.lower(),
            category.upper(),
            category.replace("-", " ").replace("_", " ")  # 尝试空格替换
        ]
        
        # 特殊处理：常见NEU-DET类别名称映射
        special_mappings = {
            "pitted_surface": ["pitted-surface", "pitted surface", "pittedSurface", "pitted"],
            "rolled-in_scale": ["rolled_in_scale", "rolled in scale", "rolledinscale", "rolled"],
            "patches": ["patch"],
            "scratches": ["scratch"],
            "crazing": ["craze"],
            "inclusion": ["inclusions"]
        }
        
        # 添加特殊映射的变体
        for key, variants in special_mappings.items():
            if category.lower() in key.lower() or key.lower() in category.lower():
                category_variants.extend(variants)
                category_variants.append(key)  # 添加标准形式
        
        # 打印所有将尝试的类别变体
        print(f"将尝试以下类别变体: {category_variants}")
        
        # 尝试所有类别变体
        for cat_variant in category_variants:
            if cat_variant in all_datasets_results["NEU-DET"][shot_key]:
                category = cat_variant
                print(f"找到匹配的类别变体: {category}")
                break
    
    # 检查类别是否存在
    if category not in all_datasets_results["NEU-DET"][shot_key]:
        # 打印所有可用类别以便调试
        available_categories = list(all_datasets_results["NEU-DET"][shot_key].keys())
        print(f"错误：在NEU-DET/{shot_key}下找不到类别 {category} 或其任何变体")
        print(f"可用类别: {available_categories}")
        
        # 尝试查找部分匹配的类别
        matched_category = None
        for avail_cat in available_categories:
            if (category.lower() in avail_cat.lower() or 
                avail_cat.lower() in category.lower()):
                matched_category = avail_cat
                print(f"找到部分匹配的类别: {matched_category}")
                break
        
        if matched_category:
            category = matched_category
        else:
            # 最后尝试特殊类别的直接匹配
            if "pitted_surface" in available_categories and "pitted" in sample_name.lower():
                category = "pitted_surface"
                print(f"基于样本名称包含'pitted'使用类别: {category}")
            elif "rolled-in_scale" in available_categories and "rolled" in sample_name.lower():
                category = "rolled-in_scale"
                print(f"基于样本名称包含'rolled'使用类别: {category}")
            else:
                return None
    
    # 类别存在，在样本列表中查找匹配的sample_id
    samples_list = all_datasets_results["NEU-DET"][shot_key][category]
    
    # 提取样本编号
    sample_number = None
    if "_" in sample_name:
        sample_number = sample_name.split("_")[-1]
    
    # 首先尝试找到完全匹配的样本ID
    for sample_item in samples_list:
        if "sample_id" in sample_item and sample_item["sample_id"] == sample_name:
            print(f"在NEU-DET/{shot_key}/{category}下找到样本 {sample_name}")
            return sample_item
    
    # 如果没有找到完全匹配的样本ID，尝试使用部分匹配
    if sample_number:
        for sample_item in samples_list:
            if "sample_id" in sample_item and sample_number in sample_item["sample_id"]:
                print(f"在NEU-DET/{shot_key}/{category}下找到部分匹配的样本 {sample_item['sample_id']}")
                return sample_item
    
    # 尝试更灵活的匹配，查找包含样本名称任何部分的样本
    sample_parts = []
    if "-" in sample_name:
        sample_parts.extend(sample_name.split("-"))
    if "_" in sample_name:
        sample_parts.extend(sample_name.split("_"))
    
    for part in sample_parts:
        if len(part) < 3 or part.isdigit():  # 跳过短字符串和纯数字
            continue
        for sample_item in samples_list:
            if "sample_id" in sample_item and part in sample_item["sample_id"]:
                print(f"在NEU-DET/{shot_key}/{category}下找到包含关键部分 '{part}' 的样本 {sample_item['sample_id']}")
                return sample_item
    
    # 如果仍未找到样本，使用类别中的第一个样本
    if samples_list:
        first_sample = samples_list[0]
        print(f"在NEU-DET/{shot_key}/{category}下未找到样本 {sample_name}，使用该类别下的第一个样本 {first_sample.get('sample_id', '未知')}")
        return first_sample
    
    print(f"错误：在NEU-DET/{shot_key}/{category}下没有找到任何样本")
    return None

def main():
    print("开始批量生成图像...")
    
    # 解析命令行参数
    args = parse_arguments()
    dataset_name = args.dataset  # 不使用默认值，让dataset_group逻辑正常工作
    shot_numbers = args.shots if args.shots else [5]  # 默认处理5-shot
    output_dir = args.output_dir  # 使用命令行参数指定的输出目录
    database_type = args.database  # 使用命令行参数指定的数据库类型
    dataset_group = args.dataset_group  # 使用命令行参数指定的数据集分组
    retrieval_results_dir_arg = args.retrieval_results_dir  # 使用命令行参数指定的检索结果目录
    
    # 设置全局数据库类型
    global DATABASE_TYPE
    DATABASE_TYPE = database_type
    
    # 如果使用miniImageNet数据库，确保目录结构存在
    if DATABASE_TYPE == "miniimagenet":
        ensure_miniimagenet_directory()
    
    # 确定要处理的数据集
    if dataset_name:
        # 如果指定了特定数据集，只处理该数据集
        datasets_to_process = [dataset_name]
    elif dataset_group:
        # 如果指定了数据集分组，处理该分组中的所有数据集
        datasets_to_process = dataset_groups.get(dataset_group, [])
        if not datasets_to_process:
            print(f"错误：找不到数据集分组 {dataset_group} 或该分组为空")
            return
    else:
        # 否则处理所有数据集
        datasets_to_process = datasets
    
    print(f"将处理以下数据集: {', '.join(datasets_to_process)}")
    print(f"处理的shot数量: {', '.join([str(s) for s in shot_numbers])}")
    print(f"结果将保存到: {output_dir}")
    print(f"使用数据库: {database_type}")
    if dataset_group:
        print(f"使用数据集分组: {dataset_group}")
    print(f"数据库路径: {coco_dataset_dir if database_type == 'coco' else miniimagenet_dataset_dir}")
    print(f"数据库路径是否存在: {os.path.exists(coco_dataset_dir if database_type == 'coco' else miniimagenet_dataset_dir)}")
    
    # 如果是miniImageNet，检查目录结构
    if database_type == "miniimagenet":
        train_dir = os.path.join(miniimagenet_dataset_dir, "train")
        print(f"miniImageNet训练目录: {train_dir}")
        print(f"训练目录是否存在: {os.path.exists(train_dir)}")
        if os.path.exists(train_dir):
            # 列出部分类别目录
            class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"训练目录中的类别数量: {len(class_dirs)}")
            if class_dirs:
                print(f"部分类别目录: {', '.join(class_dirs[:5])}")
                # 检查第一个类别目录中的图像
                first_class_dir = os.path.join(train_dir, class_dirs[0])
                images = [f for f in os.listdir(first_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                print(f"类别 {class_dirs[0]} 中的图像数量: {len(images)}")
                if images:
                    print(f"示例图像路径: {os.path.join(first_class_dir, images[0])}")
    
    print(f"使用参数: COCO图像权重={coco_image_scale}, 目标图像权重={target_image_scale}, COCO文本权重={coco_text_scale}, 目标文本权重={target_text_scale}")
    print(f"使用检索结果模式，将从JSON中获取rank 1-5的相似图像")
    
    # 加载模型
    pipe_prior_redux, pipe = load_model()
    
    # 更新全局检索结果目录
    global retrieval_results_dir, all_datasets_retrieval_file
    retrieval_results_dir = retrieval_results_dir_arg
    all_datasets_retrieval_file = os.path.join(retrieval_results_dir, "all_shots_retrieval_results.json")
    
    # 加载检索结果
    all_datasets_results = load_all_datasets_retrieval_results()
    if not all_datasets_results:
        print("错误：找不到检索结果文件，无法使用检索结果模式")
        return
    
    # 如果使用miniImageNet，从检索结果中提取类别信息
    if DATABASE_TYPE == "miniimagenet":
        print("\n从检索结果中提取miniImageNet类别信息...")
        miniimagenet_classes = extract_miniimagenet_classes_from_retrieval(all_datasets_results)
        print(f"提取到的类别: {', '.join(list(miniimagenet_classes)[:10])}...")
    
    # 测试几种不同的miniImageNet路径格式
    if DATABASE_TYPE == "miniimagenet":
        test_paths = [
            "miniimagenet/train/n01770081/n0177008100000076.jpg",
            "./miniimagenet/train/n01770081/n0177008100000076.jpg",
            "retrieval/miniimagenet/train/n01770081/n0177008100000076.jpg",
            "./retrieval/miniimagenet/train/n01770081/n0177008100000076.jpg"
        ]
        
        print("\n测试不同的miniImageNet路径格式:")
        for test_path in test_paths:
            print(f"测试路径: {test_path}")
            corrected_path = fix_miniimagenet_path(test_path)
            print(f"修正后的路径: {corrected_path}")
            print(f"路径是否存在: {os.path.exists(corrected_path) if corrected_path else False}")
    
    # 打印检索结果中的图像路径示例，帮助调试
    print("\n检索结果路径示例:")
    path_examples_found = 0
    
    # 尝试找到一些图像路径示例
    for dataset_key, dataset_value in all_datasets_results.items():
        if isinstance(dataset_value, dict):
            for shot_key, shot_value in dataset_value.items():
                if isinstance(shot_value, dict):
                    for sample_key, sample_value in shot_value.items():
                        if isinstance(sample_value, dict) and "similar_images" in sample_value:
                            similar_images = sample_value["similar_images"]
                            if isinstance(similar_images, list) and len(similar_images) > 0:
                                for img in similar_images:
                                    if isinstance(img, dict) and "image_path" in img:
                                        print(f"图像路径示例: {img['image_path']}")
                                        # 测试路径处理
                                        corrected_path = get_correct_image_path(img['image_path'])
                                        print(f"处理后的路径: {corrected_path}")
                                        print(f"路径是否存在: {os.path.exists(corrected_path) if corrected_path else False}")
                                        path_examples_found += 1
                                        if path_examples_found >= 3:
                                            break
                        if path_examples_found >= 3:
                            break
                if path_examples_found >= 3:
                    break
        if path_examples_found >= 3:
            break
    
    # 处理所有数据集
    for current_dataset in datasets_to_process:
        print(f"\n开始处理数据集: {current_dataset}")
        
        # 处理所有指定的shot数量
        for shot_number in shot_numbers:
            # 更新kshot_dir为当前处理的shot数量
            current_kshot_dir = os.path.join(lamainpaint_dir, current_dataset, f"{shot_number}_shot")
            
            if not os.path.exists(current_kshot_dir):
                print(f"警告：找不到{shot_number}-shot目录 {current_kshot_dir}，跳过处理")
                continue
                
            print(f"\n开始处理 {current_dataset} {shot_number}-shot...")
            
            # 检查当前shot数量的样本是否存在于检索结果中
            sample_files = [f for f in os.listdir(current_kshot_dir) 
                           if f.endswith('.jpg') and 
                           os.path.isfile(os.path.join(current_kshot_dir, f))]
            
            # 提取文件名（不含扩展名）作为样本名
            sample_names = [os.path.splitext(f)[0] for f in sample_files]
            
            if not sample_names:
                print(f"跳过 {current_dataset} {shot_number}-shot，因为找不到样本文件")
                continue
            
            # 检查前几个样本是否在检索结果中
            if sample_names:
                print(f"检查 {current_dataset} {shot_number}-shot 样本是否在检索结果中:")
                check_samples = sample_names[:min(5, len(sample_names))]  # 最多检查5个样本
                
                for sample_name in check_samples:
                    if current_dataset in all_datasets_results and f"{shot_number}_shot" in all_datasets_results[current_dataset]:
                        if sample_name in all_datasets_results[current_dataset][f"{shot_number}_shot"]:
                            print(f"  - 样本 {sample_name} 存在于检索结果中")
                        else:
                            print(f"  - 样本 {sample_name} 不存在于检索结果中")
                    else:
                        print(f"  - 无法检查样本 {sample_name}，因为检索结果中没有 {current_dataset}/{shot_number}_shot 路径")
            
            # 使用检索结果处理数据集
            process_kshot_dataset_with_retrieval(current_dataset, pipe_prior_redux, pipe, all_datasets_results, shot_number, output_dir)
    
    print("\n所有数据集处理完成")

if __name__ == "__main__":
    main() 