#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为beanai项目frontend/public目录下的所有图片添加斜向三条水印
水印文本：RoCrisp
"""

import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def add_diagonal_watermark_to_image(image_path, watermark_text="RoCrisp", output_dir=None):
    """
    为单张图片添加斜向三条水印
    
    Args:
        image_path: 图片文件路径
        watermark_text: 水印文本
        output_dir: 输出目录（None表示覆盖原文件）
    
    Returns:
        bool: 是否成功添加水印
    """
    try:
        # 支持的图片格式
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        # 检查文件格式
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in supported_formats:
            print(f"跳过不支持的文件格式: {image_path}")
            return False
        
        # 打开图片
        with Image.open(image_path) as img:
            # 转换为RGB模式（处理RGBA等模式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 创建水印图层
            watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            # 设置水印字体和大小
            try:
                # 根据图片大小自适应字体大小
                font_size = max(min(img.width, img.height) // 15, 20)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                    font_size = 20
            
            # 计算水印文本大小
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算对角线长度
            diagonal = math.sqrt(img.width**2 + img.height**2)
            
            # 三条水印的位置参数
            watermarks = [
                # 第一条：从左上到右下的对角线，居中
                {
                    'start': (-img.width * 0.1, -img.height * 0.1),
                    'end': (img.width * 1.1, img.height * 1.1),
                    'offset': 0,
                    'opacity': 80
                },
                # 第二条：稍微向上偏移
                {
                    'start': (-img.width * 0.1, -img.height * 0.3),
                    'end': (img.width * 1.1, img.height * 0.9),
                    'offset': -text_height * 1.5,
                    'opacity': 60
                },
                # 第三条：稍微向下偏移
                {
                    'start': (-img.width * 0.1, img.height * 0.1),
                    'end': (img.width * 1.1, img.height * 1.3),
                    'offset': text_height * 1.5,
                    'opacity': 60
                }
            ]
            
            # 添加三条斜向水印
            for i, wm_config in enumerate(watermarks):
                # 计算水印间距
                spacing = max(text_width * 1.5, 100)
                num_repeats = int(diagonal / spacing) + 2
                
                # 计算水印方向向量
                dx = wm_config['end'][0] - wm_config['start'][0]
                dy = wm_config['end'][1] - wm_config['start'][1]
                length = math.sqrt(dx**2 + dy**2)
                unit_dx = dx / length
                unit_dy = dy / length
                
                # 沿对角线重复添加水印
                for j in range(num_repeats):
                    # 计算水印中心位置
                    distance = j * spacing
                    center_x = wm_config['start'][0] + unit_dx * distance
                    center_y = wm_config['start'][1] + unit_dy * distance
                    
                    # 考虑偏移量
                    center_y += wm_config['offset']
                    
                    # 计算水印旋转角度（45度）
                    angle = 45
                    rad = math.radians(angle)
                    
                    # 计算水印位置（考虑旋转）
                    pos_x = center_x - text_width/2 * math.cos(rad) + text_height/2 * math.sin(rad)
                    pos_y = center_y - text_width/2 * math.sin(rad) - text_height/2 * math.cos(rad)
                    
                    # 检查水印是否在图片范围内
                    if (pos_x + text_width > 0 and pos_x < img.width and 
                        pos_y + text_height > 0 and pos_y < img.height):
                        
                        # 创建旋转的水印文本
                        text_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
                        text_draw = ImageDraw.Draw(text_img)
                        text_draw.text((0, 0), watermark_text, fill=(255, 255, 255, wm_config['opacity']), font=font)
                        
                        # 旋转文本
                        rotated_text = text_img.rotate(angle, expand=True, resample=Image.BICUBIC)
                        
                        # 计算旋转后的位置
                        rotated_width, rotated_height = rotated_text.size
                        final_x = int(pos_x - rotated_width/2 + text_width/2)
                        final_y = int(pos_y - rotated_height/2 + text_height/2)
                        
                        # 将旋转后的水印粘贴到水印图层
                        watermark.paste(rotated_text, (final_x, final_y), rotated_text)
            
            # 合并水印和原图
            watermarked_img = Image.alpha_composite(
                img.convert('RGBA'), watermark
            ).convert('RGB')
            
            # 确定输出路径
            if output_dir:
                # 保持目录结构
                relative_path = Path(image_path).relative_to(Path('D:/code/bean-ai/frontend/public'))
                output_path = Path(output_dir) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # 覆盖原文件
                output_path = image_path
            
            # 保存图片（保持原质量）
            if file_ext in {'.jpg', '.jpeg'}:
                watermarked_img.save(output_path, 'JPEG', quality=95)
            elif file_ext == '.png':
                watermarked_img.save(output_path, 'PNG', optimize=True)
            else:
                watermarked_img.save(output_path)
            
            print(f"✓ 已添加斜向水印: {image_path} -> {output_path}")
            return True
            
    except Exception as e:
        print(f"✗ 处理图片失败 {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_directory(directory_path, watermark_text="RoCrisp", output_dir=None, recursive=True):
    """
    处理目录中的所有图片文件
    
    Args:
        directory_path: 目录路径
        watermark_text: 水印文本
        output_dir: 输出目录
        recursive: 是否递归处理子目录
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"错误: 目录不存在 {directory_path}")
        return
    
    # 支持的图片格式
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    # 统计信息
    processed_count = 0
    success_count = 0
    
    print(f"开始处理目录: {directory_path}")
    print(f"水印文本: {watermark_text}")
    print(f"水印样式: 斜向三条对角线")
    print(f"输出目录: {output_dir if output_dir else '覆盖原文件'}")
    print("-" * 50)
    
    # 遍历文件
    if recursive:
        file_iterator = directory_path.rglob('*')
    else:
        file_iterator = directory_path.glob('*')
    
    for file_path in file_iterator:
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            processed_count += 1
            if add_diagonal_watermark_to_image(str(file_path), watermark_text, output_dir):
                success_count += 1
    
    print("-" * 50)
    print(f"处理完成!")
    print(f"总共处理: {processed_count} 个文件")
    print(f"成功添加水印: {success_count} 个文件")
    print(f"失败: {processed_count - success_count} 个文件")

def main():
    """主函数"""
    # 配置参数
    public_dir = r"D:\code\bean-ai\frontend\public"
    watermark_text = "RoCrisp"
    
    # 可选：指定输出目录（None表示覆盖原文件）
    # output_dir = r"D:\code\bean-ai\frontend\public_watermarked"
    output_dir = None  # 覆盖原文件
    
    # 检查PIL库是否安装
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("错误: 需要安装Pillow库")
        print("请运行: pip install Pillow")
        sys.exit(1)
    
    # 检查目录是否存在
    if not os.path.exists(public_dir):
        print(f"错误: 目录不存在 {public_dir}")
        sys.exit(1)
    
    # 确认操作
    if output_dir:
        print(f"将在新目录 {output_dir} 中创建带水印的图片")
    else:
        print("警告: 将直接覆盖原文件!")
        response = input("确定要继续吗? (y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            sys.exit(0)
    
    # 处理目录
    process_directory(public_dir, watermark_text, output_dir, recursive=True)

if __name__ == "__main__":
    main()