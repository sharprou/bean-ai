# beanai

<div align="center">

![beanai](frontend/public/favicon.jpg)

**基于 AI 的智能拼豆图纸生成系统**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/next.js-13+-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](https://www.typescriptlang.org/)

</div>

## 📖 项目简介

beanai 是一个创新的 AI 驱动拼豆设计工具，让用户通过简单的文本描述或图片上传，即可自动生成专业的 Q 版拼豆设计图纸和材料清单。

### 核心价值

- **降低设计门槛**：无需绘画基础，人人都能创作拼豆作品
- **节省时间成本**：自动化网格绘制和材料统计，从数小时缩短到数分钟
- **保证可实现性**：基于真实拼豆颜色卡进行精准匹配
- **提升创作体验**：实时预览、一键导出，流畅的用户体验

## ✨ 核心功能

### 多模态输入支持
- 文本描述生成（如："一只可爱的柯基犬"）
- 图片上传转换
- 实体名称识别

### 智能设计生成
- AI 驱动的 Q 版风格转换
- 自动背景移除和主体提取
- 智能网格划分（32x32 至 128x128 自适应）
- 精准颜色匹配（基于真实拼豆色卡）

### 完整输出
- 高清拼豆设计图预览
- 详细材料清单（CSV 格式）
- 可下载的设计文件

## 🏗️ 技术架构

### 前端技术栈
- **框架**：Next.js 13+ / React 18+
- **语言**：TypeScript 5.0+
- **样式**：Tailwind CSS
- **通信**：WebSocket（实时交互）
- **状态管理**：React Context + Hooks

### 后端技术栈
- **语言**：Python 3.8+
- **AI 框架**：LangChain AI Agent
- **图像处理**：OpenCV、Pillow、rembg
- **AI 模型**：通义千问 API
- **架构模式**：模块化工具链设计

### 核心工具链

```
用户输入
    ↓
输入识别 (identify_input_type)
    ↓
┌─────────┬─────────┬─────────┐
│  文本   │  图片   │  实体   │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
描述增强   主体提取   知识图谱
    ↓         ↓         ↓
    └─────────┴─────────┘
            ↓
    图像生成 (generate_image)
            ↓
    拼豆设计生成 (generate_bean_buddy_design)
            ↓
    输出（图纸 + 材料清单）
```

## 🚀 快速开始

### 环境要求

- Node.js 16+
- Python 3.8+
- npm 或 yarn

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/sharprou/bean-ai.git
cd bean-ai
```

#### 2. 后端设置

```bash
cd backend
pip install -r requirements.txt

# 配置环境变量
# 在 backend/beanbuddy_ai/src/beanbuddy_ai/configs/config.yml 中配置 API 密钥
```

#### 3. 前端设置

```bash
cd frontend
npm install

# 配置环境变量
# 复制 .env.example 到 .env 并填写必要配置
```

#### 4. 启动服务

```bash
# 启动后端（在 backend 目录）
python -m beanbuddy_ai.src.beanbuddy_ai.register

# 启动前端（在 frontend 目录）
npm run dev
```

访问 `http://localhost:3000` 开始使用！

## 💡 使用示例

### 文本输入示例

```
输入："一只可爱的柯基犬"
    ↓
系统自动增强描述："白色背景，卡通风格，一只站立的柯基犬，耳朵竖起，尾巴翘起，线条简洁清晰"
    ↓
生成 AI 图像
    ↓
转换为拼豆设计（64x64 网格）
    ↓
输出设计图 + 材料清单
```

### 图片上传示例

```
上传图片
    ↓
自动移除背景
    ↓
提取主体并转换为 Q 版风格
    ↓
颜色匹配（映射到真实拼豆色卡）
    ↓
生成拼豆网格设计
    ↓
输出设计图 + 材料清单
```

## 🔧 核心技术亮点

### 1. AI Agent 架构
- 模块化工具设计，各组件解耦
- 智能路由，根据输入类型自动选择处理流程
- 易于扩展，新增功能只需注册新工具

### 2. 颜色匹配优化
- 欧几里得距离算法实现精准匹配
- 色卡缓存机制提升性能
- 边缘像素模糊处理减少噪点

### 3. 性能优化
- 流式处理大图像，减少内存占用
- 模型会话缓存，避免重复加载
- 异步 IO 处理，提升并发能力

### 4. 实时通信
- WebSocket 实现前后端实时交互
- 进度反馈，提升用户体验

## 📁 项目结构

```
beanai/
├── frontend/                 # 前端项目
│   ├── components/          # React 组件
│   ├── pages/              # Next.js 页面
│   ├── types/              # TypeScript 类型定义
│   └── utils/              # 工具函数
├── backend/                 # 后端项目
│   └── beanbuddy_ai/       # AI 核心模块
│       └── src/
│           └── beanbuddy_ai/
│               ├── configs/     # 配置文件
│               ├── models/      # AI 模型
│               └── tools/       # 工具链
├── csv/                     # 生成的设计文件
└── docs/                    # 项目文档
```

## 🛠️ 核心模块说明

### 输入识别 (identify_input_type)
- 自动识别文本、图片或实体名称
- URL 验证和文件类型检测
- LLM 辅助分类

### 描述增强 (enhance_description)
- 将简短描述扩展为详细的 AI 绘画提示词
- 添加风格、细节、背景等信息
- 确保输出符合拼豆工艺特性

### 主体提取 (extract_subject)
- 使用 rembg 库移除图片背景
- U2Net 模型实现精准主体提取
- Q 版风格转换优化

### 图像生成 (generate_image_from_text)
- 调用通义千问 API 生成图像
- 支持多种风格和尺寸
- 错误处理和重试机制

### 拼豆设计生成 (generate_bean_buddy_design)
- 网格划分和颜色匹配
- 材料清单统计
- CSV 文件导出


</div>