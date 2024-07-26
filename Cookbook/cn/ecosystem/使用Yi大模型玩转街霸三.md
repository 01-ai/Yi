# 使用Yi大模型玩转街霸三：从安装到应用的详细指南

## 引言

欢迎来到这篇详细的教学指南！在本文中，我们将探索如何使用Yi大模型（特别是yi-large模型）来玩转经典格斗游戏街霸三。这个有趣的项目将展示大语言模型（LLM）在游戏AI领域的潜力。无论你是AI爱好者还是游戏迷，这个教程都会带给你全新的体验。

本教程适合基础较薄弱的读者，我们会逐步详细讲解每个环节，确保你能顺利完成整个过程。

## 项目概述

在这个项目中，我们将：

1. 设置必要的环境和工具
2. 获取Yi模型的API密钥
3. 配置和运行游戏环境
4. 使用Yi模型控制街霸三的角色进行对战

![img.png](assets/2/img(4-2).png)
## 实验环境

- 操作系统：Windows 或 Mac OS（本教程将以Mac OS为例）
- Python 3.10
- Docker Desktop

## 步骤1：环境准备

### 1.1 安装Docker Desktop

1. 访问 [Docker Desktop官网](https://www.docker.com/products/docker-desktop/)
2. 下载适合你系统的版本
3. 按照安装向导完成安装
4. 安装完成后，重启你的电脑

### 1.2 安装Conda

Conda是一个强大的包管理工具，我们将用它来创建虚拟环境。

1. 访问 [Conda官网](https://conda.io/projects/conda/en/latest/index.html)
2. 下载并安装Miniconda（推荐）或Anaconda
3. 安装完成后，打开终端，输入以下命令验证安装：

   ```
   conda --version
   ```

   如果显示版本号，说明安装成功。
![img_1.png](assets/2/img(4-1).png)
### 1.3 注册Diambra账号

Diambra提供了我们需要的游戏环境。

1. 访问 [Diambra注册页面](https://diambra.ai/register)
2. 填写必要信息并完成注册

## 步骤2：获取Yi模型API密钥

1. 访问 [Yi大模型开放平台](https://platform.lingyiwanwu.com/)
2. 注册并登录账号
3. 在平台上创建新的API密钥
4. 保存好你的API密钥，我们后面会用到
![img_2.png](assets/2/img(4-3).png)
## 步骤3：配置项目环境

### 3.1 克隆项目仓库

打开终端，执行以下命令：

```bash
git clone https://github.com/Yimi81/llm-colosseum.git
cd llm-colosseum
```

### 3.2 创建并激活虚拟环境

在项目目录下，执行：

```bash
conda create -n yi python=3.10 -y
conda activate yi
```

### 3.3 安装依赖

```bash
pip install -r requirements.txt
```

### 3.4 配置环境变量

1. 复制示例环境文件：

   ```bash
   cp .env.example .env
   ```

2. 编辑.env文件：

   在Mac上，你可能需要显示隐藏文件。使用快捷键 `Command + Shift + .` 切换显示/隐藏隐藏文件。

3. 打开.env文件，将`YI_API_KEY`替换为你之前获取的API密钥。
![img_3.png](assets/2/img(4-4).png)
## 步骤4：启动游戏

### 4.1 找到ROM文件路径

在Mac环境中，ROM文件通常位于：

```
/Users/你的用户名/Desktop/code/llm-colosseum/.diambra/rom
```

记住这个路径，我们称之为`<your_roms_absolute_path>`。

### 4.2 启动游戏

在终端中执行：

```bash
diambra -r <your_roms_absolute_path> run -l python script.py
```
![img_4.png](assets/2/img(4-5).png)

首次运行时，系统会要求你输入Diambra账号的用户名和密码。之后，游戏镜像会开始下载。

然后等待启动就可以了

![img_5.png](assets/2/img(4-6).png)

## 结语

恭喜你！现在你已经成功设置并运行了一个由Yi大模型控制的街霸三AI对战系统。这个项目展示了大语言模型在游戏AI领域的潜力。你可以尝试修改代码，使用不同的Yi模型（如yi-medium），或者调整提示来改变AI的行为。

记住，使用API时可能会遇到请求频率限制。如果遇到这种情况，可以考虑升级你的API计划。
![img_6.png](assets/2/img(4-7).png)
希望你能从这个项目中学到新知识，并对AI在游戏中的应用产生更多兴趣。继续探索，享受AI带来的无限可能吧！

