# 使用Yi的yi-large模型构建PDF上传和思维导图生成应用教程

## 目录
1. [简介](#简介)
2. [环境准备](#环境准备)
3. [安装必要的库](#安装必要的库)
4. [创建项目结构](#创建项目结构)
5. [编写后端代码](#编写后端代码)
6. [创建前端页面](#创建前端页面)
7. [运行和测试应用](#运行和测试应用)
8. [常见问题和解决方案](#常见问题和解决方案)


## 简介

本教程将指导您如何构建一个Web应用，该应用允许用户上传PDF文件，然后使用Yi的yi-large模型生成相应的思维导图。我们将使用Python的Flask框架作为后端，HTML、CSS和JavaScript作为前端。整个过程将被分解为小步骤，以确保每个人都能轻松理解和跟随。

特别强调的是，我们将使用Yi的yi-large模型，这是一个强大的大语言模型，能够理解复杂的文本并生成结构化的思维导图。
![img_3.png](assets/3/img(3-4).png)
## 环境准备

本教程将以Mac OS为例进行说明，但大部分步骤在Windows和Linux上也是类似的。
    
首先，我们需要确保您的Mac上安装了Python。如果还没有安装，请按照以下步骤操作：

1. 打开终端（Terminal）应用
2. 安装Homebrew（如果尚未安装）：
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. 使用Homebrew安装Python：
   ```
   brew install python
   ```
4. 安装完成后，在终端中输入以下命令来验证安装：
   ```
   python3 --version
   ```
   如果显示Python版本号，说明安装成功。

## 安装必要的库

我们需要安装一些Python库来支持我们的应用。在终端中输入以下命令：

```
pip3 install flask openai PyPDF2
```

这将安装Flask（Web框架）、OpenAI（AI接口）和PyPDF2（PDF处理）库。

## 创建项目结构

让我们创建一个新的文件夹来存放我们的项目。在终端中执行以下命令：

```
mkdir yi_mindmap_app
cd yi_mindmap_app
mkdir templates
mkdir static
mkdir uploads
```

这将创建以下结构：
- yi_mindmap_app/（主项目文件夹）
  - templates/（存放HTML模板）
  - static/（存放CSS和JavaScript文件）
  - uploads/（存放上传的PDF文件）

## 编写后端代码

现在，我们将逐步创建后端代码。首先，在主项目文件夹中创建一个名为`app.py`的文件。

### 步骤1：导入必要的库和设置Flask应用

打开`app.py`，添加以下代码：

```python
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import json
from werkzeug.utils import secure_filename
from openai import OpenAI
import openai
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'doc', 'docx'}

API_BASE = "https://api.lingyiwanwu.com/v1"
API_KEY = "您的API密钥"  # 请替换为您的实际API密钥
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)
```

这段代码导入了必要的库，创建了Flask应用实例，并设置了上传文件夹和允许的文件类型。特别注意，我们设置了API_BASE和API_KEY，这是用于连接Yi的yi-large模型的关键配置。

### 步骤2：定义辅助函数

接下来，添加以下辅助函数：

```python
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def parse_llm_response_to_mind_map(response):
    lines = response.split('\n')
    root = {"name": lines[0].strip(), "children": []}
    current_level = [root]

    for line in lines[1:]:
        if line.strip():
            depth = len(line) - len(line.lstrip())
            node = {"name": line.strip(), "children": []}

            while depth < len(current_level) - 1:
                current_level.pop()

            current_level[-1]["children"].append(node)
            current_level.append(node)

    return root
```

这些函数用于检查文件类型、从PDF中提取文本、以及将LLM的响应解析为思维导图结构。

### 步骤3：定义路由和主要功能

现在，让我们添加主要的路由和功能：

```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']

    try:
        completion = client.chat.completions.create(
            model="yi-large",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates mind maps. Please provide a detailed mind map structure for the given topic."},
                {"role": "user", "content": f"Create a mind map for: {user_message}"}
            ]
        )

        llm_response = completion.choices[0].message.content

        mind_map = parse_llm_response_to_mind_map(llm_response)

        response = {
            "message": f"Here's a detailed mind map for '{user_message}'",
            "mind_map": mind_map
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    user_message = request.json['message']

    def generate():
        try:
            stream = client.chat.completions.create(
                model="yi-large",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates mind maps. Please provide a detailed mind map structure for the given topic."},
                    {"role": "user", "content": f"Create a mind map for: {user_message}"}
                ],
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            mind_map = parse_llm_response_to_mind_map(full_response)
            yield f"data: {json.dumps({'mind_map': mind_map})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), content_type='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            mind_map = generate_mind_map_from_text(text)
            return jsonify({"message": f"File {filename} processed successfully", "mind_map": mind_map}), 200
        else:
            return jsonify({"error": "Only PDF files are supported for mind map generation"}), 400
    return jsonify({"error": "File type not allowed"}), 400

def generate_mind_map_from_text(text):
    try:
        completion = client.chat.completions.create(
            model="yi-large",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates mind maps. Please provide a detailed mind map structure for the given text."},
                {"role": "user", "content": f"Create a mind map for the following text: {text[:4000]}"}
            ]
        )

        llm_response = completion.choices[0].message.content
        return parse_llm_response_to_mind_map(llm_response)
    except Exception as e:
        print(f"Error generating mind map: {str(e)}")
        return {"name": "Error", "children": [{"name": str(e)}]}

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
```

这部分代码定义了主要的路由和功能，包括聊天、文件上传和思维导图生成。特别注意，我们在与Yi的yi-large模型交互时使用了"yi-large"作为模型名称。

## 创建前端页面

现在，让我们创建前端页面。在`templates`文件夹中创建一个名为`index.html`的文件，并逐步添加以下内容：

### 步骤1：HTML结构和头部

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yi-large PDF to Mind Map</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8 text-center">Yi-large PDF to Mind Map Converter</h1>
        
        <!-- 接下来的内容将在这里添加 -->

    </div>
</body>
</html>
```

### 步骤2：添加文件上传部分

在`<div class="container mx-auto px-4 py-8">`内添加以下内容：

```html
<div id="file-upload" class="bg-white rounded-lg shadow-md p-4 mb-8">
    <h2 class="text-xl font-semibold mb-2">Upload PDF File</h2>
    <form id="upload-form" enctype="multipart/form-data">
        <input type="file" id="file-input" name="file" accept=".pdf" class="mb-2">
        <button type="submit" class="bg-green-500 text-white px-4 py-1 rounded">Upload and Generate Mind Map</button>
    </form>
    <div id="upload-status" class="mt-2"></div>
</div>
```

### 步骤3：添加聊天界面

在文件上传部分下方添加以下内容：

```html
<div id="chat-container" class="bg-white rounded-lg shadow-md p-4 mb-8">
    <h2 class="text-xl font-semibold mb-2">Chat with Yi-large</h2>
    <div id="chat-messages" class="mb-4"></div>
    <form id="chat-form">
        <input type="text" id="user-input" class="w-full p-2 border rounded" placeholder="Type your message...">
        <button type="submit" class="mt-2 bg-blue-500 text-white px-4 py-1 rounded">Send</button>
    </form>
</div>
```

### 步骤4：添加思维导图显示区域

在聊天界面下方添加以下内容：

```html
<div id="mind-map" class="bg-white rounded-lg shadow-md p-4"></div>
```

### 步骤5：添加JavaScript代码

在`</body>`标签前添加以下JavaScript代码：

```html
<script>
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');

    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value;
        if (message.trim()) {
            addMessage('User', message);
            fetchResponse(message);
            userInput.value = '';
        }
    });

    function addMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatMessages.appendChild(messageElement);
    }

    function fetchResponse(message) {
        fetch('/chat_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function readStream() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        console.log('Stream complete');
                        return;
                    }
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop();

                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            if (data.content) {
                                addMessage('Yi-large', data.content);
                            } else if (data.mind_map) {
                                createMindMap(data.mind_map);
                            }
                        }
                    });
                    readStream();
                });
            }

            readStream();
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('System', 'An error occurred while fetching the response.');
        });
    }

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        uploadStatus.textContent = 'Uploading...';
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadStatus.textContent = 'Error: ' + data.error;
            } else {
                uploadStatus.textContent = data.message;
                if (data.mind_map) {
                    createMindMap(data.mind_map);
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadStatus.textContent = 'An error occurred during upload.';
        });
    });

    function createMindMap(data) {
        const width = 800;
        const height = 600;
        
        d3.select("#mind-map").selectAll("*").remove();
        
        const svg = d3.select("#mind-map")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        const g = svg.append("g")
            .attr("transform", `translate(${width / 2},${height / 2})`);
        
        const tree = d3.tree()
            .size([2 * Math.PI, Math.min(width, height) / 2 - 100]);
        
        const root = d3.hierarchy(data);
        tree(root);
        
        const link = g.selectAll(".link")
            .data(root.links())
            .enter().append("path")
            .attr("class", "link")
            .attr("d", d3.linkRadial()
                .angle(d => d.x)
                .radius(d => d.y));
        
        const node = g.selectAll(".node")
            .data(root.descendants())
            .enter().append("g")
            .attr("class", "node")
            .attr("transform", d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`);
        
        node.append("circle")
            .attr("r", 4);
        
        node.append("text")
            .attr("dy", "0.31em")
            .attr("x", d => d.x < Math.PI === !d.children ? 6 : -6)
            .attr("text-anchor", d => d.x < Math.PI === !d.children ? "start" : "end")
            .attr("transform", d => d.x >= Math.PI ? "rotate(180)" : null)
            .text(d => d.data.name)
            .clone(true).lower()
            .attr("stroke", "white");
    }
</script>
```

这段代码完成了前端的JavaScript功能,包括处理文件上传、与后端通信以及使用D3.js创建思维导图。

## 运行和测试应用

现在我们已经完成了后端和前端的编码,让我们来运行和测试应用程序。

1. 确保您在项目根目录下。

2. 在终端中运行以下命令启动Flask应用:

   ```
   python app.py
   ```
![img.png](assets/3/img(3-1).png)
3. 打开web浏览器,访问 `http://127.0.0.1:5000`。
![img_1.png](assets/3/img(3-2).png)
4. 您应该能看到一个包含文件上传区域、聊天界面和思维导图显示区域的网页。

5. 尝试以下操作:
   - 上传一个PDF文件,看看是否能生成思维导图。
   - 在聊天框中输入一个主题,例如"python的学习路线",看看Yi-large模型是否能生成相应的思维导图。
6. 这里我们让yi-large生成python的学习路线，可以参考进度条看到LLM的生成速度
![img_2.png](assets/3/img(3-3).png)
7. 然后可以切换是节点展示还是树状图展示，就成功啦
![img_3.png](assets/3/img(3-4).png)
## 常见问题和解决方案

1. 如果遇到API密钥相关的错误,请确保您已经正确设置了Yi-large的API密钥。

2. 如果上传大文件时遇到问题,可能需要调整Flask的最大上传文件大小设置。

3. 如果思维导图显示不正确,可能需要调整D3.js的布局参数。
