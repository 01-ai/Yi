
# Tutorial: Building a PDF Upload and Mind Map Generation Application Using Yi's yi-large Model

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Installing Necessary Libraries](#installing-necessary-libraries)
4. [Creating the Project Structure](#creating-the-project-structure)
5. [Writing the Backend Code](#writing-the-backend-code)
6. [Creating the Frontend Page](#creating-the-frontend-page)
7. [Running and Testing the Application](#running-and-testing-the-application)
8. [Common Issues and Solutions](#common-issues-and-solutions)

## Introduction

This tutorial will guide you through building a web application that allows users to upload PDF files and generate corresponding mind maps using Yi's yi-large model. We'll use Python's Flask framework for the backend and HTML, CSS, and JavaScript for the frontend. The entire process will be broken down into small steps to ensure everyone can easily understand and follow along.

It's worth emphasizing that we'll be using Yi's yi-large model, a powerful large language model capable of understanding complex text and generating structured mind maps.
![img_3.png](assets/3/img(3-4).png)

## Environment Setup

This tutorial will use Mac OS as an example, but most steps are similar on Windows and Linux.

First, we need to make sure Python is installed on your Mac. If it's not already installed, follow these steps:

1. Open the Terminal application
2. Install Homebrew (if not already installed):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Use Homebrew to install Python:
   ```
   brew install python
   ```
4. After installation, verify by typing the following command in the terminal:
   ```
   python3 --version
   ```
   If it displays the Python version number, the installation was successful.

## Installing Necessary Libraries

We need to install some Python libraries to support our application. Enter the following command in the terminal:

```
pip3 install flask openai PyPDF2
```

This will install Flask (web framework), OpenAI (AI interface), and PyPDF2 (PDF processing) libraries.

## Creating the Project Structure

Let's create a new folder to store our project. Execute the following commands in the terminal:

```
mkdir yi_mindmap_app
cd yi_mindmap_app
mkdir templates
mkdir static
mkdir uploads
```

This will create the following structure:
- yi_mindmap_app/ (main project folder)
  - templates/ (for storing HTML templates)
  - static/ (for storing CSS and JavaScript files)
  - uploads/ (for storing uploaded PDF files)

## Writing the Backend Code

Now, we'll create the backend code step by step. First, create a file named `app.py` in the main project folder.

### Step 1: Import necessary libraries and set up the Flask application

Open `app.py` and add the following code:

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

API_BASE = "https://api.01.ai/v1"
API_KEY = "Your API Key"  # Please replace with your actual API key
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)
```

This code imports the necessary libraries, creates a Flask application instance, and sets up the upload folder and allowed file types. Note that we've set API_BASE and API_KEY, which are crucial configurations for connecting to Yi's yi-large model.

### Step 2: Define helper functions

Next, add the following helper functions:

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

These functions are used to check file types, extract text from PDFs, and parse the LLM's response into a mind map structure.

### Step 3: Define routes and main functionality

Now, let's add the main routes and functionality:

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

This part of the code defines the main routes and functionality, including chat, file upload, and mind map generation. Note that we use "yi-large" as the model name when interacting with Yi's yi-large model.

## Creating the Frontend Page

Now, let's create the frontend page. Create a file named `index.html` in the `templates` folder and gradually add the following content:

### Step 1: HTML structure and header

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
        
        <!-- The following content will be added here -->

    </div>
</body>
</html>
```

### Step 2: Add file upload section

Add the following content inside `<div class="container mx-auto px-4 py-8">`:

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

### Step 3: Add chat interface

Add the following content below the file upload section:

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

### Step 4: Add mind map display area

Add the following content below the chat interface:

```html
<div id="mind-map" class="bg-white rounded-lg shadow-md p-4"></div>
```

### Step 5: Add JavaScript code

Add the following JavaScript code before the `</body>` tag:

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
            .size([2 * Math.PI, Math.min(width, height) / 2

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
This code completes the front-end JavaScript functionality, including handling file uploads, communicating with the back-end, and creating mind maps using D3.js.

## Running and Testing the Application

Now that we've finished coding the back-end and front-end, let's run and test the application.

1. Make sure you're in the project's root directory.

2. Start the Flask application by running the following command in your terminal:

```
  python app.py
  ```
![img.png](assets/3/img(3-1).png)
3. Open your web browser and go to `http://127.0.0.1:5000`.
![img_1.png](assets/3/img(3-2).png)
4. You should see a web page with a file upload area, a chat interface, and a mind map display area.

5. Try the following actions:
   - Upload a PDF file and see if it generates a mind map.
   - Enter a topic in the chat box, such as "python learning path", and see if the Yi-large model can generate a corresponding mind map. 
6. Here we let Yi-large generate the learning path of Python. You can refer to the progress bar to see the generation speed of the LLM.
![img_2.png](assets/3/img(3-3).png)
7. Then you can switch between node display and tree diagram display. That's it!
![img_3.png](assets/3/img(3-4).png)
## Common Problems and Solutions

1. If you encounter any API key-related errors, make sure you have correctly configured the API key for Yi-large.

2. If you have problems uploading large files, you may need to adjust the maximum file upload size settings for Flask.

3. If the mind map is not displayed correctly, you may need to adjust the layout parameters of D3.js. 
