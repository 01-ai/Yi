
# Mastering Street Fighter III with Yi Large Language Model

## Introduction

Welcome to this detailed tutorial! In this article, we'll explore how to use the Yi large language model (specifically the yi-large model) to master the classic fighting game Street Fighter III. This exciting project will showcase the potential of large language models (LLMs) in the realm of game AI. Whether you're an AI enthusiast or a gaming fan, this tutorial will offer you a fresh and exciting experience.

This guide is designed for readers with a basic understanding, and we'll walk through each step in detail to ensure you can successfully complete the entire process.

## Project Overview

In this project, we will:

1. Set up the necessary environment and tools
2. Obtain an API key for the Yi model
3. Configure and run the game environment
4. Use the Yi model to control Street Fighter III characters in battles

![img.png](assets/2/img(4-2).png)

## Experimental Environment

- Operating System: Windows or Mac OS (this tutorial will use Mac OS as an example)
- Python 3.10
- Docker Desktop

## Step 1: Environment Preparation

### 1.1 Installing Docker Desktop

1. Visit the [Docker Desktop official website](https://www.docker.com/products/docker-desktop/)
2. Download the version suitable for your system
3. Follow the installation wizard to complete the setup
4. After installation, restart your computer

### 1.2 Installing Conda

Conda is a powerful package management tool that we'll use to create virtual environments.

1. Visit the [Conda official website](https://conda.io/projects/conda/en/latest/index.html)
2. Download and install Miniconda (recommended) or Anaconda
3. After installation, open a terminal and enter the following command to verify the installation:

   ```
   conda --version
   ```

   If it displays a version number, the installation was successful.

![img_1.png](assets/2/img(4-1).png)

### 1.3 Registering a Diambra Account

Diambra provides the game environment we need.

1. Visit the [Diambra registration page](https://diambra.ai/register)
2. Fill in the required information and complete the registration

## Step 2: Obtaining Yi Model API Key

1. Visit the [Yi Large Language Model Open Platform](https://platform.01.ai/apikeys)
2. Register and log in to your account
3. Create a new API key on the platform
4. Save your API key securely, as we'll need it later

![img.png](assets/2/img(4-3).png)

## Step 3: Configuring the Project Environment

### 3.1 Cloning the Project Repository

Open a terminal and execute the following commands:

```bash
git clone https://github.com/Yimi81/llm-colosseum.git
cd llm-colosseum
```

### 3.2 Creating and Activating a Virtual Environment

In the project directory, execute:

```bash
conda create -n yi python=3.10 -y
conda activate yi
```

### 3.3 Installing Dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Configuring Environment Variables

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit the .env file:

   On Mac, you might need to show hidden files. Use the shortcut `Command + Shift + .` to toggle showing/hiding hidden files.

3. Open the .env file and replace `YI_API_KEY` with the API key you obtained earlier.

## Step 4: Launching the Game

### 4.1 Locating the ROM File Path

On a Mac environment, ROM files are typically located at:```
/Users/your_username/Desktop/code/llm-colosseum/.diambra/rom


Remember this path; we'll refer to it as `<your_roms_absolute_path>`.

### 4.2 Starting the Game

In the terminal, execute:```bash
diambra -r <your_roms_absolute_path> run -l python script.py
![img_4.png](assets/2/img(4-5).png)

When running for the first time, the system will prompt you to enter your Diambra account username and password. After that, the game image will start downloading.

Then, just wait for it to launch.![img_5.png](assets/2/img(4-6).png)

## Conclusion

Congratulations! You've now successfully set up and run a Street Fighter III AI battle system controlled by the Yi large language model. This project demonstrates the potential of large language models in the field of game AI. You can try modifying the code, using different Yi models (such as yi-medium), or adjusting the prompts to change the AI's behavior.

Remember, when using the API, you might encounter request frequency limitations. If this happens, consider upgrading your API plan.![img_6.png](assets/2/img(4-7).png)

I hope you've learned something new from this project and developed a greater interest in AI applications in gaming. Keep exploring and enjoy the endless possibilities that AI brings!```
