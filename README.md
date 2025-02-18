# AI UI

A user-friendly interface for interacting with AI. Includes a voiced chat bot feature with an animated and customizable avatar. Built with Electron.

![AI UI Screenshot](./img/screenshot.jpg)

## Install Guide

Download the [latest release](https://github.com/JacobBruce/AI-UI/releases) of AI UI then follow the instructions for your platform:

<details><summary>Show install instructions for Windows</summary><br>

First you will need to install Python:

- [Python](https://www.python.org/downloads/windows/) (3.12.x recommended)

CUDA is optional but highly recommended if you have an Nvidia GPU:
- [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) (12.4 recommended)

Now open a command prompt as an administrator and create a virtual Python environment using this command:
```
python -m venv C:/venv
```
This will create a folder called venv on the C drive containing the Python environment.
Replace `C:/venv` with something else to change the location of the virtual environment.

Now you can activate the virtual environment by running the activate.bat file like this:
```
C:/venv/Scripts/activate.bat
```
Now install the required Python packages into the virtual environment using this command:
```
pip install -r C:/AI_UI/engine/requirements.txt
```
Replace `C:/AI_UI/` with the location where AI UI was extracted (should contain AI_UI.exe)

If your GPU supports CUDA then run this command before installing the requirements in the previous step:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
</details>

<details><summary>Show install instructions for Linux</summary><br>

First ensure you have Python installed (3.12.x recommended). You may also need to install pip and the tool for creating virtual Python environments using these commands:

```
sudo apt install python3-pip
```
```
sudo apt install python3.12-venv
```

Now open a terminal and navigate to the directory where you want to create a virtual Python environment then run this command:
```
python3 -m venv ./venv
```
This will create a folder called venv containing the Python environment.

Now you can activate the virtual environment by running this command:
```
source ./venv/bin/activate
```
Now install the required Python packages into the virtual environment using this command:
```
pip3 install -r ./AI_UI/engine/requirements.txt
```
Replace `./AI_UI/` with the location where AI UI was extracted (should contain electron)

If your GPU supports CUDA then run this command before installing the requirements in the previous step:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Now exit the virtual environment and run this command to install the nodejs package manager:
```
sudo apt install npm
```

Now navigate to the AI UI app folder then install the required nodejs packages using these commands:
```
cd AI_UI/resources/app
```
```
npm install
```

Now navigate back to the AI_UI folder (`cd ../..`) and launch AI UI using this command:
```
./electron --no-sandbox
```
</details>

## Important Info

### Windows Users

If you place AI UI into your Program Files directory or any other protected directory then you will need to run the app in administrator mode. It is recommended to place AI UI in another location (such as `C:/AI_UI/`) to avoid the use of administrator mode.

### Linux Users

Before starting AI UI from a terminal, make sure you navigate to the location where the electron binary is located instead of trying to launch it from another directory. If you don't do this AI UI will use the wrong working directory and it will fail to read/write files.

## Getting Started

After all the requirements have been installed you can launch the AI UI app. To get started you will need to visit the Settings tab so the engine knows where to find things like the Python environment and your model files.

In this example the 'Python Binary' setting would be `C:/venv/Scripts/python.exe` for Windows or `path/to/venv/lib/python3` for Linux. Check the Console tab for errors if the engine wont start.

To make the chat bot work you'll need to download a text generation AI model from [Hugging Face](https://huggingface.co/) (must use the HF Transformers format). Models which have been fine-tuned on conversational text should work best for the chat bot.

If you want to generate images or want the chat bot to send messages with images you will also need to download an image generation AI model which uses the HF Diffusers format, many can be found on Hugging Face.

## Model Files

This repository doesn't contain the large model files from MakeItTalk, Wav2Lip, or SadTalker. They will be automatically downloaded by AI UI when they are needed.