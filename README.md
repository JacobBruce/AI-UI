# AI UI

A simple user interface for interacting with AI. Includes a voiced chat bot feature with an animated and customizable avatar. Built with Electron.

## Installing on Windows

Download the latest Windows release of AI UI then download and install the following:

- Python (3.8.10 recommended) (https://www.python.org/downloads/windows/)
- ffmpeg (https://ffmpeg.org/download.html)

Optional:
- CUDA 11.7 (for Nvidia GPU's) (https://developer.nvidia.com/cuda-11-7-0-download-archive)

Now open a command prompt as an administrator and create a Python virtual environment using this command:
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

If you want GPU support run these two commands while the virtual environment is still activated:
```
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Apparently the latest versions of Tensorflow for Windows don't have GPU support so you may also need to run these commands:
```
pip uninstall tensorflow
pip install "tensorflow<2.11"
```

Next you'll need to download a text generation AI model from a site like [Hugging Face](https://huggingface.co/models). Models which have been fine-tuned on conversational text should work best for the chat bot.

If you want the chat bot to be capable of sending images you will also need to download a separate image generation AI model which uses the Stable Diffusion pipeline, many can be found on Hugging Face.

Now you can run AI_UI.exe but it wont do much until you visit the Settings tab and fill out the required information. The 'Python Binary' setting would be `C:/venv/Scripts/python.exe` for this example setup.

## Model Files

This repository doesn't contain the large model files from MakeItTalk. However they should be included with AI UI releases if you need them.