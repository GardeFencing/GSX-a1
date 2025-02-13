from setuptools import setup, find_packages

setup(
    name="mmpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "opencv-python>=4.4.0",
        "pillow>=8.0.0",
    ],
    author="VidCLIPChatbot Team",
    description="Multimodal Processing Toolkit for Video Analysis",
    python_requires=">=3.7",
)
