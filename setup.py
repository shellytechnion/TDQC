from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name='failure_prob',
    version='0.1',
    description='TDQC: Temporal-Difference Quality Calibration for VLA Failure Detection',
    url='https://github.com/shellytechnion/TDQC',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "einops",
        "flask",
        "hydra-core",
        "imageio[ffmpeg]",
        "matplotlib",
        "natsort",
        "numpy",
        "omegaconf",
        "opencv-python",
        "pandas",
        "plotly",
        "pyyaml",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "tqdm",
        "wandb",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)