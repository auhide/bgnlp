import os
import re
from pathlib import Path
from setuptools import setup, find_packages


ROOT = os.path.abspath(os.path.dirname(__file__))
README_PATH = os.path.join(ROOT, "README.md")
REQUIREMENTS_PATH = os.path.join(ROOT, "requirements.txt")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

VERSION = '0.2.0'
DESCRIPTION = 'Package for Bulgarian Natural Language Processing (NLP)'


def _get_requirements(path):
    with open(path, "r") as f:
        requirements_str = f.read()
        packages = re.findall(r"(.+=?=?[^\n]+)\n", requirements_str)
        return packages


if __name__ == "__main__":
    setup(
        name="bgnlp",
        version=VERSION,
        author="Adam Fauzi",
        author_email="adamfzh98@gmail.com",
        description=DESCRIPTION,
        long_description_content_type="text/markdown",
        long_description=long_description,
        packages=find_packages(),
        install_requires=_get_requirements(REQUIREMENTS_PATH),
        keywords=['pytorch', 'nlp', 'bulgaria', 'machine learning', "deep learning", "AI"],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",

            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',

            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: Unix",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        # This is used because we need the resource files of the package (mainly models).
        include_package_data=True
    )
