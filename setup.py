import os
import re
import codecs
from setuptools import setup, find_packages


ROOT = os.path.abspath(os.path.dirname(__file__))
README_PATH = os.path.join(ROOT, "README.md")
REQUIREMENTS_PATH = os.path.join(ROOT, "requirements.txt")

with codecs.open(README_PATH, encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.4'
DESCRIPTION = 'Package for Bulgarian Natural Language Processing.'


def _get_requirements(path):
    with open(path, "r") as f:
        requirements_str = f.read()
        packages = re.findall(r"(.+==[^\n]+)\n", requirements_str)
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
            "Development Status :: 1 - Planning",
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
