from setuptools import setup


with open("README.md") as file:
    long_desc = file.read()


REPO_NAME = "TinyVGG_Custom_Modelwith_DDP"
AUTHOR_NAME = "Rahul-Shedge"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = []

setup(
    name=SRC_REPO,
    version="0.0.1",
    description="Pytorch Distributed Training",
    author=REPO_NAME,
    # author_email=
    long_description=long_desc,
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    packages=[SRC_REPO],
    author_email="rahulshedge555@gmail.com",
)

























