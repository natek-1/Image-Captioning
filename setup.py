from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirments(file_path:str)->List[str]:
    '''
    Using a given file path it returns all the packages that are in the given file

    file_path: path to the file with the requirments 

    returns: list object with the packages as strings
    '''
    requires = []
    with open(file_path, "r") as file:
        requires = file.readlines()
        requires = [req.replace("\n", "") for req in requires]

    if HYPHEN_E_DOT in requires:
        requires.remove(HYPHEN_E_DOT)

    return requires

setup(
    name="ImageCaption",
    version="0.0.1",
    author="Nathan Kuissi",
    author_email="nategabrielk@icloud.com",
    install_requires=get_requirments("./requirements.txt"),
    packages=find_packages()
)