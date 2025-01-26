from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """This function will require a list of requirement"""
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements ]
        
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
setup(
    name ='student_performance_prediction',
    version = '0.0.1',
    author='Ankita',
    author_email='ankpillay@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)



 