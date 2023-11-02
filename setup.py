from setuptools import setup, find_packages
import re

# auto-updating version code stolen from orbitize which stole from RadVel :)
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)

def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

setup(

    name="ACID_code",

    version=get_property("__version__", "ACID_code"),

    packages=find_packages(),

    description="Returns line profiles from input spectra by fitting the stellar continuum and performing LSD",

    install_requires=get_requires()
)