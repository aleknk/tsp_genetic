from setuptools import setup, find_packages

setup(
    name='pytsp',
    version='1.0',
    packages=find_packages(),
    entry_points={
            'console_scripts':[
                'tspga_run = pytsp.tools.run:main',
                'tspga_multirun = pytsp.tools.multirun:main'
            ]
        }
)
