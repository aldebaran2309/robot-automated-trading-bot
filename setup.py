from setuptools import setup
from setuptools import find_namespace_packages

# load the README file.
with open(file="README.md", mode="r") as fh:
    long_description = fh.read()

 
setup(

    name='python-trading-robot',

   
    version='0.1.1',

    description='A trading robot built for Python that uses the TD Ameritrade API.',

    long_description=long_description,

    long_description_content_type="text/markdown",

    

    install_requires=[
        'td-ameritrade-python-api>=0.3.0',
        'pandas==1.0.5',
        'numpy==1.19.0'
    ],

    keywords='finance, td ameritrade, api, trading robot',

    packages=find_namespace_packages(
        include=['pyrobot', 'samples', 'tests'],
        exclude=['configs*']
    ),

    include_package_data=True,

    python_requires='>=3.8',

    classifiers=[

        
        

        # Here I'll note that package was written in English.
        'Natural Language :: English',

        # Here I'll note that any operating system can use it.
        'Operating System :: OS Independent',

        # Here I'll specify the version of Python it uses.
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',

        # Here are the topics that my library covers.
        'Topic :: Database',
        'Topic :: Education',
        'Topic :: Office/Business'

    ]
)