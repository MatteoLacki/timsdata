# This Python file uses the following encoding: utf-8
from setuptools import setup, find_packages

setup(  name='timsdata',
        packages=find_packages(),
        version='1.0.3',
        description='timsdata: Bruker TDF wrapped in a Python module.',
        long_description='This Python module wraps Bruker timsTOF TDF.',
        author='Matteo Lacki',
        author_email='matteo.lacki@gmail.com',
        url='https://github.com/MatteoLacki/timsdata.git',
        keywords=['timsTOFpro', 'Bruker TDF', 'data science', 'mass spectrometry'],
        classifiers=['Development Status :: 1 - Planning',
                     'License :: Free To Use But Restricted',
                     'Intended Audience :: Science/Research',
                     'Topic :: Scientific/Engineering :: Chemistry',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8'],
        install_requires=['numpy'],
        package_dir={'timsdata':'timsdata'},
        package_data = {
                'timsdata':[
                    'cpp/win32/timsdata.dll',
                    'cpp/win32/timsdata.lib',
                    'cpp/win64/timsdata.dll',
                    'cpp/win64/timsdata.lib',
                    'cpp/libtimsdata.so'
                ]}
)
