# This Python file uses the following encoding: utf-8
from setuptools import setup, find_packages

setup(  name='timsdata',
        packages=find_packages(),
        version='0.0.1',
        description='Description.',
        long_description='Long description.',
        author='Matteo Lacki',
        author_email='matteo.lacki@gmail.com',
        # url='https://github.com/MatteoLacki/MIDIA.git',
        keywords=['Great module', 'Devel Inside'],
        classifiers=['Development Status :: 1 - Planning',
                     'License :: OSI Approved :: BSD License',
                     'Intended Audience :: Science/Research',
                     'Topic :: Scientific/Engineering :: Chemistry',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7'],
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
