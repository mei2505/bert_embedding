from setuptools import setup
import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent / 'tests'))
'''
python3 -m unittest
vim setup.py
rm -rf dist/
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
'''
setup(
    name='bertemb',
    version='0.0.1', #上げていく
    url='https://github.com/mei2505/bert_embedding.git',
    license='None',
    author='Miyu Tamura',
    description='Sharable in Google Colab',
    install_requires=['setuptools', 'transformers==3.5.1', 'torch','mecab-python3==0.7','fugashi[unidic-lite]','ipadic'],
    packages=['bertemb'],
    package_data={'bertemb': [
        '*/*', '*/*/*', '*/*.tpeg', '*/*.csv', '*/*.txt']},
    # entry_points={
    #     'console_scripts': [
    #         'kolab = kolab.main:main'
    #     ]
    # },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing',
    ],
    # test_suite='test_all.suite'
)