import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tflyrics',
    version='0.2.1',
    author='Giorgio Giuffrè',
    author_email='giorgiogiuffre23@gmail.com',
    description='Generate lyrics with TensorFlow and the Genius API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='genius tensorflow lyrics scraper tflyrics',
    url='https://github.com/ggiuffre/tf-lyrics',
    project_urls={
        'Documentation': 'https://ggiuffre.github.io/tf-lyrics',
        'Source': 'https://github.com/ggiuffre/tf-lyrics/',
        'Tracker': 'https://github.com/ggiuffre/tf-lyrics/issues',
    },
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=['tensorflow>=2.0', 'beautifulsoup4'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing'
    ],
    python_requires='>=3.6',
)
