import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diplomova-praca-jankasvk",  # Replace with your own username
    version="0.0.1",
    author="Jana Bátoryová",
    author_email="janulik11@gmail.com",
    description="Diplomova Praca lib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",  # TODO: link to right github
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['annotate_images=diplomova_prace_lib.annotate_images:main'], },
    install_requires=[
        'tensorflow>=2.0',
        'face-recognition',
        'Pillow',
        'numpy',
    ]
)
