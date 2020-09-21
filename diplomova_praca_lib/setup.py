import setuptools

setuptools.setup(
    name="diplomova-praca-jankasvk",
    version="0.0.1",
    author="Jana Bátoryová",
    author_email="janulik11@gmail.com",
    description="Diplomova Praca lib",
    long_description_content_type="text/markdown",
    url="https://github.com/JankaSvK/thesis-grizzly",
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
        'tensorflow-gpu>=2.0,<2.1.0',
        # 'face-recognition',
        'Pillow',
        'numpy',
        'opencv-python',
        'scikit-learn',
        'MiniSom',
        'matplotlib'
    ]
)
