import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
p_version = "0.0.1"

setuptools.setup(
    name="topk_classification", 
    version=p_version,
    author="Priscilla Chauke",
    author_email="priscilla.chk@gmail.com",
    description="Top-k classification and model selector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/priscilla-chk/top_k_models",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','pandas'],
)
