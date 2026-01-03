from setuptools import setup, find_packages

setup(
    name="ai-poweros",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    author="AI-PowerOS Team",
    description="Unified AI Personal Operating System",
    python_requires=">=3.9",
)
