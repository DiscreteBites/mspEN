from setuptools import setup, find_packages

setup(
	name="mspen",
	version="0.1.0",
	description="Implementing auto encoders + matrix subspace projection for electrodograms and neurograms",
	author="Marcus Ng",
	author_email="marcusngzhijie@gmail.com",
	packages=find_packages(),
	install_requires=[
		"numpy>=1.24",
		"scipy>=1.10",
		"pandas>=1.5",
		"matplotlib>=3.7",
		# Torch will be installed via the env.yml (CUDA 12.6 wheels).
	],
	extras_require={
		"dev": [
            "black", 
            "ruff", 
            "pytest",
            "notebook", 
            "ipykernel",
            "pympler"
        ],
	},
	python_requires=">=3.10,<3.12"
)
