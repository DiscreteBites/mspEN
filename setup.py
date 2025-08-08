from setuptools import setup, find_packages

setup(
	name="mspen",
	version="0.1.0",
	description="Implementing auto encoders + matrix subspace projection for electrodograms and neurograms",
	author="Marcus Ng",
	author_email="marcusngzhijie@gmail.com",
	packages=find_packages(),
	install_requires = [
		'numpy>=1.26,<2.0; platform_system == "Linux"',
		'scipy>=1.10,<1.14; platform_system == "Linux"',
		'h5py>=3.10; platform_system == "Linux"',
		'scikit-learn>=1.5,<1.6; platform_system == "Linux"',

		'python-dotenv>=1.1.1',
		'pandas>=2.3.1,<3.0.0',
		'seaborn>=0.13.2,<0.14.0',
		'bidict>=0.23.1',
		'pyprojroot>=0.3.0',
        'tqdm>=4.66.0'
	],
	python_requires=">=3.10,<3.12"
)
