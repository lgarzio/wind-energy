# wind-energy
Tools for analyzing and plotting wind energy data.

## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the wind-energy repository.

`git clone https://github.com/lgarzio/wind-energy.git`

Change your current working directory to the location that you downloaded wind-energy. 

`cd /Users/lgarzio/Documents/repo/wind-energy/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate wind-energy`

Install the toolbox to the conda environment from the root directory of the wind-energy toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.