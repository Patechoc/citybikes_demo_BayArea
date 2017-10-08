# Citybikes in the Bay Area

Example of a time-series prediction.


## Installation

Clone the repository:

```shell
$ git clone https://github.com/Patechoc/citybikes_demo_BayArea.git
$ cd citybikes_demo_BayArea
```

Then you can either us  [`Anaconda`](#installation-with-conda) or [`pip`](#installation-with-pip) to install your Python environment with the right packages:

### Installation with pip

```shell
$ virtualenv myEnv
$ source myEnv/bin/activate
$ pip install -r pip_requirements.txt
```

### Installation with conda

```shell
$ conda create -n citybikes_py36 python=3.6 --file conda_requirements.txt
``` 


## Run the notebook

```shell
jupyter notebook cityBikes_BayArea.ipynb
```