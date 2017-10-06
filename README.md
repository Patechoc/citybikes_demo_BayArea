# Citybikes in the Bay Area

Example of a time-series prediction.

## Installation

### Installation with pip

```shell
$ git clone https://github.com/Patechoc/citybikes_demo_BayArea.git
$ cd citybikes_demo_BayArea
$ virtualenv myEnv
$ source myEnv/bin/activate

```

### Installation with conda

```shell
$ conda create -n citybikes_py36 python=3.6
$ source activate citybikes_py36
(citybikes_py36) $ python --version
Python 3.6.3
(citybikes_py36) $ conda install -y jupyter numpy pandas pytz dill progressbar2 geopy scipy bokeh
(citybikes_py36) $ pip install wget uszipcode sklearn
``` 



## Run the notebook

```shell
jupyter notebook cityBikes_BayArea.ipynb
```