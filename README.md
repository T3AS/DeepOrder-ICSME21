# DeepOrder

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Implementation of DeepOrder for the paper "DeepOrder: Deep Learning for Test Case Prioritization in Continuous Integration Testing".


## Installation

Clone the GitHub repository and install the dependencies.
1. Clone the repo and go to the directory 
```
$ git clone https://github.com/DeepOrder-ICSME21/DeepOrder.git
$ cd DeepOrder

```
2. Install Anaconda (for creating and activating a separate environment)
3. Run: 
```
$ conda create -n DeepOrder python==3.6
$ conda activate DeepOrder
```
4. Inside the enviroment, run:
```
$ pip install -r requirements.txt
```
## Instructions

Download the datasets from [here](https://drive.google.com/drive/folders/14QMWd7ltb9c9NsCTw-WOn9rmJHE5xiwM?usp=sharing).

There are 4 python scripts leading to 4 separate models of DeepOrder on their datasets respectively. 

For running all the scripts together use: 
```

$ ./scripts_all.sh

```

For extra visualization presented in the paper, run:
```

$ python Extra_Visualizations/APFD_NAPFD_test_history/Effect_of_test_history_APFD_NAPFD.py
$ python Extra_Visualizations/Comparison_with_RETECS/DeepOrder_Vs_RETECS.py

```



