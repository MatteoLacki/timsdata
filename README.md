# Quick intro
**timsdata** is a module that wraps Bruker's tdf-sdk into a module, for convenience of not having to copy it in all possible projects.

# Requirements
Linux or Windows only for now.
Python3.6 or higher.


# Installation
From terminal (assuming you have python and pip included in the system PATH:

```{python}
pip3 install timsdata
```

For fresher versions:
```{python}
pip3 install git+https://github.com/MatteoLacki/timsdata
```

For development:
```{bash}
github clone https://github.com/MatteoLacki/timsdata
cd timsdata
pip3 install -e .
```

## Usage
```{python}
from timsdata.timsdata import TimsData

D = TimsData('path_to_your_data')

# choose some frame and scan limits
frame_no, min_scan, max_scan = 100, 0, 918

# pure Python structures
D.readScans(frame_no, min_scan, max_scan)

# numpy array
D.frame_array(frame_no, min_scan, max_scan)
```

Do observe, that you must know which values to put there.
If you don't, consider [TimsPy](https://github.com/MatteoLacki/timspy).


## Plans for future

We will gradually introduce cppyy to the project and fill up numpy arrays in C++.


## Law
Please read THIRD-PARTY-LICENSE-README.txt for legal aspects of using the software.
