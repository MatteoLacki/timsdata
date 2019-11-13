### Wrapper around TIMS data for Bruker.


## Installation

# requirements
Linux or Windows only for now.
Python3.6 or higher, not tested on 3.8 yet.

From terminal (assuming you have python and pip included in the system PATH:

```{python}
pip3 install timsdata
```

For fresher versions:
```{python}
pip3 install git+https://github.com/MatteoLacki/timsdata
```

From development:
```{bash}
github clone https://github.com/MatteoLacki/timsdata
cd timsdata
pip3 install -e .
```

## Usage
```{python}
from timsdata.timsdata import TimsData

D = TimsData('path_to_your_data')
D.readScans(frame_no, min_scan, max_scan)
```

For more options, stay tuned for the MIDIA module!
(or write me about it at matteo.lacki@gmail.com)


