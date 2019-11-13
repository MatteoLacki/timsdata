from timsdata.timsdata import TimsData
import sys
from pathlib import Path

from pprint import pprint


def test_tims_data():
    p = '/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d'
    T = TimsData(p)
    pprint(T.readScans(1000,0,918))

if __name__ == '__main__':
    _, path = sys.args
    test_tims_data(str(Path(path)))
