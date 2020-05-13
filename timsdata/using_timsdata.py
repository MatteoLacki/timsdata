"""Packaged Bruker's module."""
%load_ext autoreload
%autoreload 2
from timsdata.timsdata import TimsData

p = '/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d'
D = TimsData(p)

frame_no, min_scan, max_scan = 100, 0, 918
D.readScans(frame_no, min_scan, max_scan)
D.frame_array(frame_no, min_scan, max_scan)
# it's more documented, too: show timsdata.py

