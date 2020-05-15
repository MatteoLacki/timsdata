# -*- coding: utf-8 -*-
"""Python wrapper for timsdata.dll

Original code of the BRUKER company by:
Michael Krause, Sascha Winter, and Sven Bremer
only slightly pythonized by Mateusz K. Lacki.
"""

import numpy as np
import sqlite3
import os, sys
from ctypes import * #ugly
from functools import lru_cache
from pkg_resources import resource_filename as get_so_dll
from pathlib import Path
from platform import architecture, system as get_system

from .slice_ops import parse_idx
from .iterators import ComfyIter

system = get_system()
arch32or64, plat = architecture()

if system == 'Windows':
    if arch32or64 == '32bit':
        libname = get_so_dll('timsdata','cpp/win32/timsdata.dll')
    else:
        libname = get_so_dll('timsdata','cpp/win64/timsdata.dll')
elif system == 'Linux' and arch32or64 == '64bit':
    libname = get_so_dll('timsdata','cpp/libtimsdata.so')
elif system == 'Darwin':
    raise OSError('MacOS not yet supported')
else:
    raise OSError("This OS is not supported (yet).")

debug = False
dll = cdll.LoadLibrary(libname)
dll.tims_open.argtypes = [ c_char_p, c_uint32 ]
dll.tims_open.restype = c_uint64
dll.tims_close.argtypes = [ c_uint64 ]
dll.tims_close.restype = None
dll.tims_get_last_error_string.argtypes = [ c_char_p, c_uint32 ]
dll.tims_get_last_error_string.restype = c_uint32
dll.tims_has_recalibrated_state.argtypes = [ c_uint64 ]
dll.tims_has_recalibrated_state.restype = c_uint32
dll.tims_read_scans_v2.argtypes = [ c_uint64, c_int64, c_uint32, c_uint32, c_void_p, c_uint32 ]
dll.tims_read_scans_v2.restype = c_uint32
MSMS_SPECTRUM_FUNCTOR = CFUNCTYPE(None, c_int64, c_uint32, POINTER(c_double), POINTER(c_float))
dll.tims_read_pasef_msms.argtypes = [ c_uint64, POINTER(c_int64), c_uint32, MSMS_SPECTRUM_FUNCTOR ]
dll.tims_read_pasef_msms.restype = c_uint32

convfunc_argtypes = [ c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32 ]

dll.tims_index_to_mz.argtypes = convfunc_argtypes
dll.tims_index_to_mz.restype = c_uint32
dll.tims_mz_to_index.argtypes = convfunc_argtypes
dll.tims_mz_to_index.restype = c_uint32

dll.tims_scannum_to_oneoverk0.argtypes = convfunc_argtypes
dll.tims_scannum_to_oneoverk0.restype = c_uint32
dll.tims_oneoverk0_to_scannum.argtypes = convfunc_argtypes
dll.tims_oneoverk0_to_scannum.restype = c_uint32

dll.tims_scannum_to_voltage.argtypes = convfunc_argtypes
dll.tims_scannum_to_voltage.restype = c_uint32
dll.tims_voltage_to_scannum.argtypes = convfunc_argtypes
dll.tims_voltage_to_scannum.restype = c_uint32

def throwLastTimsDataError (dll_handle):
    """Throw last TimsData error string as an exception."""

    len = dll_handle.tims_get_last_error_string(None, 0)
    buf = create_string_buffer(len)
    dll_handle.tims_get_last_error_string(buf, len)
    raise RuntimeError(buf.value)

# Decodes a properties BLOB of type 12 (array of strings = concatenation of
# zero-terminated UTF-8 strings). (The BLOB object returned by an SQLite query can be
# directly put into this function.) \returns a list of unicode strings.
def decodeArrayOfStrings (blob):
    if blob is None:
        return None # property not set

    if len(blob) == 0:
        return [] # empty list

    blob = bytearray(blob)
    if blob[-1] != 0:
        raise ValueError("Illegal BLOB contents.") # trailing nonsense

    if sys.version_info.major == 2:
        return unicode(str(blob), 'utf-8').split('\0')[:-1]
    if sys.version_info.major == 3:
        return str(blob, 'utf-8').split('\0')[:-1]
        


class TimsData:
    def __init__ (self, analysis_directory, use_recalibrated_state=False):
        """Initialize TimsData.

        Args:
            analysis_directory (str, unicode string): path to the folder containing 'analysis.tdf'
            use_recalibrated_state (boolean): no idea what it does.
        """
        if sys.version_info.major == 2:
            if not isinstance(analysis_directory, unicode):
                raise ValueError("analysis_directory must be a Unicode string.")
        if sys.version_info.major == 3:
            analysis_directory = Path(analysis_directory)

        self.dll = dll
        self.handle = self.dll.tims_open(
            str(analysis_directory).encode('utf-8'),
            1 if use_recalibrated_state else 0 )
        if self.handle == 0:
            throwLastTimsDataError(self.dll)
        self.conn = sqlite3.connect(str(analysis_directory/"analysis.tdf"))
        self.initial_frame_buffer_size = 128
        self.iter = ComfyIter(self)


    def __del__ (self):
        """Destructor."""
        if hasattr(self, 'handle'):
            self.dll.tims_close(self.handle)
        self.conn.close()

            
    def __callConversionFunc(self, frame_id, input_data, func):
        """General wrapper to export function results to numpy arrays.

        Args:
            frame_id (int): The number of the considered frame.
            input_data (iterable of floats64, np.array): The floating point input.
            func (function): function to be called on the input.
        Returns:
            np.array: the evaluation of the 'func' on the 'input_data'.
        """
        if type(input_data) is np.ndarray and input_data.dtype == np.float64:
            # already "native" format understood by DLL -> avoid extra copy
            in_array = input_data
        else:
            # convert data to format understood by DLL:
            in_array = np.array(input_data, dtype=np.float64)
        cnt = len(in_array)
        out = np.empty(shape=cnt, dtype=np.float64)
        success = func(self.handle, frame_id,
                       in_array.ctypes.data_as(POINTER(c_double)),
                       out.ctypes.data_as(POINTER(c_double)),
                       cnt)
        if success == 0:
            throwLastTimsDataError(self.dll)
        return out


    def __massIdx2mz(self, frame_id, results_storage):
        success = func(self.handle, 
                       frame_id,
                       in_array.ctypes.data_as(POINTER(c_double)),
                       out.ctypes.data_as(POINTER(c_double)),
                       cnt)
        if success == 0:
            throwLastTimsDataError(self.dll)


    def indexToMz(self, frame_id, mass_idxs):
        """Translate mass indices (time of flight) to true mass over charge values.

        Args:
            frame_id (int): The frame number.
            mzs (np.array): mass indices to convert.
        Returns:
            np.array: mass over charge values."""
        return self.__callConversionFunc(frame_id, mass_idxs, self.dll.tims_index_to_mz)


    def mzToIndex(self, frame_id, mzs):
        """Translate mass over charge values to mass indices (time of flight).

        Args:
            frame_id (int): The frame number.
            mzs (np.array): Mass over charge to convert.
        Returns:
            np.array: Times of flight."""
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_mz_to_index)
        

    def scanNumToOneOverK0(self, frame_id, scans):
        """Translate scan number to ion mobility 1/k0.

        See 'oneOverK0ToScanNum' for invert function.

        Args:
            frame_id (int): The frame number.
            scans (np.array): Mass over charge to convert.
        Returns:
            np.array: Ion mobiilities 1/k0."""
        return self.__callConversionFunc(frame_id, scans, self.dll.tims_scannum_to_oneoverk0)


    def oneOverK0ToScanNum(self, frame_id, mobilities):
        """Translate ion mobilities 1/k0 to scan numbers.

        See 'scanNumToOneOverK0' for invert function.

        Args:
            frame_id (int): The frame number.
            mobilities (np.array): Ion mobility values to convert.
        Returns:
            np.array: Scan numbers."""
        return self.__callConversionFunc(frame_id, mobilities, self.dll.tims_oneoverk0_to_scannum)


    def scanNumToVoltage(self, frame_id, scans):
        """Translate scan number to voltages.

        See 'voltageToScanNum' for invert function.

        Args:
            frame_id (int): The frame number.
            scans (np.array): Mass over charge to convert.
        Returns:
            np.array: Voltages applied to release ions from TIMS."""
        return self.__callConversionFunc(frame_id, scans, self.dll.tims_scannum_to_voltage)


    def voltageToScanNum(self, frame_id, voltages):
        """Translate voltages to scan numbers.

        See 'voltageToScanNum' for invert function.

        Args:
            frame_id (int): The frame number.
            voltages (np.array): Voltages to convert.
        Returns:
            np.array: Scan numbers"""
        return self.__callConversionFunc(frame_id,
                                         voltages,
                                         self.dll.tims_voltage_to_scannum)


    def peakCnts_massIdxs_intensities(self,
                                      frame,
                                      scan_begin,
                                      scan_end,
                                      cut = False):
        """Get a numpy array with frame data.

        The first 'scan_end-scan_begin' positions contain counts of peaks per scan.
        Sum it to get the total number of peaks in a frame.
        After that, come the mass indices (recorded time) of the first scan, then intensities.
        Then the next scan and so on, until the last.
        For instance, for 3 scans it could look like:
        3 2 1 -1002434 -1002442 -4343332 9 10 23 13200234 2342421 2323424 9 9 10 33535423 10

        Args:
            frame (int): Frame number.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.
            cut (bool): cut the outcome to contain exactly the required data and nothing else. This takes additional 25% of time. 
        Returns:
            numpy.array: Frame (TIMS push) measurements.
        """
        while True:
            cnt = int(self.initial_frame_buffer_size) # for python 3.5
            buf = np.empty(shape=cnt, dtype=np.uint32)
            length = 4 * cnt
            reader = self.dll.tims_read_scans_v2
            # reader = self.dll.tims_read_scans_internal # 5% faster, not really worth it
            required_len = reader(self.handle,frame, scan_begin, scan_end, buf.ctypes.data_as(POINTER(c_uint32)), length)
            if required_len == 0:
                throwLastTimsDataError(self.dll)
            if required_len > length:
                if required_len > 16777216: # arbitrary limit for now...
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.initial_frame_buffer_size = required_len / 4 + 1 # grow buffer
            else:
                break
        if cut:
            d = scan_end-scan_begin
            return buf[0:int(d+buf[:d].sum())]
        else:
            return buf
    

    def iterScans(self, frame, scan_begin, scan_end):
        """An generator of scan spectra that omits empty scans.

        Args:
            frame (int, iterable, slice): Frames to output.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.
        Yields:
            tuple (scan number, mass indices, intensities)
        """
        buf = self.peakCnts_massIdxs_intensities(frame,
                                                 scan_begin,
                                                 scan_end,
                                                 False)
        d = scan_end - scan_begin
        for i in range(scan_begin, scan_end):
            npeaks = buf[i-scan_begin]
            if npeaks > 0:
                indices = buf[d : d+npeaks]
                d += npeaks
                intensities = buf[d:d+npeaks]
                d += npeaks
                yield (i, indices, intensities)


    def readScans(self, frame, scan_begin, scan_end):
        """Read a selection of scans from a given frame.

        Args:
            frame_id (int): Frame number.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.
        Returns:
            A list of tuples. Each tuple consists of two numpy arrays: m/z indices and intensities.
        """
        return [(MZidx,I) for _,MZidx,I in self.iterScans(frame,
                                                          scan_begin,
                                                          scan_end)]


    def frame_array(self, frame, scan_begin, scan_end):
        """Get a 2D array of data for a given frame and scan region.

        The output array contains three arrays.
        By default, these contain respectively ion mobility, mass to charge ratio, and intensity in the given frame.
        If selected raw, these will contain scan number, mass index and intensity.
        
        Args:
            frame (int, iterable, slice): Frames to output.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.
            raw (bool): Report raw data?

        Returns:
            tuple: np.array with frame number, scan numbers, mass to charge ratios/mass indices, and intensities in the selected frame.
        """
        #TODO: make all this in C++
        x = self.peakCnts_massIdxs_intensities(frame,
                                               scan_begin,
                                               scan_end,
                                               False)
        d = scan_end-scan_begin
        peak_cnts = x[:d]
        peak_cnts = peak_cnts.astype(np.int)
        tot_len = int(peak_cnts.sum())
        X = np.empty(shape=(tot_len,4), dtype=np.int) # for output
        if tot_len > 0:
            X[:,0] = frame
            X[:,1] = np.repeat(np.arange(scan_begin,scan_end), peak_cnts)
            m = 0
            for npeaks in peak_cnts[peak_cnts>0]:
                X[m:m+npeaks,2] = x[d:d+npeaks]
                d += npeaks
                X[m:m+npeaks,3] = x[d:d+npeaks]
                d += npeaks
                m += npeaks
        return X


    def iter_arrays(self, frames, scan_begin, scan_end):
        """Iter arrays containing a given scan range for the frames of interest.

        Args:
            frames (iterable): frames to report
            scan_begin (int): first scan
            scan_end (int): last scan (which is not included).

        Yields:
            np.array with frame number, scan numbers, mass to charge ratios/mass indices, and intensities in the selected frame.
        """
        for f in frames:
            frame = self.frame_array(f, scan_begin, scan_end)
            if len(frame): 
                yield frame


    def _iter(self, x):
        """Private iteration over arrays.

        You can call it explicitly, but better call it like D.iter[1:10].
        x is something that can be handled by __getitem__.
        """
        print(x)
        assert isinstance(x, tuple) and len(x) == 2, "Pass in [frames, scan_begin:scan_end], or [frames, scanNo]"
        frames, scans = x
        scan_begin, scan_end = parse_idx(scans)
        if isinstance(frames, slice) and frames.start is None:
            frames = slice(1, frames.stop)
        frames = np.r_[frames]
        yield from self.iter_arrays(frames, scan_begin, scan_end)


    def __getitem__(self, x):
        """Get a data array for given frames and scans.

        Args:
            x (tuple): First element corresponds to iterable/slice of frames. Second element corresponds to slice/iterable of scans. For now, only selection by scan_begin:scan_end is supported.
        Returns:
            np.array with frame numbers, scan numbers, mass to charge ratios/mass indices, and intensities in the selected frame.
        """
        arrays = list(self._iter(x))
        if arrays:
            return np.concatenate(arrays)
        else:
            return np.empty(shape=(0,4), dtype=np.int)


    def count_peaks(self, frame, scan_begin, scan_end):
        """Count peaks in a given frame and scan range.

        Args:
            frame (int, iterable, slice): Frames to output.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.

        Return:
            int: number of peaks.
        """
        x = self.peakCnts_massIdxs_intensities(frame,
                                               scan_begin,
                                               scan_end,
                                               False)
        return x[:scan_end-scan_begin].sum()


    def frameTIC(self, frame, scan_begin, scan_end, aggregate_scans=False):
        """Get the total ion count for given scans in a given frame.

        Args:
            frame (int, iterable, slice): Frames to output.
            scan_begin (int): Lower scan.
            scan_end (int): Upper scan.
            aggregate_scans (bool): Aggregate over scans or leave scans info.

        Returns:
            np.array or int: scans and intensities, or total intensity in selected scans.
        """
        x = self.peakCnts_massIdxs_intensities(frame,
                                               scan_begin,
                                               scan_end,
                                               False)
        scans_no = scan_end-scan_begin
        peaks_cnts = x[:scans_no]
        if not aggregate_scans:
            scans = np.nonzero(peaks_cnts)[0]
        peaks_cnts = peaks_cnts[peaks_cnts > 0].astype(np.int)
        LR = np.repeat(peaks_cnts,2)
        LR = np.insert(LR,0,0)
        LR = LR.cumsum() + scans_no
        I = np.add.reduceat(x, LR)[1::2]
        if aggregate_scans:
            return I.sum()
        else:
            return np.column_stack([scans, I])


    def readPasefMsMs(self, precursors):
        """Read peak-picked MS/MS spectra for a given list of precursors.
    
        Args:
            precursors (list): Precursors.

        Returns: 
            dict mapping 'precursor_id' to a pair of arrays (mz_values, area_values).
        """               
        precursors_for_dll = np.array(precursors, dtype=np.int64)
        result = {}
        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(precursor_id, num_peaks, mz_values, area_values):
            result[precursor_id] = (mz_values[0:num_peaks], area_values[0:num_peaks])
        
        rc = self.dll.tims_read_pasef_msms(self.handle,
                                           precursors_for_dll.ctypes.data_as(POINTER(c_int64)),
                                           len(precursors),
                                           callback_for_dll)
        if rc == 0:
            throwLastTimsDataError(self.dll)
        return result

	
    def readPasefMsMsForFrame(self, frame_id):
        """Read peak-picked MS/MS spectra for a given frame.

        Args:
            frame_id: frame number.

        Returns: 
            dict: maps 'precursor_id' to a pair of arrays (mz_values, area_values).
        """
        result = {}
        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(precursor_id, num_peaks, mz_values, area_values):
            result[precursor_id] = (mz_values[0:num_peaks], area_values[0:num_peaks])
        
        rc = self.dll.tims_read_pasef_msms_for_frame(self.handle,
                                                     frame_id,
                                                     callback_for_dll)

        if rc == 0:
            throwLastTimsDataError(self.dll)
        
        return result


    @lru_cache(maxsize=1)
    def list_tables_in_tdf(self):
        """Retrieve names of tables in 'analysis.tdf' SQLite3 database."""
        return [f[0] for f in self.conn.execute(f"SELECT name FROM sqlite_master WHERE TYPE = 'table'")]
