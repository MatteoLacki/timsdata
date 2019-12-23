"""Plot acquisition duty-cycle statistics for a PASEF TDF - Python 3.6 compatible

Variation of Sven's script by Matteo.
"""
%load_ext autoreload
%autoreload 2

import sqlite3
import numpy as np, matplotlib.pyplot as plt

from pathlib import Path
from MIDIA.fs import ls

projects = Path("/home/matteo/Projects/bruker/BrukerMIDIA")
midia = projects/"MIDIA_90min_diaPASEFruns"
analysis_dir = midia/"190917_AUR_200ngHeLa_90min_diaPASEF_py3_CE10_Slot1-1_1_544.d"

conn = sqlite3.connect(str(analysis_dir/"analysis.tdf"))

# check if range for frame ids where specified
if frame_id_high == 0:
    frame_id_high = conn.execute("SELECT MAX(Id) from Frames").fetchone()[0]

# Plot MS1 TIC
tmp = conn.execute("SELECT Id, SummedIntensities FROM Frames WHERE MsMsType=0 AND Id BETWEEN {0} AND {1} ORDER BY Id".format(frame_id_low, frame_id_high)).fetchall()
tic_ids = np.array([ tuple[0] for tuple in tmp ])
tic_intensities = np.array([ tuple[1] for tuple in tmp ])
#plt.plot(tic_ids, tic_intensities)
#plt.xlabel('frame number')
#plt.ylabel('MS1 TIC')

# Get times for precursor selection + scheduling (scheduling is really small) and frame-acquisition times
tmp = conn.execute("SELECT f.Id, p.Value FROM Frames f JOIN Properties p ON p.Frame=f.Id AND p.Property=(SELECT Id FROM PropertyDefinitions WHERE PermanentName='PrecSel_CompleteTime') AND p.Value NOT NULL ORDER BY f.Id").fetchall()
precsel_ids = [ tuple[0] for tuple in tmp ]
precsel_times = [ tuple[1] for tuple in tmp ]
tmp = conn.execute("SELECT Id, Time FROM Frames WHERE Id BETWEEN {0} AND {1} ORDER BY Id".format(frame_id_low, frame_id_high)).fetchall()
ids = np.array([ tuple[0] for tuple in tmp ])
times = np.array([ tuple[1] for tuple in tmp ])
timediffs = times[1:] - times[0:-1]

# Get frame-submission times
tmp = conn.execute("SELECT f.Id, p.Value FROM Frames f JOIN Properties p ON p.Frame=f.Id AND p.Property=(SELECT Id FROM PropertyDefinitions WHERE PermanentName='Timing_SubmitFrame') AND p.Value NOT NULL ORDER BY f.Id").fetchall()
submit_ids = [ tuple[0] for tuple in tmp ]
submit_times = [ tuple[1] for tuple in tmp ]

# Get theoretical time per frame [us]
def get_unique_value (query):
    tmp = conn.execute(query).fetchall()
    if len(tmp) != 1:
        raise RuntimeError('expect exactly one result row')
    return tmp[0][0]

cycletime_sec = 1e-6 * get_unique_value("SELECT DISTINCT(p.Value) FROM Properties p WHERE p.Property = (SELECT Id FROM PropertyDefinitions WHERE PermanentName='Digitizer_ExtractTriggerTime')")
#print cycletime_sec
numscans = get_unique_value("SELECT DISTINCT(NumScans) FROM Frames")
#print numscans
quenchtime_sec = 1e-3 * get_unique_value("SELECT DISTINCT(p.Value) FROM Properties p WHERE p.Property = (SELECT Id FROM PropertyDefinitions WHERE PermanentName='Collision_QuenchTime_Set')")
#print quenchtime_sec
exp_frame_time = numscans * cycletime_sec + quenchtime_sec

# number of empty MS frames
empty_ms = get_unique_value("SELECT COUNT(*) FROM Frames WHERE NumPeaks=0 AND MsMsType=0")
empty_msms = get_unique_value("SELECT COUNT(*) FROM Frames WHERE NumPeaks=0 AND MsMsType=8")

#print exp_frame_time
print ("Number of empty MS frames {}".format(empty_ms))
print ("Number of empty MSMS frames {}".format(empty_msms))
print ("Average abs(time excess) = {0:.2f} %".format(100 * np.mean(np.abs(timediffs - exp_frame_time)) / exp_frame_time))
print ("Average time excess = {0:.2f} %".format(100 * np.mean(timediffs - exp_frame_time) / exp_frame_time))
print ("Abs deviation from expected time {0:.6f}s".format(np.mean(timediffs - exp_frame_time)))
if 1 < len(precsel_times):
    print ("Average time precursor search + scheduling: {0:.3f}s".format((np.mean(precsel_times))))
print ("expected time for frame: {0}s".format(exp_frame_time))
print ("number of scans: {0}".format(numscans))
print ("trigger period: {0}".format(cycletime_sec))
print ("quench time: {0}".format(quenchtime_sec))

# Plot results
plt.figure()
plt.plot(ids[0:-1], timediffs)

plt.plot(precsel_ids, precsel_times, 'o', alpha=0.1)
plt.plot(submit_ids, submit_times, 'x', alpha=0.1)
plt.plot([ids[0],ids[-1]], [exp_frame_time,exp_frame_time], color=[1,0,0])
plt.legend(('time delta between consecutive frames', 'time for precsel + scheduling'))
plt.ylabel('time / sec')
plt.xlabel('frame number')
plt.ylim([0, 0.6])
plt.show()
