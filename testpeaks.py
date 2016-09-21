import matplotlib
matplotlib.use("Agg")

import numpy as np
from detect_peaks import *
from peakdetect import *
import matplotlib.pyplot as plt
import scipy.signal
import sqlite3
import random as rand;
import datetime


# np.random.seed(99899)
# rand.seed(91123121)

def findpeaks(y,win_len=10,p=3,**kwargs):
	y_smooth = scipy.signal.savgol_filter(y,win_len,p);
	return(detect_peaks(y_smooth,kwargs))


def get_uq_qcodes(c,ts_limit):
	q = "select distinct q_code, count(q_code) from lse_data group by q_code having count(q_code)>?"
	return(c.execute(q,[ts_limit]).fetchall())

def get_timeseries(c,q_code):
	q = "select date_, price from lse_data where q_code=? order by date_ asc;"
	return(c.execute(q,[q_code]).fetchall())

conn = sqlite3.connect("london/LSE_DB.db")
c = conn.cursor()

uq_qcodes = get_uq_qcodes(c,500)
q_code = rand.choice(uq_qcodes)[0];

fmt = "%Y-%m-%d"


timeseries = get_timeseries(c,q_code);
x = [i[0] for i in timeseries];
y = np.array([i[1] for i in timeseries]);
y = y.astype(np.unicode_)
y = np.genfromtxt(map(lambda x: x.encode('UTF-8'),y),filling_values=np.nan) #this is needed because of reasons

if(np.sum(np.isnan(y))>0): #{
	idxs = np.arange(y.size)
	idxs_non_null = np.logical_not(np.isnan(y));
	y = np.interp(idxs,idxs[idxs_non_null],y[idxs_non_null]);
#}

#only for peak finding
y = y - np.min(y)
y = y / np.max(y)

# y = y - np.mean(y)
# y = y / np.std(y)


# print(np.mean(y))
# print(np.std(y))


ref_date = datetime.datetime.strptime("1900-01-01",fmt);
x = [datetime.datetime.strptime(i,fmt) for i in x];
x = [(i-ref_date).days for i in x];
x = np.array(x)
# x = x[non_null]


plt.figure()
plt.subplot(3,1,1)
plt.plot(x,y)


win_len = 21
p = 3
lookahead = 10;
delta = 0.05;

y_smooth = scipy.signal.savgol_filter(y,win_len,p);
# idxs = detect_peaks(y_smooth,mpd = 10,edge="falling")
max_idxs, min_idxs  = peakdetect(y_smooth,lookahead=lookahead,delta=delta)
max_idxs = np.array(max_idxs)
min_idxs = np.array(min_idxs)

if(max_idxs.size == 0 or min_idxs.size == 0): #no peaks
	plt.savefig("test.png")
	exit(1)

max_idxs = max_idxs[:,0];
min_idxs = min_idxs[:,0];		

max_idxs = max_idxs.astype(np.int)
min_idxs = min_idxs.astype(np.int)

x_peaks_min = x[min_idxs]
x_peaks_max = x[max_idxs]
y_peaks_min = y[min_idxs]
y_peaks_max = y[max_idxs]

plt.subplot(3,1,2)
plt.plot(x,y_smooth)

plt.subplot(3,1,3)
plt.plot(x,y)
plt.plot(x_peaks_min,y_peaks_min,"go")
plt.plot(x_peaks_max,y_peaks_max,"ro")

plt.savefig("test.png")


# plt.show()





























