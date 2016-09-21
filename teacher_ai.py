import numpy as np
# from detect_peaks import *
from peakdetect import *
import scipy.signal
import sqlite3
import random as rand;
import datetime





class teacher_ai(object): #{
	def __init__(self,y_in,win_len = 21, p=3, lookahead=10,delta=0.05): #{
		y = y_in;
		y = y - np.min(y)
		y = y / np.max(y)
		
		y_smooth = scipy.signal.savgol_filter(y,win_len,p);
		max_idxs, min_idxs = peakdetect(y_smooth,lookahead=lookahead,delta=delta)
		max_idxs = np.array(max_idxs)
		min_idxs = np.array(min_idxs)
	
		if(max_idxs.size == 0 or min_idxs.size == 0): #no peaks
			max_idxs = np.array([])
			min_idxs = np.array([])
		else:
			max_idxs = max_idxs[:,0];
			min_idxs = min_idxs[:,0];		
			max_idxs = max_idxs.astype(np.int)
			min_idxs = min_idxs.astype(np.int)
			
		self.min_idxs = min_idxs;
		self.max_idxs = max_idxs;	
	#}

	def get_action(self, idx): #{
		if idx in self.min_idxs:
			return 1;
		elif idx in self.max_idxs:
			return 2;
		else:
			return 0;
	#}




#}

























