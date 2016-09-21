import sqlite3
# import Quandl
import csv
import numpy as np

idxs = [0,1,2,3,4,5,6,7,8,9,12]

csv_file = open("london/raw_160619/bigfile.csv","rt")
reader = csv.reader(csv_file,delimiter=",")
rlist = list(reader)

conn = sqlite3.connect("london/LSE_DB.db")
c = conn.cursor()

q = "INSERT INTO lse_index VALUES (?,?,?,?,?,?,?,?,?,?,?)"

rlist = np.array(rlist);

n_rows = rlist.shape[0]
for i in range(1,n_rows):
	temp = rlist[i]
	temp = temp[idxs]
	c.execute(q,temp)
	print(i+1,"/",n_rows,end="\r")

conn.commit()
conn.close()

#0,1,2,3,4,5,6,7,8,9,12
# ['id',
 # 'dataset_code',
 # 'database_code',
 # 'name',
 # 'description',
 # 'refreshed_at',
 # 'newest_available_date',
 # 'oldest_available_date',
 # 'column_names',
 # 'frequency',
 # 'type',
 # 'premium',
 # 'database_id']





























