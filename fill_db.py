import numpy as np
import quandl
import sqlite3
import datetime
from datetime import date
import time

# mydata = Quandl.get("CHRIS/"+code[0], authtoken="-Mtn79XJPFoNyHWdyjfx", returns="numpy")



conn = sqlite3.connect("london/LSE_DB.db")
c = conn.cursor()

lse_index = c.execute("select * from lse_index")
lse_index = list(lse_index)

q = "insert into lse_data values(?,?,?,?,?)"

for idx in range(2501,len(lse_index)): #{
	stock = lse_index[idx];
	code = stock[2]+"/"+stock[1]
	if(code == "LSE/FALSE"):
		continue;
	print(idx+1,"/",len(lse_index)," : ",code)
	sdata = quandl.get(code, authtoken="-Mtn79XJPFoNyHWdyjfx", returns="numpy")
	if(len(sdata[0]) < 6):
		continue;
	
	
	for i in range(len(sdata)): #{
		params = [code,str(sdata[i][0].date())]
		for j in [1,4,5]:
			params.append("NULL" if str(sdata[i][j])=="nan" else str(sdata[i][j]))		
		c.execute(q,params);	
	
	if(idx%500 == 0):
		conn.commit()
		print("############### COMMITTED ###############")
		time.sleep(2)
	#}
#}	
conn.commit()
conn.close()
# s = ("create table lse_data ("
	# "q_code TEXT not null,"
	# "date_ TEXT not null,"
	# "price REAL,"
	# "volume REAL,"
	# "last_close REAL,"
	# "PRIMARY KEY(q_code,date_) on conflict ignore)"
	# )	
	
	



























