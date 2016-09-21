
import MySQLdb
import numpy as np
import Quandl 
import matplotlib.pyplot as plt

import market_sim


# db=MySQLdb.connect(read_default_file="~/.my2.cnf")

# c=db.cursor()

# c.execute("select dataset_code from michor.CHRIS_cfutures where name like '%#1 %'")

# codelist = c.fetchall()

# for code in codelist:
	# mydata = Quandl.get("CHRIS/"+code[0], authtoken="-Mtn79XJPFoNyHWdyjfx", returns="numpy")
	# dates_data = mydata['Date']	
	# volume_data = mydata['Volume']
	# settle_data = mydata['Settle']
	
	# print(np.mean(volume_data[-10:]))
	
	# plt.figure()
	# plt.plot(dates_data,settle_data)
	# plt.title(np.mean(volume_data[-10:]))
	# plt.savefig(code[0]+".png", bbox_inches='tight')
	# plt.close()
	
	
	
	
	



# print(len(codelist))
# print(codelist[0][0])




















