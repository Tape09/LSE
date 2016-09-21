import sqlite3
# import Quandl
import csv


# csv_file = open("london/raw/bigfile.csv","rt")
# reader = csv.reader(csv_file,delimiter=",")



conn = sqlite3.connect("london/LSE_DB.db")
c = conn.cursor()


s = ("create table lse_data ("
	"q_code TEXT not null,"
	"date_ TEXT not null,"
	"price REAL,"
	"volume REAL,"
	"last_close REAL,"
	"PRIMARY KEY(q_code,date_) on conflict ignore)"
	)

c.execute(s) #CREATE table

# csv_file = open("london/raw/bigfile.csv","rt")
# reader = csv.reader(csv_file,delimiter=",")









