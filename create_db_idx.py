import sqlite3
# import Quandl
import csv


# csv_file = open("london/raw/bigfile.csv","rt")
# reader = csv.reader(csv_file,delimiter=",")



conn = sqlite3.connect("london/LSE_DB.db")
c = conn.cursor()


s = ("create table lse_index ("
	"id INTEGER not null,"
	"dataset_code TEXT not null,"
	"database_code TEXT not null,"
	"sname TEXT not null,"
	"description TEXT,"
	"refreshed_at TEXT not null,"
	"newest_date TEXT not null,"
	"oldest_date TEXT not null,"
	"column_names TEXT not null,"
	"frequency TEXT not null,"
	"db_id INTEGER not null,"
	"PRIMARY KEY(id) on conflict replace)"
	)

c.execute(s) #CREATE table

# csv_file = open("london/raw/bigfile.csv","rt")
# reader = csv.reader(csv_file,delimiter=",")









