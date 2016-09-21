
import urllib.request
import os


# url_template = "https://www.quandl.com/api/v3/datasets.csv?database_code=CHRIS&per_page=100&sort_by=id&page=%s"
url_template = "https://www.quandl.com/api/v3/datasets.csv?database_code=LSE&per_page=100&sort_by=id&page=%s&api_key=-Mtn79XJPFoNyHWdyjfx"

directory = "london/raw_160619/"
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(1,54):
	url = url_template % i
	
	fn = "page_" + str(i) + ".csv"
	urllib.request.urlretrieve(url,directory+fn)
	print(directory+fn)
	
	
















