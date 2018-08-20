import requests
import shutil
import time

i = 24530
while i<100000:
	response = requests.get('https://picsum.photos/400/400/?random',stream=True)
	filepath = "./natural_image/"+str(i) + '.jpeg'
	with open(filepath, 'wb') as out_file:
	    shutil.copyfileobj(response.raw, out_file)
	print("Downloading " + filepath)
	time.sleep(0.5)
	i += 1
	del response