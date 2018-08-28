#this is the helper function to detect if image is currupted

from skimage import io
import os
def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True



data_dir_list= ["./data/photo", "./data/vector-logo","./data/vector-non-logo"]
for data_dir in data_dir_list:
    print("Cleaning "+str(data_dir) + "...")
    for (dir, _, files) in os.walk(data_dir):
        for f in files:
        	path = os.path.join(dir, f)
        	if os.path.exists(path):
        		if verify_image(path):
        			print "Validate " + str(f) + " Good!"
        		else:
        			print "Currupted Images: " + str(f)
        			os.remove(path)
        	