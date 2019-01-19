import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = "/media/ankit/Data/ml/data/img2text"

def parse_file(fname):
	text = open(fname).read().strip()
	lines = text.split('\n')
	data = []
	for line in lines:
		xpos = re.findall('\[\[.*\]\].*y', line)[0]
		xpos = xpos[2:-5]
		xpos = list(map(int, xpos.split()))
		# print(xpos)

		ypos = re.findall('y.*\[\[.*\]\].*ornt', line)[0]
		ypos = ypos[5:-8]
		ypos = list(map(int, ypos.split()))
		# print(ypos)

		annot = re.findall('transcription.*\[u\'.*\'\]', line)[0]
		annot = annot[19:-2].strip()
		# print(annot)

		pos = [(i, j) for i, j in zip(xpos, ypos)]
		# print(pos)

		data.append({
				'pos': pos,
				'transcription': annot
			})
	return data

def read_data(ifname, tfname, debug=False):
	data = parse_file(tfname)
	img = np.array(Image.open(ifname))

	if debug:
		plt.imshow(img)
		for d in data:
			x = [p[0] for p in d['pos']]
			y = [p[1] for p in d['pos']]
			plt.plot(x, y)
			print(d['transcription'])
		plt.show()

	return {
		'image': img,
		'positions': [d['pos'] for d in data],
		'transcription': [d['transcription'] for d in data]
	}

def get_dataset_files(test=False):
	'''
	
	Get dataset files in form of (img file, txt file) pairs.
	Argument:
		test <boolean> : If True, get test dataset, else get training dataset.

	'''

	imgDir = ''
	txtDir = ''
	dataset = []

	if test:
		imgDir = os.path.join(DATA_DIR, "Images", "Test")
		txtDir = os.path.join(DATA_DIR, "Text", "Test")
	else:
		imgDir = os.path.join(DATA_DIR, "Images", "Train")
		txtDir = os.path.join(DATA_DIR, "Text", "Train")

	# Get files and sort to maintain order of img, txt pairs
	ifiles = sorted(os.listdir(imgDir))
	tfiles = sorted(os.listdir(txtDir))

	# Filter files
	ifiles = list(filter(lambda f: f.upper().endswith('.JPG'), ifiles))
	tfiles = list(filter(lambda f: f.upper().endswith('.TXT'), tfiles))

	# Join with root dir
	ifiles = [os.path.join(imgDir, f) for f in ifiles]
	tfiles = [os.path.join(txtDir, f) for f in tfiles]

	dataset = [(ifile, tfile) for ifile, tfile in zip(ifiles, tfiles)]
	return dataset

if __name__ == "__main__":
	print(get_dataset_files())