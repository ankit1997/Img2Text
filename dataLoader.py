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

def read_data():
	ind = 156
	ifname = os.path.join(DATA_DIR, "Images", "Train", "img{}.jpg".format(ind))
	tfname = os.path.join(DATA_DIR, "Text", "Train", "poly_gt_img{}.txt".format(ind))
	data = parse_file(tfname)
	img = np.array(Image.open(ifname))

	plt.imshow(img)
	for d in data:
		x = [p[0] for p in d['pos']]
		y = [p[1] for p in d['pos']]
		plt.plot(x, y)
		print(d['transcription'])
	plt.show()
		

if __name__ == "__main__":
	read_data()