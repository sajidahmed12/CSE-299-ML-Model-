from glob import glob

road_class = {
	"bad road" : "0",
	"perfect road" : "1",
	"mild bad road" : "2" ,
	"mild good road" : "3",
	"watered road" : "4"
}

labels = 'test_labels.csv'


for filename in glob("test/*"):
	temp = filename[5:]
	objectClass = road_class[temp]

	for img in glob(filename+"/*.jpg"):
		readpath = img.replace("\\","/")

		with open(labels,'a') as labelfile:
			labelfile.write(readpath+','+objectClass+'\n')

			#print(readpath+','+objectClass+'\n')
   
