import math
import os

path = "C:/Users/svkpr/Sesharamanujam-SrivenkataKrishnan/001/"

directory = os.fsencode(path)
avg_distance = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    new_filename = '001/' + filename
    with open(new_filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    values = content[1].split(' ')
    xmin =int(values[0])
    ymin = int(values[1])
    xmax = int(values[2])
    ymax = int(values[3])
    
    distance = math.sqrt(math.pow((xmax - xmin),2) + math.pow((ymax - ymin),2))
    avg_distance.append(distance)
print(round(sum(avg_distance) / (len(avg_distance) * 2)))
