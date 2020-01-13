import csv
import sys

import pandas as pd
csv.field_size_limit(sys.maxsize)
# f = open("queue.txt", "r")
progress = 0
data = []
y= []
# for x in f:
    # with open('datasetGenerated.csv', mode='w') as data_file:
    #     data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# for row in f:
#     label = -1 if progress<100 else 1
#     # data_writer.writerow([row,label])
#     data.append(row)
#     y.append(label)
#     progress = progress + 1
#     print progress
# df = pd.DataFrame({'data':data,'y':y})
# print df.head()
# df.to_csv('datasetGenerated.csv')
#
#

with open('data_file_Generated.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # with open('data_file_Generated.csv', mode='w') as data_file:
    #     data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
            print (progress)
            if not row[0] :
                continue
            data.append(row[0])
            y.append(row[1])
            progress = progress + 1

df = pd.DataFrame({'data':data,'y':y})
print df.head()
df.to_csv('data_file_Generated.csv')
