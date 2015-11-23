import csv

data = []
header = []
with open('roc2014.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    skip = True
    for row in spamreader:
        if skip:
            header = row
            skip = False
            continue
        for i in range(len(row)):
            if float(row[i]) != 0.0:
                row[i] = 1
            else:
                row[i] = 0
        data.append(row)

for i in data:
    print i

with open('test_out.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|',
                                     quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(header)
    for entry in data:
        spamwriter.writerow(entry)

