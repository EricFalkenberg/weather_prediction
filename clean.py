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
        new = []
        if float(row[0]) != 0:
            if float(row[1]) != 0:
                new = [1.0]
            else:
                new = [0.666]
        else:
            if float(row[1]) != 0:
                new = [0.333]
            else:
                new = [0.00]
        data.append(new)

with open('test_out.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|',
                                     quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['rain/snow'])
    for entry in data:
        spamwriter.writerow(entry)

