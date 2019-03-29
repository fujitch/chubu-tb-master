# -*- coding: utf-8 -*-

import csv

class datasetMaker:
    def datasetYellow(self):
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-VbT731.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        dataset = []
        for row in reader:
            dataset.append(row)
        
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-VbT.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        datasetAb = []
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb.append(row)
        return dataset, datasetAb
    
    def datasetGreen(self):
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-VbT731.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        dataset = []
        for row in reader:
            dataset.append(row)
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5B21-PI6.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[28])
            dataset[counter].append(row[29])
            dataset[counter].append(row[30])
            dataset[counter].append(row[31])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5C31-PLA.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[15])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5C31-PT016.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[1])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-EHC.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[7])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-PI6.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[2])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-PLA.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[2])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-TE.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[13])
            dataset[counter].append(row[14])
            dataset[counter].append(row[15])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-TI.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            dataset[counter].append(row[1])
            dataset[counter].append(row[2])
            dataset[counter].append(row[3])
            dataset[counter].append(row[4])
            dataset[counter].append(row[5])
            dataset[counter].append(row[6])
            dataset[counter].append(row[7])
            dataset[counter].append(row[8])
            dataset[counter].append(row[9])
            dataset[counter].append(row[10])
            dataset[counter].append(row[11])
            dataset[counter].append(row[12])
            dataset[counter].append(row[13])
            dataset[counter].append(row[14])
            dataset[counter].append(row[15])
            dataset[counter].append(row[16])
            dataset[counter].append(row[17])
            dataset[counter].append(row[18])
            dataset[counter].append(row[19])
            dataset[counter].append(row[20])
            dataset[counter].append(row[21])
            dataset[counter].append(row[22])
            dataset[counter].append(row[23])
            dataset[counter].append(row[24])
            dataset[counter].append(row[25])
            dataset[counter].append(row[26])
            dataset[counter].append(row[27])
            counter += 1
            
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-VbT.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        datasetAb = []
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb.append(row)
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5B21-PI6.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[28])
            datasetAb[counter].append(row[29])
            datasetAb[counter].append(row[30])
            datasetAb[counter].append(row[31])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5C31-PLA.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[15])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5C31-PT016.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[1])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-EHC.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[7])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-PI.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[2])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-PLA.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[2])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-TE.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[7])
            datasetAb[counter].append(row[8])
            datasetAb[counter].append(row[9])
            counter += 1
        tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-TI.csv", "r", encoding="utf_8")
        reader = csv.reader(tb_csv)
        header = next(reader)
        counter = 0
        for row in reader:
            if row[0].find("00:00") == -1 and row[0].find("10:00") == -1 and row[0].find("20:00") == -1 and row[0].find("30:00") == -1 and row[0].find("40:00") == -1 and row[0].find("50:00") == -1:
                continue
            datasetAb[counter].append(row[1])
            datasetAb[counter].append(row[2])
            datasetAb[counter].append(row[3])
            datasetAb[counter].append(row[4])
            datasetAb[counter].append(row[5])
            datasetAb[counter].append(row[6])
            datasetAb[counter].append(row[7])
            datasetAb[counter].append(row[8])
            datasetAb[counter].append(row[9])
            datasetAb[counter].append(row[10])
            datasetAb[counter].append(row[11])
            datasetAb[counter].append(row[12])
            datasetAb[counter].append(row[13])
            datasetAb[counter].append(row[14])
            datasetAb[counter].append(row[15])
            datasetAb[counter].append(row[16])
            datasetAb[counter].append(row[17])
            datasetAb[counter].append(row[18])
            datasetAb[counter].append(row[19])
            datasetAb[counter].append(row[20])
            datasetAb[counter].append(row[21])
            datasetAb[counter].append(row[22])
            datasetAb[counter].append(row[23])
            datasetAb[counter].append(row[24])
            datasetAb[counter].append(row[25])
            datasetAb[counter].append(row[26])
            datasetAb[counter].append(row[27])
            counter += 1
        return dataset, datasetAb
    
# test
if __name__ == '__main__':
    datasetMaker = datasetMaker()
    dataset, datasetAb = datasetMaker.datasetGreen()