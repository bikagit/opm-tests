import csv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltbis
path = os.getcwd()


krw = []
krnw = []
krM = []
sT = []
S2 = []

with open(path+'/../data/KrPcData_Swi0_06.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        S2.append(float(row[3]))
        krnw.append(float(row[6]))
        krw.append(float(row[5]))
#        krM.append(float(row[3]))
#        sT.append([float(row[4])])
     
     
#print(        np.array(S2).T)
with open(path+'/sat.csv', 'w', newline='') as file:
##        for S in Sh:
##        smax = .0
##        for s in S:
##        smax = max(s,smax)
        file.write(str(np.array(S2).T))
##        SandSmax.append([s, smax])
#        np.array(S2).T

#print(np.array(S2).T)

#import csv
#l =[]
#with open(path+'/sat.csv', newline='') as ipfile, open("op1.csv", "w") as opfile:
#    reader = csv.reader(ipfile)
#    writer = csv.writer(opfile)
##    next(reader)
#    for rec in reader:
#      for i in range(len(rec)):
##         if float(rec[i]) < 10:
#       l.insert(0,[rec[i]])
#         #if float(rec[i]) > 10 and float(rec[i]) < 20:
#         #  l.insert(1,[rec[i]])
#    writer.writerow(l)
        
plt.plot(S2[0:96], krnw[0:96],  'b',label = "KRNWD")
#plt.plot(S2[97:166], krnw[97:166], 'r', label = "KRNWI1")
plt.plot(S2[167:181], krnw[167:181],'--', label = "KRNWD2")
plt.plot(S2[182:214], krnw[182:214],'--', label = "KRNWD2")
plt.plot(S2[215:269], krnw[215:269],'--', label = "KRNWD2")
plt.plot(S2[270:338], krnw[270:338],'--', label = "KRNWD2")
#plt.plot(S2[339:395], krnw[339:395],'*-', label = "KRNWI2")
#plt.plot(S2[396:432], krnw[396:432],'*-', label = "KRNWI2")
#plt.plot(S2[433:450], krnw[433:450],'*-', label = "KRNWI2")
plt.legend(fontsize=12)
#plt.plot(S2, krw,label = "KRW")
plt.savefig("hystpic.png")
