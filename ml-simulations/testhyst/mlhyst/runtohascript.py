import csv
import numpy as np
import os
import matplotlib.pyplot as plt


def evaluate(Sh, data, wp):


    with open(path+'/sat.csv', 'w', newline='') as file:
        for S in Sh:        
            for s in S:        
                file.write(str(s)+"\n")


    pathCall = "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
    os.system(pathCall)


    krw = []
    krnw = []
    krM = []
    sT = []


    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            #S2.append(float(row[0]))
            krnw.append(float(row[2]))
            krw.append(float(row[1]))
            krM.append(float(row[3]))
            sT.append(float(row[4]))


    i = 0
    start = 0
    end = 0
    for S in Sh:
        end = len(S) + start
        input = "D" + str(i)
        # if S[0] > S[-1]:
        #     input = "I" + str(i)
        #     i = i + 1
       
        plt.plot(S, krnw[start:end], label = "KRNW"+input)
        # plt.plot(S, krw[start:end],label = "KRW"+input)
        start = end


swl = 0.0
path = os.getcwd()
S = np.linspace( 0,1.0-swl, 10)
S2 = 1.0 - S - swl
#evaluate( [S, S2], "")
# evaluate([S,S2], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
smax = 0.

SSB = []
SSA = []
SSC = []

SS = []
for smax in S:
    SS.append(np.linspace(0,1-smax,30))

SSI = []
for smax in S:
    SSI.append(np.linspace(1-smax,1.0,30))



SSD = []
for smax in S:
    SSD.append(np.linspace(1.0,1-smax,30))

SS1 = []

for smax in S:
    SS1.append(np.linspace(0,smax, 10))


# evaluate(SSE, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# evaluate(SSD, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")


# SSB.append(np.linspace(0.75,1.0,30))
# SSB.append(np.linspace(0.7,1.0,30))
# SSB.append(np.linspace(0.6,1.0,30))
# SSB.append(np.linspace(0.5,1.0,30))
# SSB.append(np.linspace(0.4,1.0,30))
SSB.append(np.linspace(0.3,1.0,30))
# SSB.append(np.linspace(0.2,1.0,30))
SSB.append(np.linspace(0.1,1.0,30))



SSC.append(1-np.linspace(0.75,1.0,30))
SSC.append(1-np.linspace(0.7,1.0,30))
SSC.append(1-np.linspace(0.6,1.0,30))
SSC.append(1-np.linspace(0.5,1.0,30))
SSC.append(1-np.linspace(0.4,1.0,30))
SSC.append(1-np.linspace(0.3,1.0,30))
# SSB.append(np.linspace(0.2,1.0,30))
SSC.append(1-np.linspace(0.1,1.0,30))


# SSA.append(np.linspace(0.95,0,30))
# SSA.append(np.linspace(0.35,0.,30))
# SSA.append(np.linspace(0.6,0.,30))
# SSA.append(np.linspace(0.95,0.,30))
SSA.append(np.linspace(0.9,0.,30))
# SSA.append(np.linspace(0.4,0.,30))
# SSA.append(np.linspace(0.2,0.0,30))
SSA.append(np.linspace(1.0,0.10,30))
# SSA.append(np.linspace(0.1,0.5,30))
SSA.append(np.linspace(0.0,0.10,30))



S0 = np.linspace(0.9, 0., 50)
S1 = np.linspace(0., 0.9, 50)


S2 = np.linspace(0.5, 0.0, 50)

S3 = np.linspace(0.4, 0.0, 50)
S6 = np.linspace(0.3, 0.0, 50)
S7 = np.linspace(0.2, 0.0, 50)
S8 = np.linspace(0.6, 0.0, 50)

S0rev = np.linspace( 0.9,0.0, 50)
S1rev = np.linspace(0.12, 0.9, 50)


S2rev = np.linspace(0.5, 0.3, 50)

S3rev = np.linspace(0.4, 0.01, 50)
S6rev = np.linspace(0., 0., 50)
S7rev = np.linspace(0., 0.2, 50)
S8rev = np.linspace(0.9, 0., 50)


#evaluate([S1,S2,S3,S4], "1D_3PHASE_KILLOUGH_BOTH", "W")
# evaluate([S3rev,S1rev], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# evaluate([S0], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")

# evaluate([ S8, S3], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# evaluate([  S6, S7], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# evaluate([1-S7,1-S6,1-S3,1-S2,1-S8,1-S0], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# 
# evaluate([S0rev,S8rev,S2rev,S3rev], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")

# evaluate([1-S3,1-S2], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")

# evaluate([S0], "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")

# evaluate(SSI, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
evaluate(SSI, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")
# evaluate(SSA, "/Users/macbookn/activopmwkspc/edgedev/opm-tests/spe1/SPE1CASE2_2P", "WO")


plt.legend()
#plt.savefig("CARLSON.png")
plt.savefig("Killough.png")
# plt.show()