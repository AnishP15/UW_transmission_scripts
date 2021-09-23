import re
import scipy.io as spio
from scipy.io import savemat
import numpy as np 
import sys
from numpy import matrix 
import cmath 
import os
import sys
from os import path
from functools import reduce
import matplotlib.pyplot as plt
import time

#start_time = time.time()

parameters_file = "Parameters.txt"
file = open(parameters_file, "r")
line = file.read()
sum_loop_time = 0

# Loading input values from Parameters.txt

start_idx = line.index("Orbitals set")
end_idx = line.index("Energy Range")
Orbitals = list(line[start_idx + len("Orbitals set"):end_idx].strip().split("\n"))
Orbitals = list(map(float, Orbitals))

start_idx = line.index("Energy Range")
end_idx = line.index("Inject Site (atoms number)")
energy = list(line[start_idx + len("Energy Range"):end_idx].strip().split("\n"))
energy = list(map(float, energy))

start_idx = line.index("Inject Site (atoms number)")
end_idx = line.index("Extract Site (atoms number)")
LSite = list(line[start_idx + len("Inject Site (atoms number)"):end_idx].strip().split("\n"))
LSite = list(map(float, LSite))

start = line.find("Extract Site (atoms number)") + len("Extract Site (atoms number)")
end = line.find("GammaL")
RSite_txt = list(line[start:end].split("\n"))
RSite = [i for i in RSite_txt if i]
RSite = list(map(float, RSite))

start_idx = line.index("GammaL")
end_idx = line.index("GammaR")
gammaL = line[start_idx + len("GammaL"):end_idx].strip().split("\n")
gammaL = ''.join(gammaL)
gammaL = float(gammaL)

start_idx = line.index("GammaR")
end_idx = line.index("Probes Site (atoms number)")
gammaR = line[start_idx + len("GammaR"):end_idx].strip().split("\n")
gammaR = ''.join(gammaR)
gammaR = float(gammaR)


start_idx = line.index("Probes Site (atoms number)")
end_idx = line.index("Broadening (for DOS)")
probe_site = list(line[start_idx + len("Probes Site (atoms number)"):end_idx].strip().split("\n"))
probe_site = list(map(float, probe_site))


start_idx = line.index("Broadening (for DOS)")
end_idx = line.index("Probe (for Decoh)")
broadening = line[start_idx + len("Broadening (for DOS)"):end_idx].strip().split("\n")
broadening = ''.join(broadening)
broadening = float(broadening)

list_characters = []
for l in file:
    for character in l:
        list_characters.append(character)



start_idx = line.index("Probe (for Decoh)")
end_idx = len(list_characters)-1
bProbe = list(line[start_idx + len("Probe (for Decoh)"):end_idx].strip().split("\n"))

with open(parameters_file) as f:
    matrix_name = f.readline().rstrip('\n')

# Load matrices
matrix = spio.loadmat(matrix_name+'.mat', squeeze_me=True)
matrix = matrix[matrix_name]

eta = 0
sizeH = len(matrix)
if len(matrix) != int(sum(Orbitals)):
    print("Matrix doesn't match with orbitals list!")

# Initialize gamma
sumSig = np.zeros((sizeH, sizeH))
sites = LSite + RSite

gamma1 = gammaL*np.ones(len(LSite)) 
gamma2 = gammaR*np.ones(len(RSite))
gamma = [item for item in gamma1] + [item for item in gamma2]
# This subroutine gets the exact location of the atom in the Hamiltonian based on the number of orbitals per atom

for i in range(0, len(sites)):
    isite = sites[i] 
    isite_num = int(isite) - 1
    sum_orbitals = 0
    TempSumOrb = np.sum(Orbitals[:isite_num+1])
 
    TempLen1 = int(TempSumOrb) - int(Orbitals[isite_num])
    TempLen2 = int(TempSumOrb)
    Len = TempLen2 - TempLen1
    for k in range(TempLen1, TempLen2):
        sumSig[k][k] = gamma[i] 

z = complex(0,-1)
sumSig = z * sumSig/2

# Loop inititalization
NE = len(energy)

# Check avialable files from Checkpoint
Tname = "Tran_"+matrix_name+"_gammaL_"+str(gammaL)+"_gammaR_"+str(gammaR)+"Ballistic"
qq=0
try: 
    mat = spio.loadmat(Tname, squeeze_me=True)
    T = mat["T"].tolist()
    if -1 in T:
        qq = T.index(-1)
    else:
        sys.exit("Transmission is already complete!")

except FileNotFoundError:
    T = -1*np.ones((1, NE))
    savemat(Tname+".mat", {"T": T})
    T = T[0]
    qq = 0

z = complex(1,0)

# Entering the loop

for ne in range(qq, NE):
    E = energy[ne]

    start_loop_time = time.time()

    # Matrix Inversion
    Gr = np.linalg.solve((E + z*eta)*np.eye(sizeH) - matrix - sumSig, np.eye(sizeH))
    end_loop_time = time.time()
    loop_time = end_loop_time - start_loop_time
    sum_loop_time += loop_time

    Gr = np.asmatrix(Gr)
    Ga = Gr.getH()

    # Transmission between left and right contact atoms
    Tmat = np.zeros((len(LSite), len(RSite)))

    for i in range(0,len(LSite)):
        isite = LSite[i] - 1 
        isite_num = int(float(isite))
        sumi_orbitals = 0
       
        TempSum = np.sum(Orbitals[:isite_num+1])
        TempLeni1 = int(TempSum - Orbitals[int(isite)])
        TempLeni2 = int(TempSum)
        Leni = TempLeni2 - TempLeni1
        Gammai = gammaL * np.eye(int(Leni))

        for j in range(0, len(RSite)):
            jsite = RSite[j] - 1
            jsite_num = int(float(jsite))
            sumj_orbitals = 0
           
            TempSum2 = np.sum(Orbitals[:jsite_num+1])
            TempLenj1 = int(TempSum2 - Orbitals[int(jsite)])
            TempLenj2 = int(TempSum2)
            Lenj = TempLenj2 - TempLenj1
            Gammaj = gammaR * np.eye(int(Lenj))

            Tmat[i][j] = np.real(np.trace(Gammai @ Gr[TempLeni1 : TempLeni2, TempLenj1 : TempLenj2] @ Gammaj @ Ga[TempLenj1 : TempLenj2, TempLeni1: TempLeni2]))

    T[ne] = sum(sum(Tmat)) # Ballistic Transmission is the sum of Tij
    spio.savemat(Tname+".mat", {"T": T, "Energy":energy})

    # Printing individual transmission values to track output
    print(E, T[ne])

print("Finished Ballistic Transmission!")

# Timing Output
#end_time = time.time()
#print("Total Run Time:", end_time - start_time)
#print("Total Time spent on inversion:", sum_loop_time)
#print("Average Time spent per inversion", sum_loop_time/NE)
#print("total points", NE)

# Graphing Output vs. Real Transmission values
#trans_mat = spio.loadmat('Tran_twoHemeParHisCysMult1Opt_gammaL_0.1_gammaR_0.1.mat', squeeze_me=True)
trans_mat = spio.loadmat('Tran_hemeMult3_gammaL_0.1_gammaR_0.1.mat', squeeze_me=True)

plot1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
plot2 = plt.subplot2grid((3, 3), (2, 0), colspan=2)

plot1.plot(range(0,NE), T, color ="red")
plot1.title.set_text("Output")
plot1.set_xlabel("Energy Range")
plot1.set_ylabel("T Values")
plot2.plot(range(0,NE), trans_mat["T"], color ="green")
plot2.title.set_text("Real Transmission Matrix")
plot2.set_xlabel("Energy Range")
plot2.set_ylabel("T Values")
plt.show()




