# Crank-Nicolson method for propagation rho,z
# with air detail
# In summer 2021

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

start_time = datetime.now()


V = 1						# v = 0 for planar geometry and v = 1 for cylindrical geometry
landa = 800*1e-9            # Wave legenth
k0 = 2*np.pi/landa			# Centeral wave number
w = 10*1e-4					# Beam width

c = 3*1e8				    # Speed of light
n = 1						# Refractive index
n0 = 1						# Linear index coefficient


N= 1024					    # Number of grid points
Nz = 1000					# Number of steps

z = np.linspace(0,2,Nz)     # Discretizing propagation direction
r0 = np.linspace(0,0.02,N)  # Discretizing space domain

dr = abs(r0[2]-r0[1])	    # Grid size
dz = max(z)/Nz				# Step size
delta = dz/(4*k0*(dr**2))	# Normalized dispersion coefficient

##################### Definition Crank–Nicolson's u and V ##########################

u = np.zeros(N)			    # Definition Crank–Nicolson's u
v = np.zeros(N)			    # Definition Crank–Nicolson's v

for i in range(1,N-1):
    u[i] = 1 - V/(2*i)
    v[i] = 1 + V/(2*i)

#################### Definition L+ and L- #######################

L1 = np.zeros((N,N),dtype=np.complex_)				# Definition L+
L2 = np.zeros((N,N),dtype=np.complex_)				# Definition L-

for i in range(1,N-1):
    L1[i,i-1] = (0+1j)*delta*u[i]
    L1[i,i] = 1-(0+1j)*2*delta
    L1[i,i+1] = (0+1j)*delta*v[i]
    L2[i,i-1] = -(0+1j)*delta*u[i]
    L2[i,i] = 1+(0+1j)*2*delta
    L2[i,i+1] = -(0+1j)*delta*v[i]

################### Boundary conditions #####################

L2[0,0]=1+4*(0+1j)*delta
L2[0,1]=-4*(0+1j)*delta
L1[0,0]=1-4*(0+1j)*delta
L1[0,1]=4*(0+1j)*delta
L2[N-1,N-1]=1


IL2 = np.linalg.inv(L2)     # Inverse L-

################### Definition initial field #####################

E = np.zeros(N,dtype=np.complex_)

Sx = np.zeros((N,Nz),dtype=np.complex_)

E0 = np.sqrt(1)                 # Initial intensity of pulse
E = E0 * np.exp(-((r0)/w)**2)   # Beam structure
Sx[:,0] = E

################### Propagation field #####################


for j in range(1,Nz):
    nn = j
    
    E = np.matmul(L1,E)
    E = np.matmul(IL2,E)
        
    Sx[:,nn] = E
    if (nn%200) == 0:
        print(nn)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

################### Plot #####################

Ixz = abs(Sx)**2
Ixz = np.append(np.flipud(Ixz),Ixz,axis=0)
r0 = np.linspace(-0.005,0.005,N*2)

plt.contourf(z*1e2,r0*1e3,Ixz,levels=50, cmap='hot')
cbar=plt.colorbar(shrink=0.75)         # Colorbar
plt.ylim([-1,1])
plt.grid()
plt.xlabel('$z (cm)$')                 # Axes labels, title, plot and axes range
plt.ylabel('$R (mm)$')
plt.gca().set_aspect(20)
#plt.savefig('Ixz.png')
plt.show()                              # Displays figure on screen

plt.plot(z*1e2,np.max(Ixz,axis=0))
plt.grid()
plt.xlabel('$z (cm)$')                 # axes labels, title, plot and axes range
plt.ylabel('$Intensity (w/m^{-2})$')
plt.gca().set_aspect('auto')
plt.show() 
