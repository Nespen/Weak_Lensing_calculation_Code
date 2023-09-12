#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import astropy
import pylab
import scipy
from scipy.integrate import simps
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import camb
from camb import get_matter_power_interpolator
from camb import model, initialpower
import colossus.cosmology
from colossus.cosmology import cosmology
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
h = 0.7
omega_b  = 0.025
omega_m = 0.3
omega_c = omega_m - omega_b
omega_de=0.7
ns=0.965
cosmo=FlatLambdaCDM(H0=h*100, Om0=omega_m)
astropy_cosmo = astropy.cosmology.FlatLambdaCDM(H0=h*100, Om0=omega_m, Ob0=omega_b)
pars.set_cosmology(H0=h*100, ombh2= omega_b*h**2, omch2=omega_c*h**2, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=ns, r=0)
pars.set_matter_power( kmax=500)#necessary 
pars.set_for_lmax(2500, lens_potential_accuracy=0)
#pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
PK = results.get_matter_power_interpolator(pars, k_hunit =False, hubble_units = False)
sigma_8=results.get_sigma8()


C=const.c.to(u.Mpc/u.s)

pi=np.pi
G=const.G

zl=0.3
zs=2.0
con=6.0

M_nl=1
M=1


astropy_cosmo.name='my_cosmo'
colossus_cosmo = cosmology.fromAstropy(astropy_cosmo, sigma_8, ns, name = 'my_cosmo')
D=colossus_cosmo.growthFactorUnnormalized(zl) # growth factor

r_200=1.0*u.Mpc
roh_c=cosmo.critical_density(zl).to(u.M_sun/u.Mpc**3)
roh_c0=cosmo.critical_density(0).to(u.M_sun/u.Mpc**3)
delta_c=roh_c-roh_c0
rohc0=roh_c0.value
rohc=roh_c.value
deltac=delta_c.value
r_xi=cosmo.comoving_distance(zl)
a=cosmo.scale_factor(zl)
D_s=cosmo.angular_diameter_distance(zs)
D_1=cosmo.angular_diameter_distance(zl)
D_1s=cosmo.angular_diameter_distance_z1z2(zl, zs)
print(con)
print(sigma_8)
print(delta_c)
print(roh_c)
print(D_s, D_1, D_1s)
print(C, G)
"""def D(z):
    a=cosmo.scale_factor(z)
    y=np.arange(100)/100+10**-20
    return(2.5*omega_m*scipy.integrate.simps(1/((y**3)*(omega_m*y**-3+omega_de*(1-omega_m-omega_de)**-2)**1.5)), y)"""

Sigma_crit=(((C**2)/(4*pi*G))*D_s/(D_1*D_1s)).to((u.M_sun)/u.Mpc**2)
print(Sigma_crit)
def b1(M, M_nl):
    x=M/M_nl
    return(0.53+0.39*(x**0.45)+(0.13/(40*x+1)+5*(10**-4)*(x**1.5)))
    
def r_s(r_200, con):
    return(r_200/con)
rs=r_s(r_200, con).value
print(rs)
#def Sigma_crit(D_s, D_1, D_1s):
 #   return ((C.value**2)/(4*pi*G.value))*D_s/(D_1*D_1s)

def delta_c(con):
    return ((200*(con**3))/(3*np.log10(1+con)-(con/(1+con))))
check=((2*rs*deltac*rohc)/(((0.5/rs)**2)-1))#*(1-(2/np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(1+x))))
print('check')
print(check)
def Sigma_NFW(R, r_s, delta_c, roh_c):
    x=(R/r_s)
    if x < 1.0:
        return(((2*r_s*delta_c*roh_c)/(((x**2)-1)*1000000))*(1-(2/np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(1+x)))))
    elif x == 1.0:
        return((2*r_s*delta_c*roh_c)/(3*1000000))
    elif x > 1.0:
        return(((2*r_s*delta_c*roh_c)/(((x**2)-1)*1000000))*(1-(2/np.sqrt((x**2)-1))*np.arctan(np.sqrt((x-1)/(1+x)))))
    else:
        print('error: z not a number')
        return 0
    
def Sigma_NFW_bar(R, r_s, delta_c, roh_c):
    x=(R/r_s)
    if x < 1.0:
        return(((4*r_s*delta_c*roh_c)/((x**2)*1000000))*((2/np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(1+x)))+np.log(x/2)))
    elif x == 1.0:
        return((4*r_s*delta_c*roh_c)*(1+np.log(1/2))/1000000)
    elif x > 1.0:
        return(((4*r_s*delta_c*roh_c)/((x**2)*1000000))*((2/np.sqrt((x**2)-1))*np.arctan(np.sqrt((x-1)/(1+x)))+np.log(x/2)))
    else:
        print('error: x not a number')
        return 0
vSigma_NFW=np.vectorize(Sigma_NFW)
def xi(r):
    k=10**(5*np.arange(500)/500 -3)
    #x_axis=np.arange(200)
    #for x in np.arange(200):
       # y_xi=10**(-3*np.arange(200)/200 +4)
       # x_axis[x]=(y_xi[x]**2)*((np.sin(y_xi[x]*r)/(y_xi[x]*r))*PK.P(zl, y_xi[x]))
    #print(x_axis)
    return((1/(2*(pi**2)))*scipy.integrate.simps((k**2)*((np.sin(k*r)/(k*r))*PK.P(0, k)), k))  
vxi=np.vectorize(xi)


def Sigma_1(R, z):
    y=np.arange( 100)/50
    #print(y)
    #print(R)
    #print(zxi((1+z)*np.sqrt((R**2)+(y**2))))
    # x_axis=y
    #for x in np.arange(200):
       # y=np.arange(-100, 100)*100
        #x_axis[x]=(xi((1+z)*np.sqrt((R**2)+(y[x]**2))))    
    return(((1+z)**3)*roh_c0.value*scipy.integrate.simps((vxi((1+z)*np.sqrt((R**2)+(y**2)))), y))
vSigma_1=np.vectorize(Sigma_1)
def Sigma_2Halo(R, z):
    return(b1(M, M_nl)*omega_m*(sigma_8**2)*(D**2)*vSigma_1(R, z))

def Sigma_2Halo_bar(R, z):
    R_sign=R*(np.arange(200)/200+0.01)
    return(2/(R**2)*scipy.integrate.simps((R_sign*Sigma_2Halo(R_sign, z)), R_sign))

def kappa(R, z, r_s, delta_c, roh_c):
    return((Sigma_NFW(R, r_s, delta_c, roh_c)+Sigma_2Halo(R, z))/Sigma_crit.value)

def gamma(R, z, r_s, delta_c, roh_c):
    return((Sigma_2Halo_bar(R, z)-Sigma_2Halo(R, z)+Sigma_NFW_bar(R, r_s, delta_c, roh_c)-Sigma_NFW(R, r_s, delta_c, roh_c))/Sigma_crit.value)
def mu(R, z, r_s, delta_c, roh_c):
    return(1/(((1-kappa(R, z, r_s, delta_c, roh_c))**2)-(gamma(R, z, r_s, delta_c, roh_c)**2)))






vmu=np.vectorize(mu)
vSigma_2Halo_bar=np.vectorize(Sigma_2Halo_bar)
vkappa=np.vectorize(kappa)
vgamma=np.vectorize(gamma)
vSigma_2Halo=np.vectorize(Sigma_2Halo)
R=np.arange(100)/100*3+0.25
vSigma_NFW=np.vectorize(Sigma_NFW)
vSigma_crit=np.vectorize(Sigma_crit)

M=-20
M_star=-20.84
alpha_LF=-1.6
alpha=(10**(0.4*M_star-M))-alpha_LF-1
w_lens=((alpha-1)**2)*(vmu(R, zl, rs, deltac, rohc)-1)
#w=(1/N)*np.sum(w_lens)



print(R)

yval=vSigma_2Halo(R, zl)
yval=vgamma(R, zl, rs, deltac, rohc)
#yval=w_lens
print(yval)
pylab.plot(R, yval)
#pylab.yscale('log')
pylab.savefig("./gamma.png")
pylab.show


#print(Sigma_NFW(1*u.Mpc, r_s(r_200, con), delta_c(con), roh_c))


# In[ ]:





# In[ ]:




