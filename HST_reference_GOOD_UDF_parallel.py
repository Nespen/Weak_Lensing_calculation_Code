#!/usr/bin/env python
# coding: utf-8

# In[12]:


import astropy
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import os
import numpy as np
import pylab
import asciitable
import pandas
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from sklearn.utils import shuffle
from scipy.optimize import minimize
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(mp.cpu_count()-2)


# In[2]:


#loading GOODS
#UDF_cat=np.genfromtxt("UDF_photo_z/table5.dat")
names=['x', 'y', 'F435W', 'F435We', 'F435Wa', 'F435Wae', 'F606W', 'F606We', 'F606Wa', 'F606Wae', 'F775W', 'F775We'
       , 'F775Wa', 'F775Wae', 'F850LP', 'F850LPe', 'F850LPa', 'F850LPae', 'z_w']
#colspecs = [(1, 5), (7, 12), (14, 23), (25, 34), (98, 105),(106, 113)]
GOODS_cat=ascii.read("n-goods_z.cat", names=names)
GOODS_cat2=ascii.read("s-goods_z.cat", names=names)
GOODS_cat=astropy.table.vstack([GOODS_cat, GOODS_cat2], join_type='outer', metadata_conflicts='warn')
#print(UDF_cat)
#UDF_catnp=UDF_cat.to_numpy()
#print(GOODS_cat[2])
GOODS_data=GOODS_cat.as_array()


# In[3]:


#loading UDF

#UDF_cat=np.genfromtxt("UDF_photo_z/table5.dat")
colspecs = [(0, 6), (7, 12), (14, 23), (25, 34), (98, 105),(106, 113),(114, 121),(824,829),(830,834),(835,839),(898, 906),(859, 864),(865, 870),(871, 876)]
UDF_cat=pandas.read_fwf("table5.dat", colspecs=colspecs)
print(len(UDF_cat))
UDF_catnp=UDF_cat.to_numpy()
#print(UDF_catnp[2])
#spec_cat=spec_cat[spec_cat["col4"].astype("float64")>1]
print(UDF_catnp[:,9])

#print(len(UDF_catnp[:,9][UDF_catnp[:,9]>5]))

#UDF_catnp=UDF_catnp[(UDF_catnp[:,9]-UDF_catnp[:,8])<0.5]
print(len(UDF_catnp))


f1=4
f1n='F775W'
#4
f2=5
f2n='F850LP'
#5
f3=6
f3n='F105W'


# In[4]:


#combining catalogs

cluster_name=("CL0152","CL1226","El_Gordo"  ,"ISCS1434+34","RCS2345","RDCS1252","XMMU1229","XMMU2235")



precomcat=np.zeros((GOODS_data.shape[0], 4))
precomcat[:,0]=GOODS_data['F775W']
precomcat[:,1]=GOODS_data['F850LP']
precomcat[:,2]=0    #GOODS_data['F105W'] does not exist
precomcat[:,3]=GOODS_data['z_w']
print(precomcat.shape)
complete_cat=precomcat
precomcat=np.zeros((UDF_catnp.shape[0], 4))
precomcat[:,0]=UDF_catnp[:,4]
precomcat[:,1]=UDF_catnp[:,5]
precomcat[:,2]=UDF_catnp[:,6]
precomcat[:,3]=UDF_catnp[:,10]
print(precomcat.shape)
complete_cat=np.append(complete_cat, precomcat, axis=0)
print(complete_cat.shape)


f1=0
f1n='F775W'
#4
f2=1
f2n='F850LP'
#5
f3=2
f3n='F105W'


# In[7]:


#calculating s
f1=0
f1n='F775W'
#4
f2=1
f2n='F850LP'
#5
f3=2
f3n='F105W'


m=28
filt=0
binsize=0.2
data=complete_cat

#data=data[data[:,f2]>26]
data=data[(data[:,f1]-data[:,f2])<0]
#data=data[(data[:,f2]-data[:,f3])<0.3]
#print(len(data))
def scalc(m,binsize,data,filt):

    #data=data[data[:,filt]<(m+1)]
    bins=np.arange((m-(5*binsize)),m+(5*binsize),binsize)
    #print(bins.shape)
    numcat=np.zeros(len(bins))
    for x in range(len(bins)):
        #lowerlimit=(m+(5*binsize)-(10*binsize)+(x*binsize))
        upperlimit=(m-(5*binsize)+((x+1)*binsize))
        #bincat=data[np.logical_and(data[:,filt]>lowerlimit,data[:,filt]<upperlimit)]
        bincat=data[data[:,filt]<upperlimit]
        number=bincat.shape[0]
        numcat[x]=number
        #print(number)
    #print(numcat)
    magcat=np.arange(m+(6*binsize)-(10*binsize),m+(6*binsize),binsize)
    #print(magcat)
    s_calc=np.polyfit(magcat, np.log(numcat), 2, rcond=None, full=True, w=None, cov=True)
    #print(s_calc)
    a=2*s_calc[0][0]
    b=s_calc[0][1]
    c=s_calc[0][2]
    s=a*m+b
   # print(s)
   # print(s_calc)
    return(s,a,b,c)
print(scalc(28,binsize,data,filt)[0])


# In[61]:


data=UDF_catnp
## udf filters
f1=4
f1n='F775W'
#4
f2=5
f2n='F850LP'
#5
f3=6
f3n='F105W'
##
data=data[(data[:,f1]-data[:,f2])<0]
#print(scalc(28,binsize,data,filt)[0])

filt=4
F850_values=np.arange(23,29,0.01)

m_range=np.arange(23,29,0.01)
binsize=0.2
data=UDF_catnp
data=data[(data[:,f1]-data[:,f2])<0]
iterations=0
#print(m_range)

for m in m_range:
    F850_values[iterations]=scalc(m,binsize,data,filt)[0]
    iterations=iterations+1
    print(m)
    print(scalc(m,binsize,data,filt)[0])
print(F850_values)


filt=5
F775_values=np.arange(23,29,0.01)

m_range=np.arange(23,29,0.01)
binsize=0.2
data=UDF_catnp
data=data[(data[:,f1]-data[:,f2])<0]
iterations=0
#print(m_range)
for m in m_range:
    F775_values[iterations]=scalc(m,binsize,data,filt)[0]
    iterations=iterations+1
print(F775_values)


filt=6
F105_values=np.arange(23,29,0.01)

m_range=np.arange(23,29,0.01)
binsize=0.2
data=UDF_catnp
data=data[(data[:,f1]-data[:,f2])<0]
iterations=0
#print(m_range)
for m in m_range:
    F105_values[iterations]=scalc(m,binsize,data,filt)[0]
    iterations=iterations+1
print(F105_values)
pylab.plot(m_range,  F850_values,label=' F850LP')
pylab.plot(m_range, F775_values,label=' F775W')
pylab.plot(m_range,  F105_values,label=' F105W')
pylab.title('s for different cutoff Magnitudes UDF')
pylab.xlabel('$m_{cut}$')
pylab.ylabel('s')
pylab.legend(fontsize=8)
pylab.savefig("./s_calcs/s_calcs_polyfit_UDF_binsize_2 ", dpi=2000)
pylab.show


# In[5]:


#covariance matrices

f1=0
f1n='F775W'
#4
f2=1
f2n='F850LP'
#5
f3=2
f3n='F105W'


realizations=2500
random_int=3
samples=10000
C=np.ndarray((10,realizations), dtype=float)
m=28
filt=0
binsize=0.1
data=complete_cat
data=data[(data[:,f1]-data[:,f2])<0]
print(len(data))
def Ccalc(m,binsize,data,filt,realizations,random_int,samples):



    #print(len(data))


    bins=np.arange((m-(5*binsize)),m+(5*binsize),binsize)
    C=np.ndarray((realizations,10), dtype=float)
    numcat=np.zeros(len(bins))
    for r in range(realizations):
        rdata=shuffle(data, n_samples=samples, random_state=random_int*r)
        for x in range(len(bins)):            
            upperlimit=(m-(5*binsize)+((x+1)*binsize))   
            bincat=rdata[rdata[:,filt]<upperlimit]
            number=bincat.shape[0]
            #print(number)
            C[r][x]=np.log(number)
    C_mean=np.ndarray((realizations), dtype=float)
    C_i_j=np.ndarray((10,10), dtype=float)
    #print(C[0])
    for y in range(len(bins)):
        for x in range(len(bins)):
            for r in range(realizations):
                C_mean[r]=(C[r][y]-np.mean(C[y]))*(C[r][x]-np.mean(C[x]))
            
            C_i_j[x][y]=np.mean(C_mean)
    return(C_i_j)
Ccalc(m,binsize,data,filt,realizations,random_int,samples)
cmatrix=Ccalc(m,binsize,data,filt,realizations,random_int,samples)
plt.imshow(cmatrix,origin='lower')
plt.colorbar(orientation='vertical')

#plt.savefig("./s_calcs/covariance matrix_samples10000", dpi=2000)
plt.show


# In[8]:


#Chi squared


data=complete_cat
s,a,b,c=scalc(m,binsize,data,filt)
C_test=np.array([[0.00040083,0.00040032],[0.00040032,0.00040007]])
realizations=500
random_int=3
samples=10000
var=[a,b,c]
print(var)
def Chi_sq(var,m,binsize,data,filt,realizations,random_int,samples):
    C_i_j=Ccalc(m,binsize,data,filt,realizations,random_int,samples)
    inv_C=np.linalg.inv(C_i_j)
    #print(inv_C)
    #print(np.dot(inv_C,C_i_j))
    #pylab.imshow(C_i_j,origin ="lower")
    #pylab.colorbar()

    data=data[(data[:,f1]-data[:,f2])<0]
    bins=np.arange((m-(5*binsize)),m+(5*binsize),binsize)   
    numcat=np.zeros(len(bins))
    D=np.zeros(len(bins))
    for x in range(len(bins)):            
        upperlimit=(m-(5*binsize)+((x+1)*binsize))   
        bincat=data[data[:,filt]<upperlimit]
        number=bincat.shape[0]
        magcat=np.arange(m+(6*binsize)-(10*binsize),m+(6*binsize),binsize)
        #print(magcat[x])
        #print(upperlimit)
        D[x]=(1/2*var[0]*magcat[x]**2+var[1]*magcat[x]+var[2])-np.log(number)
    #print("arrays")
    #print(D)
    #print(inv_C)

    first_out=np.dot(D,inv_C)
    second_out=np.dot(first_out,D)
    return second_out


Chi_min=minimize(Chi_sq, x0=var,  method="Nelder-Mead", args=(m,binsize,data,filt,realizations,random_int,samples))
print("Nelder-Mead Chi_min=")
print(Chi_min)





## udf filters
f1=0
f1n='F775W'
#4
f2=1
f2n='F850LP'
#5
f3=2
f3n='F105W'


def s_man(m,binsize,data,filt,realizations,random_int,samples):
    Chi_min=minimize(Chi_sq, x0=var,  method="Nelder-Mead", args=(m,binsize,data,filt,realizations,random_int,samples))
    F_value=Chi_min.x[0]*m+Chi_min.x[1]
    return(F_value)




realizations=500
random_int=3
samples=10000
m_range=np.arange(23,29,0.1)
binsize=0.2
data=complete_cat
data=data[(data[:,f1]-data[:,f2])<0]


filt=0
# Step 2: `pool.apply` the `
F850_values= [pool.apply(s_man, args=(m,binsize,data,filt,realizations,random_int,samples)) for m in m_range]
# Step 3: Don't forget to close
pool.close() 
print(F850_values)


#for m in m_range:
#    Chi_min=minimize(Chi_sq, x0=var,  method="Nelder-Mead", args=(m,binsize,data,filt,realizations,random_int,samples))
 #   F850_values[iterations]=Chi_min.x[0]*m+Chi_min.x[1]
 #   iterations=iterations+1
 #   print(m)
 #   print(scalc(m,binsize,data,filt)[0])
#print(F850_values)


filt=1


F775_values= [pool.apply(s_man, args=(m,binsize,data,filt,realizations,random_int,samples)) for m in m_range]
pool.close() 
print(F775_values)

#UDF cat
realizations=500
random_int=3
samples=2000
m_range=np.arange(23,29,0.1)
binsize=0.2

data=UDF_catnp
f1=4
f1n='F775W'
#4
f2=5
f2n='F850LP'
#5
f3=6
f3n='F105W'

data=data[(data[:,f1]-data[:,f2])<0]
filt=4
F850_values_UDF= [pool.apply(s_man, args=(m,binsize,data,filt,realizations,random_int,samples)) for m in m_range]

pool.close() 
print(F850_values_UDF)
filt=5
F775_values_UDF= [pool.apply(s_man, args=(m,binsize,data,filt,realizations,random_int,samples)) for m in m_range]



pylab.plot(m_range,  F850_values_UDF,label=' F850LP UDF only')
pylab.plot(m_range, F775_values_UDF,label=' F775W UDF only')
pylab.plot(m_range,  F850_values,label=' F850LP GOODS+UDF')
pylab.plot(m_range, F775_values,label=' F775W GOODS+UDF')
pylab.title('s for different cutoff Magnitudes own calcs')
pylab.xlabel('$m_{cut}$')
pylab.ylabel('s')
pylab.legend(fontsize=8)
pylab.savefig("./s_calcs/s_calcs_self_both_binsize_2 ", dpi=2000)
pylab.show


# In[ ]:




