#!/usr/bin/env python
# coding: utf-8

# # Multiprocessing

# > Multiprocessing refers to the ability of a system to support more than one processor at the same time. Applications in a multiprocessing system are broken to smaller routines that run independently. The operating system allocates these threads to the processors improving performance of the system.
# 
# https://www.geeksforgeeks.org/multiprocessing-python-set-1/

# ## Multiprocessing in Python
# * `multiprocessing` for a single node with several cores
# * `joblib` also works in distruibuted nodes
# * `dask` for arrays in distruibuted nodes

# ### `multiprocessing`

# Example
# 
# See: https://towardsdatascience.com/how-can-data-scientists-use-parallel-processing-17194fffc6d0

# In[1]:


from multiprocessing import Pool


# In[2]:


def f(x):
    return x**2


# In[176]:


result = [f(x) for x in list(range(100000))]


# In[177]:


pool = Pool(8)
result = pool.map(f,list(range(100000)))
pool.close()


# ![img](https://miro.medium.com/max/700/1*n8_M7_0O2Rp3TCuqLDeeHg.png)

# Implementation

# In[3]:


import numpy as np
import itertools
import sys
from anomalies import anomaly
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

z=anomaly.free

def _get_chiral(q,q_max=np.inf):
    #Normalize to positive minimum
    if 0 in q:
        return None,None
    if q[0]<0:
        q=-q
    #Divide by GCD
    GCD=np.gcd.reduce(q)
    q=(q/GCD).astype(int)
    if ( #not 0 in z and 
          0 not in [ sum(p) for p in itertools.permutations(q, 2) ] and #avoid vector-like and multiple 0's
          #q.size > np.unique(q).size and # check for at least a duplicated entry
          np.abs(q).max()<=q_max
           ):
        return q,GCD
    else:
        return None,None
    
def get_solution(l,k,zmax=30):
    q,gcd=_get_chiral( z(l,k) )
    #if q is not None and np.abs(q).max()<=zmax:#
    if q is not None and np.abs(q).max()<=zmax:
        return {'l':l,'k':k,'z':list(q),'gcd':gcd}
    else:
        return {}    

def get_solution_from_list(lk,zmax=30):
    n=len(lk)
    l=lk[:n//2]
    k=lk[n//2:]
    return get_solution(l,k,zmax)


assert get_solution_from_list([1,2,1,-2])['z']==[1, 1, 1, -4, -4, 5]


# Prepare running

# In[4]:


d=[{'n':6,'N':300000,'max':7},
   {'n':7,'N':500000,'max':10},
   {'n':8,'N':800000,'max':12},
   {'n':9,'N':700000,'max':12},
   {'n':10,'N':26000000,'max':12},
   {'n':11,'N':90000000,'max':12},
   {'n':12,'N':80000000,'max':12}]

n=6
dd=[dd for dd in d if dd.get('n')==n][0]

N=dd['N'] 
mm=n-2

Single=False


# ### Single-processing

# In[5]:


if Single:
    s=time.time()
    ll=np.random.randint(1,dd['max']+1,(N,mm))*(-1)**np.random.randint(0,2,(N,mm))
    ll=np.unique(ll, axis=0)
    print('grid → ',time.time()-s,ll.shape)

    s=time.time()
    sls=[get_solution_from_list(lk) for lk in ll if get_solution_from_list(lk)]
    print('sols → ',time.time()-s,len(sls))
    del ll


# ### Multiprocessing
# After
# ```bash
# pip3 install dask[complete]
# ```

# In[6]:


import dask.array as da
import pandas as pd
from multiprocessing import Pool


# In[7]:


#axis parameter not yet implemented in dask: `da.unique` → https://stackoverflow.com/a/53389741/2268280
ll=da.random.randint(1,dd['max']+1,(N,mm))*(-1)**da.random.randint(0,2,(N,mm))
ll=ll.to_dask_dataframe().drop_duplicates().to_dask_array()

s=time.time()
ll=ll.compute()
print('grid → ',time.time()-s,ll.shape)

s=time.time()
pool = Pool(8)
sls = pool.map(get_solution_from_list,ll)
pool.close()

sls=[d for d in sls if d]
print('sols → ',time.time()-s,len(sls))


# Unique solutions

# In[10]:


df=pd.DataFrame(sls)
df.sort_values('gcd')
df['zs']=df['z'].astype(str)
df=df.drop_duplicates('zs').drop('zs',axis='columns').reset_index(drop=True)
print('unique solutions → ',df.shape)


# In[ ]:




