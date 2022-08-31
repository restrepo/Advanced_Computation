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
import time


# In[2]:


def f(x):
    return x**2


# In[176]:


N=100000#00
result = [f(x) for x in range(N)]


# In[177]:


pool = Pool(8)
result = pool.map(f,range(N))
pool.close()


# ![img](https://miro.medium.com/max/700/1*n8_M7_0O2Rp3TCuqLDeeHg.png)

# ## Implementation.
# * Use the official module to find solutions 
# * filter the chiral ones  with a maximum integer of 32
# * Build a function suitable for multiprocessing

# In[1]:


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
          np.abs(q).max()<=q_max
           ):
        return q,GCD
    else:
        return None,None
    
def get_solution(l,k,zmax=32):
    q,gcd=_get_chiral( z(l,k) )
    #if q is not None and np.abs(q).max()<=zmax:#
    if q is not None and np.abs(q).max()<=zmax:
        return {'l':l,'k':k,'z':list(q),'gcd':gcd}
    else:
        return {}    

def get_solution_from_list(lk,zmax=32):
    n=len(lk)
    l=lk[:n//2]
    k=lk[n//2:]
    return get_solution(l,k,zmax)


assert get_solution_from_list([1,2,1,-2])['z']==[1, 1, 1, -4, -4, 5]


# Prepare running

# In[2]:


d=[{'n':6,'N':4000000,'max':11},
   {'n':7,'N':50000000,'max':15},
   {'n':8,'N':50000000,'max':10},
   {'n':9,'N':50000000,'max':10},
   {'n':10,'N':50000000,'max':10},
   {'n':11,'N':50000000,'max':10},
   {'n':12,'N':50000000,'max':10}]

n=7
dd=[dd for dd in d if dd.get('n')==n][0]

N=dd['N'] 
mm=n-2

Single=False


# ### Single-processing

# In[17]:


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

# In[19]:


import dask.array as da
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count


# In[ ]:


size_old=0
imax=10
i=0
df=pd.DataFrame()
Δ_size=1
while Δ_size>0:
    #axis parameter not yet implemented in dask: `da.unique` → https://stackoverflow.com/a/53389741/2268280
    ll=da.random.randint(1,dd['max']+1,(N,mm))*(-1)**da.random.randint(0,2,(N,mm))
    ll=ll.to_dask_dataframe().drop_duplicates().to_dask_array()

    s=time.time()
    ll=ll.compute()
    print('grid → ',time.time()-s,ll.shape)

    s=time.time()
    pool = Pool(cpu_count())
    sls = pool.map(get_solution_from_list,ll)
    pool.close()
    del ll

    sls=[d for d in sls if d]
    print('sols → ',time.time()-s,len(sls))

    #Unique solutions
    df=df.append(  sls,ignore_index=True    )  
    df.sort_values('gcd')
    df['zs']=df['z'].astype(str)
    df=df.drop_duplicates('zs').drop('zs',axis='columns').reset_index(drop=True)
    print('unique solutions → ',df.shape)
    Δ_size=df.shape[0]-size_old
    if Δ_size>0:
        size_old=df.shape[0]
    if i>imax:
        break

    i+=1


# 

# In[104]:


#i=0
i+=1
i


# In[99]:





# In[16]:


raise Exception('Appendix')


# Appendix

# ### Check RAM USAGE

# In[91]:


d=[{'n':6,'N':4000000,'max':11},
   {'n':7,'N':50000000,'max':12},
   {'n':8,'N':50000000,'max':10},
   {'n':9,'N':50000000,'max':10},
   {'n':10,'N':50000000,'max':10},
   {'n':11,'N':50000000,'max':10},
   {'n':12,'N':50000000,'max':10}]

n=7
dd=[dd for dd in d if dd.get('n')==n][0]

N=dd['N'] 
mm=n-2

Single=False


# In[92]:


for i in range(2):
    print(i)
    #axis parameter not yet implemented in dask: `da.unique` → https://stackoverflow.com/a/53389741/2268280
    ll=da.random.randint(1,dd['max']+1,(N,mm))*(-1)**da.random.randint(0,2,(N,mm))
    ll=ll.to_dask_dataframe().drop_duplicates().to_dask_array()

    s=time.time()
    ll=ll.compute()
    
    if i==0:    
        f=open('ll.npy','wb')
        np.save(f,ll)
        f.close()    

    if i==1:
        print('→',i)
        with open('ll.npy', 'rb') as f:
            ll_old = np.load(f)
        
        #WARNING: Not multiprocessing
        ll=pd.DataFrame(np.concatenate((ll,np.zeros((1,mm)).astype(int),ll_old))).drop_duplicates(keep=False).reset_index(drop=True)
        ll=ll[:ll[(ll[0]==0) &  (ll[mm-1]==0)].index[0]].values    
    

    print('grid → ',time.time()-s,ll.shape)
    


# ### Check number of silutions

# In[82]:


sl=pd.read_json('solutions.json')


# In[83]:


#sl['zs']=sl['solution'].astype(str)
#sl=sl.drop_duplicates('zs').drop('zs',axis='columns').reset_index(drop=True)


# In[96]:


sl=sl[sl['n']==7]
sl.shape


# In[85]:


(sl['l']+sl['k']).apply(lambda l:np.abs(l).max()).max()


# In[92]:


s=set()


# In[ ]:


sl.rename({'solution':'z'},axis='columns')


# In[ ]:


sl=sl[['z']].append(df[['z']]).reset_index(drop=True)
sl['zs']=sl['z'].astype(str)
sl=sl.drop_duplicates('zs',keep=False).drop('zs',axis='columns').reset_index(drop=True)
sl.shape


# In[ ]:


sl


# In[14]:


{ tuple(x) for x in sl['solution'].to_list() }.difference( { tuple(x) for x in df['z'].to_list() }  )


# In[15]:


{ tuple(x) for x in df['z'].to_list() }.difference( { tuple(x) for x in sl['solution'].to_list() }  )


# In[99]:


df['z'].apply(lambda l:np.abs(l).max()).max()


# In[ ]:




