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
from multiprocessing import cpu_count
import time


# In[2]:


def f(x):
    return x**2


# In[60]:


s=time.time()
N=10#000000 
result = [f(x) for x in range(N)]
print(time.time()-s)


# WARNING: 

# In[61]:


s=time.time()
pool = Pool(cpu_count())
result = pool.map(f,range(N))
pool.close()
print(time.time()-s)
LAZY=False


# In[62]:


if LAZY:
    s=time.time()
    pool = Pool(cpu_count())
    result=[]
    result=[ r for r in pool.imap(f,range(N),chunksize=N//cpu_count())]
    pool.close()
    print(time.time()-s,result[:10])


# In[63]:


if LAZY:
    result=[]
    s=time.time()
    pool = Pool(cpu_count())
    result=[r for r in pool.imap_unordered(f,range(N),chunksize=N//cpu_count())]
    pool.close()
    print(time.time()-s,result[:10])


# ![img](https://miro.medium.com/max/700/1*n8_M7_0O2Rp3TCuqLDeeHg.png)chunksize=

# ## Implementation.
# * Use the official module to find solutions 
# * filter the chiral ones  with a maximum integer of 32
# * Build a function suitable for multiprocessing

# ### Functions

# In[53]:


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


# ### Prepare running

# In[56]:


d=[{'n':6,'N':4000000,'max':11,'imax':0},
   {'n':7,'N':8000000,'max':13,'imax':0},
   {'n':8,'N':50000000,'max':15,'imax':10},
   {'n':9,'N':50000000,'max':10,'imax':40},
   {'n':10,'N':70000000,'max':10,'imax':100},
   {'n':11,'N':50000000,'max':10,'imax':100},
   {'n':12,'N':50000000,'max':10,'imax':100}]

n=7
dd=[dd for dd in d if dd.get('n')==n][0]

N=dd['N'] 
mm=n-2

Single=False


# ### Single-processing

# In[17]:


if Single:
    s=time.time()
    ll=np.random.randint(-dd['max'],dd['max']+1,(N,mm))
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

# In[55]:


import dask.array as da
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count


# In[58]:


UNORDERED=False
size_old=0
imax=dd['imax']
i=0
df=pd.DataFrame()
Δ_size=1
while Δ_size>0:
    #axis parameter not yet implemented in dask: `da.unique` → https://stackoverflow.com/a/53389741/2268280
    ll=da.random.randint(-dd['max'],dd['max']+1,(N,mm))
    ll=ll.to_dask_dataframe().drop_duplicates().to_dask_array()

    s=time.time()
    #See: https://docs.dask.org/en/stable/scheduler-overview.html#configuring-the-schedulers
    #.compute(num_workers=XX) → defaults to number of cores
    ll=ll.compute()
    print('grid → ',time.time()-s,ll.shape)

    s=time.time()
    pool = Pool(cpu_count())
    if not UNORDERED:
        sls = pool.map(get_solution_from_list,ll)
    else:
        sls=[r for r in pool.imap_unordered(get_solution_from_list,ll,chunksize=len(ll)//cpu_count())]
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
    if i>=imax:
        break

    i+=1


# In[69]:


df.to_json(f'solution_{n}.json',orient='records')


# In[16]:


raise Exception('Appendix')


# ## Appendix

# ### Check RAM USAGE

# In[57]:


d=[{'n':6,'N':4000000,'max':11}, #'N':4000000,'max':11
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


# In[60]:


for i in range(2):
    print(i)
    #axis parameter not yet implemented in dask: `da.unique` → https://stackoverflow.com/a/53389741/2268280
    ll=da.random.randint(-dd['max'],dd['max']+1,(N,mm))
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
        ll=pd.DataFrame(np.concatenate((ll,
                                        (np.ones((1,mm))*(dd['max']+1)).astype(int), #introduce separator
                                        ll_old)
                                      )).drop_duplicates(keep=False).reset_index(drop=True)
        ll=ll[:ll[(ll[0]==dd['max']+1) &  (ll[mm-1]==dd['max']+1)].index[0]].values    
    

    print('grid → ',time.time()-s,ll.shape)
    


# In[36]:


ll[[0 in x for x in ll]]


# In[28]:





# ### Check number of solutions
# From: https://doi.org/10.5281/zenodo.5526707

# In[26]:


df=pd.read_json('solution_8.json')


# In[27]:


df.shape


# In[28]:


import pandas as pd
import numpy as np


# In[33]:


sl=pd.read_json('solutions.json')


# In[34]:


#sl['zs']=sl['solution'].astype(str)
#sl=sl.drop_duplicates('zs').drop('zs',axis='columns').reset_index(drop=True)


# In[35]:


sl=sl[sl['n']==10]
sl.shape


# In[ ]:





# In[ ]:


5569


# In[14]:


(sl['l']+sl['k']).apply(lambda l:np.abs(l).max()).max()


# In[22]:


sl=sl.rename({'solution':'z'},axis='columns')


# In[25]:


sl=pd.concat((sl,df)).reset_index(drop=True)
sl['zs']=sl['z'].astype(str)
sl=sl.drop_duplicates('zs',keep=False).drop('zs',axis='columns').reset_index(drop=True)
sl.shape


# In[34]:


df[:3]


# In[40]:


#sl.to_json('missing_8.json',orient='records')


# In[44]:


sl


# In[54]:


sl[sl['solution'].astype(str)=='[19, -22, -22, 27, 27, -28, 31, -32]']


# In[36]:


sl['l'].iloc[3]


# In[37]:


get_solution_from_list( (sl['l']+sl['k']).iloc[3] )


# In[32]:


get_solution(sl['l'].iloc[0],sl['k'].iloc[0])


# In[33]:


z(sl['l'].iloc[0],sl['k'].iloc[0])


# In[29]:


#{ tuple(x) for x in sl['z'].to_list() }.difference( { tuple(x) for x in df['z'].to_list() }  )


# In[30]:


#{ tuple(x) for x in df['z'].to_list() }.difference( { tuple(x) for x in sl['solution'].to_list() }  )


# In[31]:


df['z'].apply(lambda l:np.abs(l).max()).max()


# In[37]:


[8, 9, 11, -17, -17, -17, 23]==[8, 9, 11, -17, -17, -17, 23]


# In[ ]:


import numpy as np


# In[15]:


np.unique( np.random.randint(-2,3,(200,2)),axis=0 )


# In[5]:


import numpy as np


# In[6]:


import pandas as pd


# In[7]:


df=pd.read_json('/home/restrepo/Downloads/solution_6.json')


# In[25]:


df[df[6].apply(lambda l: np.abs(l).max())<=32].shape


# In[37]:


p=Pool()


# In[38]:


get_ipython().run_line_magic('pinfo', 'p.map_async')


# In[ ]:




