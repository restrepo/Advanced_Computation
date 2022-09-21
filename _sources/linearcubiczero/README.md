# Find interger lists with linear and cubic add to zero
The program we will search non vectorlike solutions with a fix number of integers, $z_i$, starting with $n=5$. 

For that, it must use two lists of integers, $l$ and $k$, which are build from a single list of $n-2$ arbitrary integers, to be generated either from a grid or from a random scan.

The maximum in absolute value of the list $z$, $z_{\rm  max}$, can be obtained from a much lesser maximum in absolute value of the lists $l$ and $k$:
$m={\rm max}(|l|,|k|).$



For a given $z_{\rm  max}$, we can find the optimal $m$ by increasing  it in one unit and checking that not extra solutions are found.

In the algorithm, we will generate a list of $n-2$-integer lists, ${\bf L}$, with elements between $-m$ and $m$. Therefore, the list ${\bf L}$ will contain the following number of $n-2$ lists:

$$
N_{\rm unique}=(2m+1)^{(n-2)}\,.
$$

From each one, we will extract the $l$ and $k$ input list of dimensions ${\rm dim}\\, l=({\rm dim}\\,  z)//2$ and ${\rm dim}\\,  k=n-2-{\rm dim}\\, l$. 

Because for large $n$ this list will no fit in RAM, we choose to generate $i$-lists ${\bf L}$ from a random scan, each one with $N$ lists of $n-2$ integers from $-m$ to $m$. The process will be repeated from $i=0$ until $i=i_{\rm max}$. Note that $N\gg N_{\rm unique}$ to guarantee that the full grid is obtained when $i_{\rm max}=0$


## USAGE
```bash
linearcubiczero --N=500000 --m=9 --zmax=30 --imax=0 --output_name='solution' 6
```
It will generate a file with the name `solution_6.json` with the solutions for $n=6$ and $z_{\rm max}=30$.