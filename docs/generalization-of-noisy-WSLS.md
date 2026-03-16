## The original definition for two actions
The noisy version of the win-stay-lose-shift (WSLS) model for the "two-bandit" case (where there are only two actions) is defined in Wilson & Collins (2019; p. 6 of 33) as follows. 
The probability of chosing option $k$ at trial $t$ is: 

$$
p_t^k =

\begin{cases}

1 - \frac{\varepsilon}{2} 
& 
\text{if } (c_{t-1} = k \text{ and } r_{t-1} = 1) \text{ OR } (c_{t-1} \neq k \text{ and } r_{t-1} = 0) 

\\

\frac{\varepsilon}{2}     
& 
\text{if } (c_{t-1} \neq k \text{ and } r_{t-1} = 1) \text{ OR } (c_{t-1} = k \text{ and } r_{t-1} = 0)

\end{cases}

$$

where $c_t$ is the action at trial $t$, $r_t$ is the reward at trial $t$. The model has one free parameter, the overall level of randomness or noisyness of action selection, $\varepsilon \in [0, 1]$. 

## The generalized version

### The properties of the original model
1.  The action selection is uniformly random at probability $\varepsilon$. 
1.  Otherwise, win-stay-lose-shift. 
1.  When $\varepsilon = 0.0$, it is deterministically win-stay-lose-shift. 
1.  When $\varepsilon = 1.0$, the action is chosen totally uniformly randomly (action 1 and 2 are chosen at probability 0.5 and 0.5, respectively), independent of the previous action and reward. 
1.  And of course, $p$ is probability: $p \geq 0$ and $\sum p = 1$. 

We preserve these properties above. 

### Generalized definition

For $K$ actions ($k=1, 2, ..., K$), the probability of choosing action $k$ is defined as follows. 
When $r_{t-1} = 1$: 

$$
p_t^k =

\begin{cases}

1 - \frac{K-1}{K}\varepsilon
& 
\text{if } c_{t-1} = k

\\

\frac{\varepsilon}{K}
& 
\text{if } c_{t-1} \neq k
\end{cases}
$$

and when $r_{t-1} = 0$: 

$$
p_t^k = 

\begin{cases}

\frac{\varepsilon}{K}
& 
\text{if } c_{t-1} = k

\\

\frac{1}{K-1} - \frac{\varepsilon}{K(K-1)}
& 
\text{if } c_{t-1} \neq k

\end{cases}
$$

The five properties are preserved as follows: 

1. The random part in the action selection probabilities is $\varepsilon$ 
1. See below. 
1. When $\varepsilon=0$, with $r_{t-1}=1$, $c_{t-1} = c_t$. With $r_{t-1}=0$, the probability of $c_{t-1} = c_t$ is 0 and of all other ($K-1$) actions is $\frac{1}{K-1}$. 
1. When $\varepsilon=1$, with $r_{t-1}=1$, $p_t^k = \frac{1}{K}$. With $r_{t-1}=0$, the probability of $c_{t-1} = c_t$ is 0 and of all other ($K-1$) actions is $\frac{1}{K-1}$. 
1. By $K>1$ and $\varepsilon \geq 0$. 


<!-- 
$$
p_t^k =
\begin{cases}
1 - \frac{K-1}{K}\varepsilon
& 
\text{if } (c_{t-1} = k \text{ and } r_{t-1} = 1)
\\
\frac{\varepsilon}{K}
& 
\text{if } (c_{t-1} \neq k \text{ and } r_{t-1} = 1)
\\
\frac{\varepsilon}{K}
& 
\text{if } (c_{t-1} = k \text{ and } r_{t-1} = 0)
\\
\frac{1}{K-1} - \frac{\varepsilon}{K(K-1)}
& 
\text{if } (c_{t-1} \neq k \text{ and } r_{t-1} = 0)
\end{cases}
$$ 
-->

### Implementation 

See [[]] in MultiBandits.jl. 

## Reference

Wilson, R. C., & Collins, A. G. (2019). Ten simple rules for the computational modeling of behavioral data. eLife, 8, e49547. https://doi.org/10.7554/eLife.49547
