# Supervised Learning with XGBoost

## Recall the base math

$$
F_M(x) = F_0(x) + \eta \sum_{m=1}^M f_m(x)
$$

- $M$: # trees
- $f_m$: score from tree $m$, also $w_m$
- $F_0$: init prediction
- $\eta$: learning rate

$$
z_i = F(x_i) \rightarrow p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

Then BCE loss:

$$
L_i = -y_i\log p_i - (1 - y_i)\log(1 - p_i)
$$

$$
g_i = \frac{\partial L}{\partial F(x_i)} = p_i - y_i \\
h_i = \frac{\partial^2 L}{\partial F^2(x_i)} = p_i(1 - p_i)
$$

## Init $F_0$

Suppose actual prob is

$$
p = \frac{1}{n}\sum_i y_i
$$

Now we need a formula of $F_0$ based on $p$

Since

$$
\sigma \left(\log \frac{p}{1 - p} \right) = p
$$

then set:

$$
F_0 = \log\frac{p}{1 - p}
$$