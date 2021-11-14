---
title: Deep Lanscape Forecasting for Real-Time Bidding
date: 2021-11-14 12:00:00
tags: [RTB, Survival Analysis]
categories: 论文
---
 
本文解决的问题是为advertiser预测一个一个竞价流量的价值$z$，在二价的场景下面称作market price。这个问题主要有两个难点。

1. 对于在RTB市场fail的bid，只能知道自己的出价$b$低于market price $z$，这类被称作censored data。
2. 通常会假设$z$是服从于某个假定的分布，但在实际应用中通过不了类似的假设检验

论文作者使用了两种方式来解决这个问题。一是用survival analysis来解决censored data。二是用一个RNN Model来解决z的分布的问题。假设市场价格$z \sim p(z)$，在出价$b$的情况下，成功bid的概率是
$$
W(b) = Pr(z < b) = \int_0^b p(z)d(z)
$$
失败bid的概率则是
$$
S(b) = Pr(z \ge b) = \int_b^\infty p(z)d(z) = 1 - W(b)
$$
分别对应生存模型里面的death和survival概率。在这里$z$是一个连续的变量，我们可以把它拆解成为一个离散的形式。因为竞价$b$都有一个最小的单位。假设 
$$ 0 < b_{1} < b_{2} < \cdots < b_n  $$
$b_n$ 是一个bid的上限。 
$$b_i = b_{i-1} + 1, V_j = (b_j, b_{j+1}] $$

那么上面连续的形式可以变成一个离散的形式。
$$
W(b_l) = Pr(z < b_l) = \sum_{j<l}Pr(z \in V_j)
$$

$$
S(b_l) = Pr(z \ge b_l) = \sum_{j \ge l }Pr(z\in V_j)
$$

根据上面的定义，可以得到$$p_j=Pr(z \in V_j) = S(b_{j+1}) - S(b_j)$$

论文里接下来定义了一个条件概率
$$
h_l = Pr(z \in V_l | z\ge b_{l-1}) = \frac{p_l}{S(b_{l-1})}
$$
接下来一个重点是怎么建模这些概率了，作者使用了一个RNN来建模这个条件概率$h_l$
$$
h_l = Pr(z \in V_l | z \ge b_{l-1}, \mathbf{x}, \theta) = f_\theta(\mathbf{x},\theta|\mathbf{r}_{l-1})
$$
$\mathbf{r}_{l-1}$是上一个时刻RNN的隐藏状态。那么根据$h_l$的定义，上面的$S, W$都可以写成$h_l$的形式。即
$$
S(b_l) = Pr(z \ge b_l) = Pr(z \notin V_1, z\notin V_2, \cdots, z\notin V_{l-1}) \\
= Pr(z \notin V_1) \times Pr(z \notin V_2|z\notin V_1) \times \cdots Pr(z \notin V_{l-1}| z\notin V_1,\cdots, z\notin V_{l-2}) \\ 
= \prod_{j\le{l-1}} (1 - Pr(z\in V_j | z \ge b_{j-1})) = \prod_{j\le l-1} (1 - h_j)
$$

$$
W(b_l) = 1 - \prod_{j\le {l-1}} (1 - h_j)
$$



整个架构如下图所示：

<img src="/images/paper/rtb_deep_forecast_1.png" alt="rnn"  style="zoom:100%;" />


另外根据上面定义有
$$Pr(z\in V_l|\mathbf{x}) = h_l \times S(b_{l-1})$$

有了概率形式表示之后，就可以根据最大似然来训练整个网络模型。损失函数是作者的另外一个创新吧。一个是考虑了PDF，另外一个是考虑了CDF来训练模型。首先我们得到的数据分为两种，一种是赢了这次竞价，所以知道对应的market price $z$，另外是不知道，只知道我们对应的竞价$b$，$z \ge b$

一是如果我们观测到这个bid赢了之后，知道对应的market price，所以第一个损失函数是基于PDF
$$
L_1 = -\log  \prod _{x \in D_{win}}Pr(z \in V_l | \mathbf{x})
$$
第二是如果赢了，我们要最大似然获胜概率$W(b_l)$，另外如果输了要最大似然失败概率$S(b_l)$.
$$
L_2 = -\log \prod_{x \in D_{win}} W(b|x) - \log \prod _{x\in D_{loss}} S(b|x)
$$
最终的loss为$L = \alpha L_1 + (1 - \alpha) L_2$



思考：

为什么使用RNN？更多的我觉得是不是建模成条件概率之后更容易训练一点？相比于完全预测$Pr(z\in V_j)$给一个条件形式的概率更容易让网络学习。


