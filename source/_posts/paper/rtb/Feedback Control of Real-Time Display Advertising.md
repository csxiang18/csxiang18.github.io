---
title: Feedback Control of Real-Time Display Advertising
date: 2021-10-29
categories: 论文
tags: RTB, RL
---

一篇基于反馈控制系统来控制DSP的竞价策略。解决的问题是需要在一定的budget约束条件下，通过调整我的竞价，达到控制广告主的一些KPI的目的，比如本文里面的eCPC，AWR或者CTR等。

最早的基于request的竞价策略为
$$
b(t) = b_0 \frac{\theta_t}{\theta_0} \qquad (1)
$$
$\theta_t$是当前的utility，比如CTR, expected revenue等等。$b_0$是调整过后的基准竞价，$\theta_0$是一个在当前target条件下一个平均的utility，比如平均的CTR。

文章解决的问题是想利用反馈控制来调整我的出价，以机制（1）的竞价方式为基准，达到控制某一些KPI的目的。

<img src="/images/paper/feedback_1.png" alt="bid arch" style="zoom:50%;" />

其中Bid Calculator就是机制(1)，得到一个bid price $b(t)$，然后输入到一个新的叫Actuator里面去。Actuator就是文中利用反馈控制信号来条件$b(t)$得到一个新的$b_a(t)$。控制机制为
$$
b_a(t) = b(t) \exp (\phi(t))
$$
其中$\phi(t)$是反馈系统所产生的反馈控制信号。作者也试过用线性模型，如$b_a(t) = b(t)(1 + \phi(t))$来调整出价，但是效果不好，所以还是用了这种指数形式。

接下来就是怎么得到这个反馈控制信号了$\phi(t)$。文章使用了两种控制器，一种是PID控制器，另外一种是水位控制器。具体形式如下，对了提一嘴，控制器的目标是希望控制一个动态系统能够输出一个理想的控制信号的。所以控制器都会带一个$x_r(t)$，表示那一个时刻控制器想要达到的理想状态。本文里面以eCPC为例。

如果是PID控制器，首先会计算一个误差项$e(t_k) = x_r(t_k) - x(t_k)$。那么产生的控制信号$\phi(t)$就是
$$
\phi(t_{k+1}) = \lambda e(t_k) + \lambda_I \sum_{j=1}^k e(t_j) \Delta t_j + \lambda_D \frac{\Delta e(t_k)}{\Delta t_k}
$$
如果是水位控制器，那么产生的控制信号是
$$
\phi(t_{k+1}) = \phi(t_k) + \gamma (x_r(t_k) - x(t_k))
$$
接下来对于控制器而言，就是怎么设置系统需要达到的理想的eCPC $x_r(t)$了。在这里，广告主没有设置一个它需要达到的eCPC来作为参考，那么作为DSP而言，一个设置eCPC的方式是使得广告的点击量尽可能的大。可以以此为基准来求出我想要控制的eCPC是多少。假设对于DSP集成的第i个ADX，我的eCPC为$\zeta_i$。$c_i(\zeta_i)$是在该ADX在该eCPC下的一个点击数，所以对于DSP而言，理想的eCPC应该满足以下的性质
$$
\max_{\zeta_i,\cdots,\zeta_n} \sum_i c_i(\zeta_i) \\
s.t. \qquad \sum_i c_i(\zeta_i) \zeta_i = B
$$
B是对应的budget。用朗格朗日法以及对$c_i(\zeta_i)$做一些形式上的假定，我们可以求得，对每个ADX而言，最优的$\zeta_i$。具体细节可以看论文，这就是我们的reference value。通过得到该reference value，就可以得到对每个ADX而言，我的出价应该是多少。

