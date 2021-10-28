---
title: Real-Time Bidding by Reinforcement Learning in Display Advertising

date: 2021-4-19

categories: 论文

tags: RTB, RL
---
# Real-Time Bidding by Reinforcement Learning in Display Advertising

很早的一篇用RL来解决advertiser竞价的问题。把advertiser的竞价行为看成是一轮一轮的游戏。在每一轮游戏过程中总共有T次竞价，advertiser的budget的上限为B，每次竞价中opportunity的特征为$x$.

首先对state的定义为$S = (t, b, x_t)$，表示当前剩余轮数，剩余budget，和当前的request特征。这时候agent要做的决策就是决定一个价格$a$，在这篇论文里$a$是一个整数。另外还有两个量$m(\delta, x)$和$\theta(x)$，$m(\delta,x)$表示和该广告主竞争的其余广告主在特征$x$下出价$\delta$的一个概率分布。如果该广告主出价为$a \ge \delta$，那么在这次竞价中该广告主就会获胜，并且budget会消耗$\delta$。$\theta(x)$是该次机会中的CTR。是做为RL中的reward。那么根据以上定义，在这个竞价MDP的过程中，转移概率和奖励函数就可以如下定义：
$$
\begin{align}
& \mu(a, (t,b,x_t), (t-1, b-\delta, x_{t-1})) = p_x(x_{t-1})m(\delta, x_t) \\
& \mu(a, (t,b,x_t), (t-1, b, x_{t-1})) = p_x(x_{t-1})\sum_{\delta=a+1}^\infty m(\delta, x_t) \\
& r(a, (t,b,x_t), (t-1, b-\delta, x_{t-1})) = \theta(x_t) \\
& r(a, (t,b,x_t), (t-1, b, x_{t-1})) = 0
\end{align}
$$
其中$\delta \in \{0,\cdots, a\}$.有了转移概率和奖励函数，就用DP的方法去求解value-function  $V(t,b,x)$。在论文里面作者考虑一个更通用的场景，也就是在没有观察到特征$x$的情况下，求解value-function  $V(t,b) = \int V(t,b,x)dx$。根据bellman方程，当$\gamma=1$有
$$
V^\pi(s) = \sum_{s^{'}}\mu(\pi(s),s, s^{'})(r(\pi(s), s, s^{'}) + V^\pi(s^{'}))
$$
那么根据上述MDP过程$V(t, b, x)$可以由如下$V(t,b)$表示：
$$
\begin{align}
V(t,b, x) &= \mathop{\max}_{0 \le a \le b} \Big{\{} \sum_{\delta=0}^a \int m(\delta, x)p_x(x_{t-1}) (\theta(x) + V(t-1, b-\delta, x_{t-1}))dx_{t-1} \\ & + \sum_{\delta=a+1}^\infty \int m(\delta,x)p_x(x_{t-1}) V(t-1, b,x_{t-1})dx_{t-1} \Big{\}}
\\ &= \mathop{\max}_{0 \le a\le b}\Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta)) + \sum_{\delta=a+1}^\infty m(\delta, x)V(t-1, b) \Big{\}}  \qquad (1)
\end{align}
$$
所以最优的action $a(t, b, x)$有
$$
\begin{align}
a(t, b, x) &= \mathop{\text{argmax}}_{0\le a \le b} \Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta)) \\ &+ \sum_{\delta=a+1}^\infty m(\delta, x)V(t-1, b) \Big{\}}  \qquad (2)
\end{align}
$$
其中包含三个量$m(\delta, x), \theta(x), V(t-1, *)$。首先根据(1)式与$V(t,b) = \int V(t,b,x)dx$来求解$V(t-1, *)$
$$
\begin{align}
V(t, b) &= \int p(x) \mathop{\max}_{0 \le a\le b}\Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta)) \\ &+ \sum_{\delta=a+1}^\infty m(\delta, x)V(t-1, b) \Big{\}} dx \\ & = \mathop{\max}_{0 \le a \le b}\Big{\{} \sum_{\delta=0}^a \int p(x) m(\delta, x)\theta(x) dx \\ &+ \sum_{\delta=0}^a V(t-1, b-\delta)\int p(x) m(\delta,x) dx + V(t-1,b) \sum_{\delta=a+1}^\infty \int p(x) m(\delta,x)dx \Big{\}} \\ &= \mathop{\max}_{0 \le a\le b}\Big{\{} \sum_{\delta=0}^a \int p(x) m(\delta, x)\theta(x) dx \\ &+ \sum_{\delta=0}^a V(t-1, b-\delta) m(\delta) + V(t-1, b) \sum_{\delta=a+1}^\infty m(\delta) \Big{\}}
\end{align}
$$
第一项$\int p(x) m(\delta, x)\theta(x) dx$，论文假设有$m(\delta, x) = x$，即市场上bid price的一个概率分布是和特征$x$没有关系的。这是之前一篇文章所假设的。所有$\int p(x) m(\delta, x)\theta(x) dx = m(\delta)\theta_{\text{avg}} $ 。所以最终$V(t,b)$的计算方式为
$$
V(t, b) \approx \mathop{\max}_{0 \le a\le b}\Big{\{} \sum_{\delta=0}^a m(\delta)\theta_{\text{avg}} + \sum_{\delta=0}^a V(t-1, b-\delta) m(\delta) + V(t-1, b) \sum_{\delta=a+1}^\infty m(\delta) \Big{\}}
$$
在计算得到$V(t,b)$之后，那么对于action而言有(2)式，另外注意到$\sum_{\delta=0} ^\infty m(\delta, x) = 1$，所以(2)可以化简成
$$
\begin{align}
a(t, b, x) &= \mathop{\text{argmax}}_{0\le a \le b} \Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta)) \\ &+ \sum_{\delta=a+1}^\infty m(\delta, x)V(t-1, b) \Big{\}}  \\ &= \mathop{\text{argmax}}_{0\le a \le b} \Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta)) \\&+ V(t-1, b) -  \sum_{\delta=0}^a m(\delta, x)V(t-1, b) \Big{\}}
\end{align}
$$
注意到$V(t-1,b)$与a无关，所以
$$
\begin{align}
a(t, b, x)  &= \mathop{\text{argmax}}_{0\le a \le b} \Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta)) \\ & -  \sum_{\delta=0}^a m(\delta, x)V(t-1, b) \Big{\}} \\ &= \mathop{\text{argmax}}_{0\le a \le b} \Big{\{} \sum_{\delta=0}^a  m(\delta,x)(\theta(x) + V(t-1, b-\delta) - V(t-1, b)) \\ &= \mathop{\text{argmax}}_{0\le a \le b} \Big{\{} \sum_{\delta=0}^a m(\delta, x) g(\delta) \Big{\}}
\end{align}
$$
分析一下$g(\delta) = \theta(x) + V(t-1, b-\delta) - V(t-1, b)$的性质，首先有$V(t-1, b - \delta)$是关于$\delta$的一个递减函数，那么$g(\delta)$也是一个关于$\delta$的递减函数，所以如果有$g(b) \ge 0$，那么$a(t,b,x) = b$。如果$g(b) \le 0$，因为$g(0) = \theta(x) \ge 0$，所以一定存在一个数$A$满足$g(A) \ge 0, g(A+ 1)\le0$。那么此时就有$a(t, b, x) = A$。所以完整的算法如下图所示：

<img src="/images/paper/rtb_rl_1.png" alt="bid_flow" style="zoom:40%;" />


另外作者还讨论了当$m(\delta, x) \neq m(\delta)$的时候，因为ADX 关于bid price的分布是和$x$有关的。但是对于一些特定的feature来说可能是无关的，比如说都在同一些segment下面，所以可以根据$x$的特征来进行一个划分，比如说不同的demo等等. 全集$X = \sqcup_i x_i$，那么之前
$$
\int p(x) m(\delta, x)\theta(x) dx = \sum_i \int_{x_i} p(x_i)m(\delta, x_i)\theta(x_i) dx_i \\ \approx \sum_im_i(\delta) \int_{x_i} p(x_i)\theta(x_i)dx_i = \sum_i m_i(\delta)(\theta_{avg})_i p(x \in x_i)
$$
另外如果表$V(t, b)$太大，整体时间复杂度为$O(TB)$。可能导致时间/空间复杂度太高，论文中提出了可以在小数据集上先计算$V(t,b)$，然后再用一个参数化的模型用RMSE来进行拟合。
