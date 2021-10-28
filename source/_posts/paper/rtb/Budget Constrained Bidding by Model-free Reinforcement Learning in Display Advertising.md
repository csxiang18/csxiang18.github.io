---
title: Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising

date: 2021-6-2

categories: 论文

tags: RTB, RL
---

​	DSP/广告主再RTB采取的出价策略通常是在一定的budget限制条件下，得到最优的价值。所以可以形式化表示为
$$
\max \sum_{i=1,\cdots,N} x_iv_i \\
s.t. \sum_{i=1}^n x_ic_i \le B
$$
$v_i$是当前流量的价值，通常可以用CTR/CVR去衡量。在上一篇RTB的出价策略中，见[竞价策略一](https://zhuanlan.zhihu.com/p/366170860)。我们可以知道，在二价的场景下，最优的出价策略为$b_i =\frac{v_i}{\lambda}$。，$\lambda$则是设置的超参数，之前论文的作者使用了在训练集上得出了最优的$\lambda^*$。但是在线上实时竞价的过程中，RTB市场是高度动态和变化的，所以设置一个最优的$\lambda^*$是很难的，本篇文章就使用了RL的方式来控制$\lambda$。

​	首先对于RL问题而言，建模成一个MDP过程。首先需要对如下几个变量进行定义。

1. 整个Episode长度为$T$（论文里面$t$和$t + 1$之间的时间间隔为15分钟。所以可以认为在这15分钟以内$\lambda$是一个常量
2. State：当前的步数$t$，剩余budget，budget消耗速率，$t$和$t-1$之间竞价成功率，平均CPM以及上一轮的$r_{t-1}$
3. Action: $\lambda_t = \lambda_{t-1} \times (1 + \beta_\alpha)$，$\beta_\alpha$是当前所采取的Action，$\lambda$使用增量调整的方式
4. Transition: 我们采用model-free的方式，不需要对Transition概率进行建模
5. reward: $r_t = \sum_{i \in I_t}x_iv_i$ ,$I_t$表示那一段时间内的流量
6. discount factor: $\gamma$，在本篇论文里，使用了$\gamma = 1$，因为需要优化总体的reward，而不论这里的reward是最近得到的，还是在比较靠后的step中得到

有了以上的MDP建模之后，照理来说就可以使用深度学习经典的RL框架去进行学习。但是在本篇论文里面，作者提到了原始的reward可能存在的两个问题：

1. reward没有考虑到budget的限制。也许可以采用CMDP框架的方式，比如引入对budget的惩罚项$r_t' = r_t + \alpha c_t$
2. 用原始reward学习到的策略更多的是一种贪心的策略，并且缺乏有效的exploration。

所以作者提出了一个新的reward

1. reward自然而然的嵌入了约束
2. 容易实现
3. 不局限于当前有budget限制bidding的RL场景

新的reward形式如下:
$$
r(s,a) = \mathop{max}_{e\in E(s,a)}\sum_{t=1}^T r_t^{(e)}
$$
$E(s,a)$表示了在场景$s$采用了动作$a$的所有的episodes。然后对于该episode，所有t步的reward的和作为当前$s, a$的reward。我的理解是有点归因的意思，就是整个agent在这个episode所产生的所有reward可以归因到该$s,a$。作者在论文里面也论述了对于优化新的reward所产生的最优policy $\pi_r^*$也一定是原reward的最优policy $\pi^*$。

一个自然而然的问题是如何得到新的reward$r(s,a)$。因为$s,a$都是连续一些特征，所以没办法用打表的方式记录下来，因此论文里面，作者使用了一个reward-net去训练得到新的reward $r(s,a)$。训练reward的算法如下：

<img src="/images/paper/rtb_rl_2.png" alt="bid_flow" style="zoom:60%;" />


同之前MDP一样，在新的reward，可以使用RL框架进行训练了，作者使用了DQN作为RL的训练方法。其中一个小技巧是采用了自适应的$\epsilon$，就是在$\epsilon$采样的时候对$\epsilon$进行一个退火吧。DQN结合reward-net的训练算法如下：

<img src="/images/paper/rtb_rl_3.png" alt="bid_flow" style="zoom:80%;" />


整体在线上serve的时候，使用$\frac{v_i}{\lambda_t}$进行serve。
