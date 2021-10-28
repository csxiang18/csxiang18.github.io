---

title: Optimal Real Time Bidding for Display Advertising

date: 2021-4-11

categories: 论文

tags: RTB

---

# Optimal Real Time Bidding for Display Advertising

主要解决了在有budget constrait的约束条件下，DSP如何出价去maximize自己的KPI (CTR, CVR,...)

很经典的一个框架，解决的问题是为DSP设计一个竞价机制，要满足

	1. budget约束
	2. 最大化广告主的KPI

整体DSP的竞价过程如下图所示：

<img src="/images/paper/bid_flow_1.png" alt="bid_flow" style="zoom:80%;" />

假设有N个request，形式化整体问题为找到一个泛函$b$，满足
$$b = \mathop{\max}_b N\int \theta(x)w(b(\theta(x), x), x) p(x)dx \\
\text{s.t.} \quad N \int b(\theta(x), x) w(b(\theta(x), x), x) p(x)dx \le B
$$
$b(\theta(x), x)$是一个竞价函数，取决于request $x$和$\theta(x)$，$\theta(x)$是如果竞价成功所获得的相应的KPI。可以是CTR, CVR, ... $w(b)$是出价$b$在RTB中赢得该request的一个概率, $p(x)$是关于$x$的一个概率密度函数。

这里这篇文章做了两点假设

1. $b(\theta(x), x) = b(\theta(x))$
2. $w(b(\theta(x), x), x) = w(b(\theta(x)))$

也就是说一旦预测的KPI确定后，以CTR为例，竞价主要取决于CTR有多少，另外$w$也只取决于出的价钱多少，而不取决于这个request $x$。

那么上面那个式子可以简化为
$$
b = \mathop{\max}_b N\int \theta(x)w(b(\theta(x)) p(x)dx \\
\text{s.t.} \quad  N \int b(\theta(x)) w(b(\theta(x)) p(x)dx \le B$$
然后根据概率密度函数我们可以得到$p_\theta(\theta(x)) = \frac{p_x(x)}{||\nabla \theta(x)||} \quad (3)$  。所以上式可以变为在$\theta$上的积分，为什么要变为在$\theta$上的积分，我觉得主要还是依据于假设1。因为一旦request $x$过来我们估计完他的CTR之后，我们的竞价主要是依据这个CTR来出价的。依据(3)式，我们可以将目标变为

$$\begin{align}
b &=  N\int \theta(x)w(b(\theta(x)) p_\theta(\theta(x))||\nabla (\theta(x))||dx \\
&=N\int \theta(x)w(b(\theta(x)))p_\theta(\theta(x))d(\theta(x)) \\
& = N \int \theta w(b(\theta))p(\theta)d\theta
\end{align}$$
约束也是一样，所以整体目标变为
$$
b = \mathop{\max}_b N\int \theta w(b(\theta) p(\theta)d\theta \\
\text{s.t.} \quad  N \int b(\theta) w(b(\theta)) p(\theta)d\theta \le B
$$
带有约束问题的优化，引入朗格朗日乘子，优化为
$$
\mathcal{L}(b, \lambda) = \int \theta w(b(\theta))p(\theta)d\theta - \lambda (\int b(\theta) w(b(\theta)) p(\theta)d\theta - \frac{B}{N})
$$
根据欧拉朗格朗日公式$J(y) = \int _a ^b L(x, y, y^{'})$ ，见维基百科[欧拉-朗格朗日公式](https://zh.wikipedia.org/wiki/%E6%AD%90%E6%8B%89-%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E6%96%B9%E7%A8%8B) 。满足$\frac{\partial}{\partial x}\frac{\partial L}{\partial y^{'}} = \frac{\partial L}{\partial y}$。

在这里$\frac{\partial L }{\partial y^{'}} = 0$，所以根据欧拉-朗格朗日公司我们可以得到
$$
\theta p(\theta) \frac{\partial w(b(\theta))}{\partial b(\theta)} - \lambda p(\theta)(w(b(\theta)) + b(\theta)\frac{\partial w(b(\theta))}{\partial b(\theta)}) = 0 \\
\lambda w(b(\theta)) = (\theta - \lambda b(\theta))\frac{\partial w(b(\theta))}{\partial b(\theta)} \qquad (4)
$$
接下来就是如何设计在竞价中获胜的概率$w$和$b$的关系了。在论文中采用了两种形式，这两种形式都能推到出来最优$b$的解。第一种形式为
$$
w(b(\theta)) = \frac{b(\theta)}{c + b(\theta)}
$$
带入(4)，可以得到最优的$b$的形式为
$$
b = \sqrt{\frac{c}{\lambda}\theta + c^2} - c
$$
第二种形式为
$$
w(b(\theta)) = \frac{b^2(\theta)}{c^2 + b^2(\theta)}
$$
最优解为
$$
b = c((\frac{\theta + \sqrt{c^2\lambda^2 + \theta^2}}{c\lambda})^{\frac{1}{3}} - (\frac{c\lambda}{\theta+\sqrt{c^2\lambda^2 + \theta^2}})^{\frac{1}{3}})
$$
至于$\lambda$的设定，本篇文章使用了在训练集上调出一个最优的$\lambda$。

在作者提供的补充材料 [Optimal Real-Time Bidding Frameworks Discussion](https://arxiv.org/pdf/1602.01007.pdf) 里面，谈到了不同机制下的bid竞价机制应该怎么设计，在一价里面，也就是上面这篇文章中注意约束项里面是
$$
\int b(\theta)w(b(\theta))p(\theta)d\theta \le B
$$
但实际上如果竞价为$b$，在市场中DSP需要花费的是$c(b)$，如果是一价，则是$c(b) = b$，如果是二价则是$c(b) = \frac{\int _0 ^b zp(z)dz}{\int p(z)dz}$ 。

所以补充材料也探讨了在二价中最优出价为$b = \theta/\lambda$。
