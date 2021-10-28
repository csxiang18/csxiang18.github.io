---
title: Forecasting
date: 2020-8-06
categories: 论文
tags: Forecast
---
# Time-series Forecasting Papers

## Forecast High Dimensional Data

要解决的问题：基于campaign的target的attribute来预测某个campaign未来的流量。但因为attribute的组合太多，事先是没办法进行预测的。论文解决了如何在线预测各种attribute组合的问题。

### Method:
有三种方法，FIM，PIM，SJM
首先需要构造一系列的base query: $Q_k$，对每个$Q_k$而言，需要预测未来时间的流量，论文里面采用SARIMA模型，这部分是需要离线去做的。在线来的某个query q，需要找到其对应的最小$Q_k$，找到最小$Q_k$后，q的预测流量为$B(Q_k|T)*R(q|Q_k)$，其中$B(Q_k|T)$代表对于base query:$Q_k$在时间段T里预测的流量。

#### FIM(Fully independence Model)

在计算$R(q|Q_k)$的时候，假设q所对应的attribute为$a_1, a_2, \cdots, a_n$，那么

$$
R(q|Q_k) = \prod_{a_i \in q} R(A_i=a_i|Q_k)
$$

对于计算$R(A_i=a_i|Q_k)$，首先获取在时间$t$之前的历史数据集$P_t$ 。那么
$$
R(A_i=a_i|Q_k)=\frac{|\{p \in Q_k \cap P_t \text{   and   } a_i \text{satisfy } p\}|}{|Q_k\cap P_t|}
$$

#### PIM(Partwise independence Model)

对于某些attribute而言，它们其实是相关的，比如年龄和收入，因为完全假设独立并不成立，PIM在计算$R(q|Q_k)$的时候考虑到了attribute之间的相关性，会选用一些相关的attribute进行合并
$$
R(A_i=a_i,A_j=a_j)=\frac{|\{p \in Q_k \cap P_t \text{   and   } a_i, a_j \text{satisfy } p\}|}{|Q_k\cap P_t|}
$$

#### SJM(Partwise independence Model)

不对attribute的独立性做任何假设，首先会有一些采样点$S$，实验里有20M，对于每个$Q_k$而言，离线计算$|Q_k\cap S|$大小。在线query q的时候，使用bitmap的方式，计算S内有多少个点满足q。$n = |S \cap q|$，那么$R(q|Q_k) = \frac{n}{|Q_k \cap S|}$

### System Arch
![System Arch](/images/paper/forecast_high_1.png)
