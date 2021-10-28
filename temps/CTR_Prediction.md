# CTR Prediction



##  Predicting Clicks: Estimating the Click-Through Rate for New Ads

论文大概是第一篇使用LR来进行CTR预估的文章

模型很简单$f(x) = sigmoid(x)$ ，主要是对各种特征进行了探索

权重初始化：$\mu$为0的Gaussian, $\sigma$用validation选

数据集：过滤掉了view数少于100的广告，因为这部分数据集可能有噪声。文章实验表明阈值越高越好。

特征：

对于每个特征$f_i$都添加了$\log(f_i + 1), f_i^2$，然后进行均值方差归一化。

Term CTR:因为预测某个广告的CTR，与之相关的广告的CTR也是会有关的。

其中一个特征$f(ad) = \frac{\alpha\overline{CTR} + N(ad_{term}CTR(ad_{term}))}{\alpha + N(ad_{term})}$。$\overline{CTR}$是训练集上的CTR，而$CTR(ad_{term})$是和该广告相关的CTR

Related Term CTR：找到和该广告中词语有关的广告，计算平均的CTR，并把相关广告的数量也加入feature中

广告质量feature:抽取了81个feature外加10000个常用词作为unigram feature。

订单feature:广告是和订单相关的，抽取订单的一些feature，如订单中term的词数，用分类器对term进行分类来获得对应的熵等，称作order specific feature。

外部数据feature: 搜索引擎中与该ad相关的有多少个页面，ad term被搜索的频率等

## Practical Lessons from Predicting Clicks on Ads at Facebook

Facebook的一篇论文，使用了LR+Decision Tree的方式来构造

![image-20191225140301655](/Users/changsheng.xiang/Learn/Tutorials/Notes/PersonalPlan/notes/paper/images/LR_DR_1.png)



输入进入LR的特征。如上图所示：在决策树里面，有两颗子树。第一颗子树有3个叶子节点，第二颗子树有2个叶子节点。最终决策树形成的feature有5维。前3维为第一颗子树把x分到了第几个节点，如第一个叶子节点。后两维为第二颗子树把x分到了第几个叶子节点，如第2个叶子节点。那么最终形成的feature为[1, 0, 0, 0, 1]



衡量的Metric:

$$
NE = \frac{-\frac{1}{N}\sum_{i=1}^N(\frac{1+y_i}{2}\log (p_i)+\frac{1-y_i}{2}\log (1 - p_i))}{-(p\log(p) + (1 - p)\log (1 - p)}
$$

Calibration: 预测的正样本数量与Test中正样本数量比。有这个指标可以衡量under-delivery 或者 over-delivery。



#### 实验：

##### 数据

首先拿第一天数据做训练，后面数据做Test，发现Test效果会随着天数增加而变差，所以最好每天train一个新model。但可能Train Model太费时了，导致24小时内Train不完。所以他们采用一个折中方案。决策树隔几天训练，但LR使用Online的方式更新。

##### 学习率

实验了不同学习率：

1. Per coordinate weight: $\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{i=1}^{t-1}\Delta_{t,i}^2}}$，梯度越大，导致学习率越小。累积梯度小的学习率应该大一点
2. Per weight square root learning rate $\eta_{t,i} = \frac{\alpha}{\sqrt{n_{t,i}}}$ $n_{t,i}$表示第t轮训练的时候，有第$i$个feature的instance总数。具有该feature样本数越多的权重学习率越小。
3. Per weight learning rate $\eta_{t,i} = \frac{\alpha}{n_{t_i}}$，上面的分母不开根号
4. Global learning rate: $\eta_{t,i} = \frac{\alpha}{\sqrt{t}}$，训练越久，学习率越小
5. Constant learning rate: $\eta_{t,i} = \alpha$

实验结果：Per weight < Global < Constant < Per weight sqrt < per coordinate

##### Feature Importance

在决策树中，每个feature在split的时候得到error reduction，有n颗子树sum起来作为该feature的boost feature importance，实验发现大概top10的feature占了一般的累积importance。所以可以砍掉一部分feature

##### Negative sampling

使用了0.025的负采样率。如果原始CTR为0.1%，在使用1%的负采样之后，训练集中的CTR会变为10%。所以在训练完之后，需要对概率进行一个修正，修正预测概率为

$$
q = \frac{p}{p + (1-p)/w}
$$