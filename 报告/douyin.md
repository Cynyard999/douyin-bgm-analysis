# 抖音热门音乐特征分析

#### 项目地址： https://github.com/Cynyard999/NJUSEDouyinDataAnalysis

## 成员

| 姓名   | 学号      | GitHub账户               | 主要分工 |
| ------ | --------- | ------------------------ | -------- |
| 张卓楠 | 181830249 | github.com/sunflower-zzn | 数据处理 |
| 巫夷   | 181250153 | github.com/RickyWu9      | 数据提取 |
| 邱星曜 | 181830154 | github.com/Cynyard999    | 数据建模 |

## 1. 摘要

本次大作业选取了抖音当下最热门的400首音乐，通过一系列方法提取每首歌的波形特征，再经过降维以及机器学习等手段，得出其大致音乐流派并粗略分类，并通过可视化方法呈现数据，

关键词：特征提取，机器学习，k-means聚类，Librosa，隐马尔可夫模型

## 2. 引言

​	随着移动网络与数字多媒体技术的飞速发展，基于快餐文化而快速崛起的短视频平台已经充斥在人们生活的各个角落，而随着人们的生活进行的”越来越快“，人们的时间貌似也越来越值钱，原本十几分钟才能讲完的事情，被浓缩到几分钟，甚至是十几秒就要讲完。

​	文本变为视频，无疑是满足了人们对于外界认知的获取速率的要求，但短视频平台产生的海量，庞大的视频数据确实大大超出了受众的需求和接收能力，因此，在一个月活五亿的平台上脱颖而出，抓住观众的感官，让视频观看量达到成千上万甚至达到百万级，千万级是所有短视频创作者的第一要务。

​	正如抖音名字所呈现的，音乐是抖音短视频的灵魂，背景音乐的选用是否恰当直接关系到作品的人们程度，因此，究竟什么样的音乐才能成为爆款，推动视频的传播，值得深入研究。

## 3. 研究方法

### 	3.1 数据获取

#### 		3.3.1 获取当下最热门的400首音乐

#### 		3.3.2 音乐格式转换与时长处理

### 	3.2 数据处理

#### 		3.2.1

#### 		3.2.2

### 	3.3 特征提取

### 	3.4 音乐分类

### 音乐频谱图 

将mp3文件转成wav文件后，提取文件中所有的帧的信息。若文件为单通道，则直接将所有帧形成一维矩阵，若为双通道，则提取左声道的帧形成一维矩阵。最后将一维矩阵归一化，再将离散的点连线作图。

#### 图例

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/DancingWithYourGhost.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/%E5%B1%B1%E5%A6%96.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/%E6%83%B3%E8%A7%81%E4%BD%A0%E6%83%B3%E8%A7%81%E4%BD%A0%E6%83%B3%E8%A7%81%E4%BD%A0.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/%E7%88%B1%EF%BC%8C%E5%AD%98%E5%9C%A8.wav.png" width="400"/> 
</figure>

### 音乐语谱图

将音乐频谱图中得到的一维矩阵，将该矩阵形成谱图。

#### 图例

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E4%BD%A0%E5%95%8A%E4%BD%A0%E5%95%8A.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E5%B1%B1%E5%A6%96.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E5%BE%AE%E5%BE%AE.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E7%88%B1%EF%BC%8C%E5%AD%98%E5%9C%A8.wav.png" width="400"/> 
</figure>

### 音乐音调变化图

由于wav文件是模仿音乐的波形，是时域上的变化，因此想要想要得到频域上的变化，就需要利用傅里叶变换，将波形拆分成多个正弦曲线。不同的正弦曲线，代表的不同音调的波形，所以我们无法获取绝对应高，只能比较时间维度上，相对的音调变化。我们认为，频率高且声音响度大的正弦曲线代表了高音，而频率低且声音响度低的曲线代表了低音。于是，我们对音乐帧进行分段，8000个帧为一小段，进行快速傅里叶变换，并对分解的曲线进行对应其响度上的加权，最终获得该小段上的相对音高。最终将这些音高做成曲线。

#### 图例

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/DancingWithYourGhost.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E4%B8%80%E5%8D%83%E9%9B%B6%E4%B8%80%E6%AC%A1%E6%88%91%E7%88%B1%E4%BD%A0.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E5%8F%AB%E6%88%91baby.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E6%97%A7%E6%A2%A6%E4%B8%80%E5%9C%BA.wav.png" width="400"/> 
</figure>

### midi文件生成

根据读取wav文件的帧，对其进行傅里叶变换获取音高信息，最终导入mid文件

山妖原音乐

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96.wav"> </audio>
山妖mid文件音乐

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96mp32mid.mp3"> </audio>
### 音乐自相似矩阵图

读取midi文件,根据其音调信息生成谱图

#### 图例

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%98%B5/DancingWithYourGhost.mid.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%98%B5/%E4%BD%A0%E5%95%8A%E4%BD%A0%E5%95%8A.mid.png" width="400"/> 
</figure>
## 研究方法

### 音乐特征提取分类

在之前的数据处理部分，我们绘制了不同音乐间的歌的**波形结构**，但我们发现即使同一名歌手演唱的歌曲之间的差异也很大，而我们想要探讨这些抖音热歌之间的共同点以及与非热门歌曲之间的区别，为此我们想到了提取音乐的特征并进行**聚类**来探讨他们之间的**关联性**。

#### 特征提取

我们首先需要做的是音乐的特征提取，我们从原始的WAV文件中提取统计要素，我们人为地选取了42个音频特征值：

- **歌曲波形的统计矩**，包括*均值*、*标准差*、*偏态*和*峰态*，同时，我们通过平滑窗(递增平滑，长度分别为1,10,100,1000)来获取这些特征在不同时间尺度上的表现；
- 为了体现信号的短时变化，我们可以计算一下**波形一阶差分幅度的统计矩**，同样也通过平滑窗来获取这些特征(*均值*、*标准差*、*偏态*和*峰态*)在不同时间尺度上的表现；
- 最后，我们计算一下**波形的频域特征**，这里我们只计算歌曲在不同*频段*(将整个频段均分为10份)的能量占比，不过直接对歌曲的波形数据作快速傅里叶变换的话其计算量过于庞大了，因此先让波形数据通过长度为5的平滑窗再对其作快速傅里叶变换。

最终得出如下42个特征值：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/42%E7%89%B9%E5%BE%81%E5%80%BC%E7%A4%BA%E4%BE%8B.png)

```python
# amp1mean
# amp1std
# amp1skew
# amp1kurt
# amp1dmean
# amp1dstd
# amp1dskew
# amp1dkurt
# amp10mean
# amp10std
# amp10skew
# amp10kurt
# amp10dmean
# amp10dstd
# amp10dskew
# amp10dkurt
# amp100mean
# amp100std
# amp100skew
# amp100kurt
# amp100dmean
# amp100dstd
# amp100dskew
# amp100dkurt
# amp1000mean
# amp1000std
# amp1000skew
# amp1000kurt
# amp1000dmean
# amp1000dstd
# amp1000dskew
# amp1000dkurt
# power1
# power2
# power3
# power4
# power5
# power6
# power7
# power8
# power9
# power10
```

##### 筛选特征值的最佳子集

然而在使用这些特征值的过程中我们发现，由于这些特征值是人为提取的，所以并不能很好地表现出歌曲特征，并且这些特征之间的相关系数是不为0的，也就是说存在冗余特征，因此我们需要对特征值进行筛选。

我们从每首歌曲中抽取两个15s 的样本，并尝试找到一种算法，该算法可以最好地将每首歌曲中的两个样本匹配在一起。

为了找到在所有歌曲中提供最佳平均匹配特征值子集，我们使用**遗传算法（R 中的 genalg 包）**来控制42 个特征值中的每个特征值。下面的图显示了遗传算法超过 100 代的目标函数（即歌曲的两个样本由最近的邻域分类器匹配在一起）的结果。

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95%E7%AD%9B%E9%80%89%E7%89%B9%E5%BE%81%E5%80%BC.png)

如果我们选用全部42个特征值，那么目标函数的错误率会达到275，而通过筛选上图中右侧的18个特征值，我们能够将目标函数的错误率在100次以上迭代中降低到90，这是一个重大的优化。

通过对42个特征值数组的提取重组，我们可以得到音乐的18个特征值向量：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/18%E7%89%B9%E5%BE%81%E5%80%BC%E7%A4%BA%E4%BE%8B.png)

##### 归一化和降维处理

为了最后服务于我们的歌曲聚类目标以及最后结果的可视化，我们需要将这些数据进行归一化处理和降维处理。

使用StandardScaler进行去均值和方差归一化，调用sklearn的preprocessing库进行数据的归一化。

使用PCA主成分分析方法进行数据降维，将18维数据降为2维数据，调用sklearn的decomposition进行主成分分析降维。

最终处理结束后绘制数据分布图：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE%E7%BB%98%E5%9B%BE.png)

### K-Means聚类

为了分析这些音乐数据之间的关联关系，我们使用了无监督分类的方式试图找到他们之间的关联关系。

##### 传统K-Means算法

KMeans算法通过尝试在等方差组中分离样本来对数据进行聚类，从而最小化称为惯性或聚类内平方和的标准。此算法要求指定群集数。它可很好地扩展到大量的样品中，并已被广泛应用于许多不同的领域。k-means算法将一组样本划分为不相交的聚类，每个样本都用群集中样本的均值描述。这些手段通常称为聚类"中心"。

我们绘制了k=5-k=13时的聚类结果：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/KMeans.png)

##### MiniBatchKMeans算法

MiniBatchKMeans是KMeans算法的变体，该算法使用小型批处理来缩短计算时间，同时仍在尝试优化相同的目标函数。小型批处理是输入数据的子集，在每个训练迭代中随机采样。这些小型批处理大大减少了收敛到本地解决方案所需的计算量。与其他减少 k-means 收敛时间的算法相比，小批 k-means 产生的结果通常只比标准算法稍差。

我们绘制了k=5-k=13时的聚类结果：
![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/MiniBatchKMeans.png)

##### 选择K值

### 音乐流派分类



##  4. 结论



## 5. 参考文献

[R语言中的遗传算法](http://blog.fens.me/algorithm-ga-r/)

[sklearn聚类算法官网](https://scikit-learn.org/stable/modules/clustering.html#)

 [基于神经网络的音乐流派分类](https://medium.com/@navdeepsingh_2336/identifying-the-genre-of-a-song-with-neural-networks-851db89c42f0)



## 6. 不足以及展望

只研究了声学三要素相关的知识，没有研究音乐三要素（旋律，和声，节奏）相关的内容，并没有直接的midi文件，没有生成音乐