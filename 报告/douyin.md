# 抖音热门音乐特征分析

**项目地址： https://github.com/Cynyard999/NJUSEDouyinDataAnalysis**

> 最后写完了再加toc

# 成员

| 姓名   | 学号      | GitHub账户               | 主要分工 |
| ------ | --------- | ------------------------ | -------- |
| 张卓楠 | 181830249 | github.com/sunflower-zzn | 数据处理 |
| 巫夷   | 181250153 | github.com/RickyWu9      | 数据提取 |
| 邱星曜 | 181830154 | github.com/Cynyard999    | 数据建模 |

# 1. 摘要

本次大作业选取了抖音当下最热门的400首音乐，通过一系列方法提取每首歌的波形特征，再经过降维以及机器学习等手段，得出其大致音乐流派并粗略分类，并通过可视化方法呈现数据，

**关键词**：特征提取，机器学习，k-means聚类，Librosa，隐马尔可夫模型，midi

# 2. 引言

​	随着移动网络与数字多媒体技术的飞速发展，基于快餐文化而快速崛起的短视频平台已经充斥在人们生活的各个角落，而随着人们的生活进行的”越来越快“，人们的时间貌似也越来越值钱，原本十几分钟才能讲完的事情，被浓缩到几分钟，甚至是十几秒就要讲完。

​	文本变为视频，无疑是满足了人们对于外界认知的获取速率的要求，但短视频平台产生的海量，庞大的视频数据确实大大超出了受众的需求和接收能力，因此，在一个月活五亿的平台上脱颖而出，抓住观众的感官，让视频观看量达到成千上万甚至达到百万级，千万级是所有短视频创作者的第一要务。

​	正如抖音名字所呈现的，音乐是抖音短视频的灵魂，背景音乐的选用是否恰当直接关系到作品的人们程度，因此，究竟什么样的音乐才能成为爆款，推动视频的传播，值得深入研究。

# 3. 研究方法

## 3.1 数据获取

### 3.1.1 获取当下最热门的400首音乐

​	由于数据量要求过大，仅仅通过人工获取数据明显是一个不现实的手段，我们选择利用爬虫工具对数据进行大批量的获取。但由于抖音短视频官网做了相当严密的反爬虫机制，很难从抖音官网获取视频信息，以及得到其背景音乐，所以我们选择使用第三方抖音数据分析网站来获取我们需要的热门音乐数据。

​	我们使用了*新榜、抖查查、66榜、飞瓜数据、卡思数据、蝉妈妈*等近十个分析网站，最终选取了蝉妈妈作为音乐数据来源。

​	注册会员后，通过点击音乐榜单，在f12控制台的network窗口监听到浏览器发出的http请求以及服务器返回的数据，经过筛选得到获取音乐榜单的请求：

*search?keyword=&page=1&size=50&orderby=user_count&incr_type=7d&order=desc*

​	通过分析header信息，利用python的*request*库模拟header对服务器发出请求，并且筛选得到音乐的名称，播放量等信息，写入csv:

```python
try:
  response = requests.get(url, headers=headers)
  if response.status_code == 200:
    list = response.json().get('data').get('list')
    for item in list:
      del item['cover_image']
      del item['use_trend']
      del item['hot_awemes']
      del item['is_fav']
      return list
except requests.ConnectionError as e:
   print('Error', e.args)
...
with open(csvPath, "w") as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(["音乐ID", "音乐名", "创作者", "歌曲时间", "使用人数", "使用人数增量"])
  for item in music_info:
    writer.writerow([item.get('music_id'), item.get('title'), item.get('author'), item.get('audition_duration'),
                     item.get('user_count'), item.get('user_incr')])
    print("csv写入成功！")

```

​	在得到音乐信息后，通过音乐的id，再次向服务器请求，获取音乐的mp3地址，下载后存储在本地:

```python
 try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
      src = response.json().get('data')
      print(name, end=": ")
      if src != '' and src != 'None':
        music = requests.get(src)
        # 下面填写本地存储的路径，记得后缀添加mp3
        open(downLoadPath + name + '.mp3', 'wb').write(music.content)
        print("成功")
     else:
      print("歌曲不存在")
	except requests.ConnectionError as e:
		print('Error', e.args)  # 输出异常信息
```

## 3.2 音频处理

### 3.2.1 音乐格式转换与时长处理

​	由于下载的音乐格式均为*mp3*，而后续所需的所有格式均为wav，所以先进行进一步的mp3到wav的转化:

```python
for file in files: #遍历文件夹
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
          # 读取mp3的波形数据
          sound = AudioSegment.from_file(path+"/"+file, format='MP3')
          # 将读取的波形数据转化为wav
          f = wave.open(file+".wav", 'wb')
          f.setnchannels(1)  # 频道数
          f.setsampwidth(2)  # 量化位数
          f.setframerate(16000)  # 取样频率
          f.setnframes(len(sound._data))  # 取样点数，波形数据的长度
          f.writeframes(sound._data)  # 写入波形数据
          f.close()
```

​	由于大多数短视频的时长只有15s到20s，所以我们可以进一步将wav缩短至15s，因为我们认为，对于观看者而言，前15秒不能吸引到注意力，那么观看者将会毫不犹豫的切换到下一个视频。缩短音乐时长也同时降低了后期处理的复杂性:

```python
def split_music(begin, end, filepath, filename):
    # 导入音乐
    song = AudioSegment.from_mp3(filepath)

    # 取begin秒到end秒间的片段
    song = song[begin * 1000: end * 1000]

    # 存储为临时文件做备份
    temp_path = '../音乐/backup/' + filename+'.wav'
    song.export(temp_path)
    return temp_path
```

## 3.3 音频分析

> 音频信号是（Audio）带有语音、音乐和音效的有规律的声波的频率、幅度变化信息载体。 根据声波的特征，可把音频信息分类为规则音频和不规则声音。其中规则音频又能够分为语音、音乐和音效。规则音频是一种连续变化的模拟信号，可用一条连续的曲线来表示，称为声波。声音的三个要素是音调、音强和音色。声波或正弦波有三个重要参数：频率 、幅度和相位 ，这也就决定了音频信号的特征。即**不一样频率和相位的正弦波的一个叠加**

### 3.3.1 基础特征提取

**3.3.1.1 波形分析**

*均值*、*标准差*、*偏态*和*峰态*

**3.3.1.6 音调提取**

**3.3.1.2 过零率**

**3.3.1.3 光谱质心变化**

**3.3.1.4 光谱衰减**

**3.3.1.5 梅尔频率倒谱系数**

### 3.3.2 谱图生成

**3.3.2.1** **频谱图**

**3.3.2.2** **语谱图** 

### 3.3.3 midi文件生成





### 3.3.4 波形统计特征处理

> ​	在之前的波形特征提取部分，我们绘制了不同音乐间的歌的**波形结构**，但我们发现即使同一名歌手演唱的歌曲之间的差异也很大，而我们想要探讨这些抖音热歌之间的共同点以及与非热门歌曲之间的区别，为此我们想到了提取音乐的特征并进行**聚类**来探讨他们之间的**关联性**。

我们首先需要做的是波形的特征提取，我们从原始的WAV文件中提取统计要素，我们人为地选取了42个音频特征值：

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

### 3.3.5 筛选特征值的最佳子集

然而在使用这些特征值的过程中我们发现，由于这些特征值是人为提取的，所以并不能很好地表现出歌曲特征，并且这些特征之间的相关系数是不为0的，也就是说存在冗余特征，因此我们需要对特征值进行筛选。

我们从每首歌曲中抽取两个15s 的样本，并尝试找到一种算法，该算法可以最好地将每首歌曲中的两个样本匹配在一起。

为了找到在所有歌曲中提供最佳平均匹配特征值子集，我们使用**遗传算法（R 中的 genalg 包）**来控制42 个特征值中的每个特征值。下面的图显示了遗传算法超过 100 代的目标函数（即歌曲的两个样本由最近的邻域分类器匹配在一起）的结果。

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95%E7%AD%9B%E9%80%89%E7%89%B9%E5%BE%81%E5%80%BC.png)

如果我们选用全部42个特征值，那么目标函数的错误率会达到275，而通过筛选上图中右侧的18个特征值，我们能够将目标函数的错误率在100次以上迭代中降低到90，这是一个重大的优化。

通过对42个特征值数组的提取重组，我们可以得到音乐的18个特征值向量：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/18%E7%89%B9%E5%BE%81%E5%80%BC%E7%A4%BA%E4%BE%8B.png)

### 3.3.6 归一化和降维处理

为了最后服务于我们的歌曲聚类目标以及最后结果的可视化，我们需要将这些数据进行归一化处理和降维处理。

使用StandardScaler进行去均值和方差归一化，调用sklearn的preprocessing库进行数据的归一化。

使用PCA主成分分析方法进行数据降维，将18维数据降为2维数据，调用sklearn的decomposition进行主成分分析降维。

最终处理结束后绘制数据分布图：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE%E7%BB%98%E5%9B%BE.png)


## 3.4 音乐分类
### 3.4.1 K-Means聚类

为了分析这些音乐数据之间的关联关系，我们使用了无监督分类的方式试图找到他们之间的关联关系。

#### 3.4.1.1 传统K-Means算法

KMeans算法通过尝试在等方差组中分离样本来对数据进行聚类，从而最小化称为惯性或聚类内平方和的标准。此算法要求指定群集数。它可很好地扩展到大量的样品中，并已被广泛应用于许多不同的领域。k-means算法将一组样本划分为不相交的聚类，每个样本都用群集中样本的均值描述。这些手段通常称为聚类"中心"。

我们绘制了k=5-k=13时的聚类结果：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/KMeans.png)

#### 3.4.1.2 MiniBatchKMeans算法

MiniBatchKMeans是KMeans算法的变体，该算法使用小型批处理来缩短计算时间，同时仍在尝试优化相同的目标函数。小型批处理是输入数据的子集，在每个训练迭代中随机采样。这些小型批处理大大减少了收敛到本地解决方案所需的计算量。与其他减少 k-means 收敛时间的算法相比，小批 k-means 产生的结果通常只比标准算法稍差。

我们绘制了k=5-13时的聚类结果：
![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/MiniBatchKMeans.png)

#### 3.4.1.3 选择K值

对于K-Means聚类算法来说，K值的选择是至关重要的，很多情况下K值的估计是非常困难的，对于像计算全部微信用户的交往圈这样的场景就完全的没办法用K-Means进行。对于可以确定K值不会太大但不明确精确的K值的场景，可以进行迭代运算，然后找出Cost Function最小时所对应的K值，这个值往往能较好的描述有多少个簇类。

对于我们本次作业中分析的项目，音乐的特征分类，我们主要采用了两个评价指标，分别是：calinski_harabasz_score评估系数和轮廓系数 silhouette_score。

**calinski_harabasz_score评估系数**

CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。

我们通过调用sklearn中的metrics函数，计算出不同k值下K-Means模型的CH指数，结果如下图：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/CH%E6%8C%87%E6%A0%87.png)

可以看到效果并不如我们预期的那样出现峰值，而是呈现随着k值增长，CH指数不断增加的趋势，这一方面说明CH指数评价指标并不适合我们的音乐特征数据，由之前的原始数据我们可以看到样本数据的特征分布并不呈现比较完美的分簇化趋势，而是中心集中，因此当k值接近样本数量时CH指数会上升至最高。另一方面也推动我们寻找其他的聚类评价指标，于是我们找到了另一种评价指标——轮廓系数。

**轮廓系数 silhouette_score**

- 计算样本i到同簇其他样本的平均距离ai。ai 越小，说明样本i越应该被聚类到该簇。将ai 称为样本i的**簇内不相似度**。
- 簇C中所有样本的a i 均值称为簇C的簇不相似度。
- 计算样本i到其他某簇Cj 的所有样本的平均距离bij，称为样本i与簇Cj 的不相似度。定义为样本i的**簇间不相似度**：bi =min{bi1, bi2, ..., bik} bi越大，说明样本i越不属于其他簇。
- 根据样本i的簇内不相似度a i 和簇间不相似度b i ，定义样本i的轮廓系数

![img](https://upload-images.jianshu.io/upload_images/6315044-8cb8fca0ec651d7e.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

该值取值范围为[−1,1][-1, 1][−1,1]， 越接近1则说明分类越优秀。在`sklearn`中函数`silhouette_score()`计算所有点的平均轮廓系数，而`silhouette_samples()`返回每个点的轮廓系数。我们调用了sklearn中的metrics库函数计算不同k值下K-Means模型的轮廓系数，绘制图形如下：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E8%BD%AE%E5%BB%93%E7%B3%BB%E6%95%B0.png)

可以看到，相较于CH指数，轮廓系数能够显著的显示某一K值的局部峰值，如图即K=10时K-Means算法聚类效果最好，我们可以设置K=10。

#### 3.4.1.4 聚类结果
最后，我们可以根据聚类结果生成音乐信息分类情况，为我们后续的热门音乐流派分析提供对比集：

```
class 0 [数量：20]：['叫我baby', '山妖', 'Pew Pew!', '师傅我坚持不住了', '捣蛋鬼', '桃花笑', '123我爱你-剪辑版', 'Pew Pew！', '意不意外', '谁家的姑娘长得这么漂亮', '@用户创作的原声', '飞歌童装拼货团', '小兔子卖报', '@青梅创作的原声', '可不可以给我你的微信？', '弹光头 铛～', '春', '魔性的笑声', '女孩子花点钱怎么了_怎么了！！', '我们是一家人-剪辑版']

class 1 [数量：73]：['你啊你啊', 'DancingWithYourGhost', '用户原声', '旧梦一场', '溯', 'Crying Over You', 'Dancing With Your Ghost', '溯 (Reverse) feat. 马吟吟', 'Soap', '旧梦一场（DJ沈念版）', '世界这么大还是遇见你（剪辑版）', '风景', '一生与你擦肩而过', '往后余生', '走着走着花就开了', '下山', '答案', '我要吃肉肉~1', 'Raindrops', '你是我唯一的执着-女声剪辑版', '我的老家在农村-剪辑版', '纸短情长', 'Asobimasu ', '我爱你中国', '多年以后', '祖国的花朵_剪辑版', '@黄姐创作的原声', '多年以后 - 剪辑版', '努力奋斗', '放假了', '往后余生~1', '调皮', '小鹿乱撞', '桥边姑娘-海伦', '最天使（Cover 曾轶可）', '风筝误', '最浪漫的事', '@82狗狗🐕罗姐创作的原声', '好好爱自己', '可不可以', '王菲-我和我的祖国', '我又想你了-刘奕辰', '如果有一天我们都老了', '伪装-剪辑版', '你若三冬', '惊雷', '好嗨哟disco剪辑版', '@哈皮创作的原声', '夏天的风', '往后余生_王贰浪', '万爱千恩（剪辑版）', '目不转睛', '@张哈哈🥕创作的原声', '闯码头-剪辑版', '霸气哥只是个传说', '@邝邝邝邝邝啊创作的原声', '风景（副歌）', '流浪~1', '赢在江湖（童声版）-张振轩', '小简单-小樱俊-剪辑版', '@机器李创作的原声', '蓝天白云', '过活', '勇气', '用户创作的原声~4', 'feeling', 'Way Back Home', '白羊', '不再等候的姑娘', '带你去旅行', '你的答案_阿冗', '我的快乐就是想你', '远走高飞']

class 2 [数量：53]：['官方回答', '用了必热', '后来遇见他', '江南', '干就完了-剪辑版', 'Times', '百花香', '挣钱难DJ', 'Keep This Fire Burning', '因为当时阳光灿烂', '生僻字（剪辑版）', 'HandClap', '用户创作的原声~1', '@翠竹轩创作的原声', '老公你在哪', '@燕儿✨记录生活创作的原声', '@༄࿆ζั͡௸风雅এ᭄创作的原声', 'coco创作的原声', '还有多少个十年(励志版剪辑)', '@Z小仙女创作的原声', '@咏梅服装商行创作的原声', '酒醉的蝴蝶一崔伟立', 'Horizon-Janji', '@用户创作的原声~1', '半壶纱', '@💋王姑娘🍊创作的原声', '我爱虹虹兔', '余情未了', 'Nirvana', '你莫走', '@绍兴龍哥创作的原声', '如果就这么老了', '灰姑娘', '爱火抖音版', 'Stranger', '@Q音最🔥音乐-小张创作的原声', '雅欣跑调大王原创的音乐', '@不二小倩创作的原声', '@段磊创作的原声', '一生有你 (《一生有你》电影同名推广曲)', '爱', '@Hà nội  栗子创作的原声', '情花几时开dj', '下辈子不一定遇见', '地震发生后', '灰姑娘~1', '幸福飞翔', '@人生如梦创作的原声', '@🍬糖果🍬创作的原声', '@一双小眼睛创作的原声', '@空心创作的原声', '沙滩往事', '我的肚子好饿']

class 3 [数量：68]：['爱，存在', '想见你想见你想见你', '爱存在', '脆弱星球', '陪你长大 - 大攀（长大版）', '超级喜欢你', '灞波儿奔奔波儿灞', '一起长大的幸福', '我愿意平凡的陪在你身旁-王七七', '最炫小苹果', '怎么开心怎么活', '我们不一样', '万爱千恩-副歌版', '无敌小可爱剪辑版2', '咖喱咖喱超萌版 - 陈奕雯', '《怀念青春》扎心版-剪辑版', '流浪', '摘自皇家音乐阿焱', '《你像三月桃花开》剪辑版', '莎啦啦', '我真的很不错', '向天空呼喊我的寂寞', '绝不会放过', '无敌小可爱（对唱版2）', '忘情牛肉面（女声剪辑版）', '甜心小宝贝', '打工苦打工累', '新健康歌', '@Sun😆创作的原声', '@湘仔建哥🔥创作的原声', '@张茜《谜一样的生活》创作的原声', '最美的光 - 于萱媛 剪辑版', '甜蜜爱情甜蜜一生', '别被爱情冲昏了头', '只是太爱你', '누구 없소 (Feat. B.I of iKON)', '别知己-剪辑版', '忙忙忙-剪辑版', '1, 2, 3, 4 (One, Two, Three, Four) - Fun Elektro Mix', '有你就幸福', '心如止水', '挣钱不容易', '开心快乐每一天', '@🍃💋封心❤锁🔒爱…🍃💅创作的原声', '娘子，我们去哪儿？', '暖暖的小幸福', '胖嘟嘟—剪辑版', '下辈子不一定遇见-剪辑版-梅朵', '爱火蓝琪儿', '《宝贝》', '不放弃', '一个被磨难成就的出家人', '@双胞胎😘俊娜喜娜创作的原声', '快乐家园', '好嗨哟', '@R袁晓蕊创作的原声', '@加色魅小米创作的原声', '海阔天空', '女人累不累', '像梦一样自由', '萤火虫对星星说', '光辉岁月', '小星星', 'Heartbeat', '海草舞', '你笑起来真好看', '我要吃肉肉', '一千零一次我爱你']

class 4 [数量：16]：['情人', '最美的花dj', "I Don't Wanna See You Any More", '@水果小肆创作的原声', 'Future', '芒种', '春有百花 - 陌上花开（剪辑版古筝）', 'Alone on the way', '来过来尝尝这个', 'Mistério', '@杨玉环创作的原声', '洗澡歌', '完整版网易搜罗狗', '我们不怕', '微微', '惜别']

class 5 [数量：7]：['画眉鸟叫', '我要变好看', '绿色', '@🌈🌈风雨彩虹🌈🌈创作的原声', '画皮', '西游记序曲', '变小药水']

class 6 [数量：76]：['来吧开整', 'waiting for love', '你长这样谁要', '叮叮叮', 'chuchacha', '将会面对什么样的结局', '花开的时候你就来看我', '酒醉的蝴蝶', '我怎么这么好看', '最美的期待', '学猫叫-完整剪辑版', '小可爱 （我的小可爱 今天有没有乖）', '123我爱你', '听我说谢谢你－李昕融', '就是不想长大！', '学猫叫-剪辑版', '火山用户创作的原声', '黄梅戏-慕容晓晓', '殇雪（剪辑版）', '三月里的小雨_王恰恰', 'Make Some TikTok', '生日快乐', '@德华饰品创作的原声', '一曲红尘', '@我是苏朋友创作的原声', '宝贝宝贝我爱你（剪辑版）', '天在下雨我在想你', '一路向北-剪辑版', '捉泥鳅', '@✨🌟淡然一笑🌟✨创作的原声', '金久哲 - 干就完了（剪切版）', '你会爱我到什么时候-剪辑版', '敖包相会', '男人要有担当', '小小的太陽', '千年等一回', '站着等你三千年-高潮版', '醉千年', '女人是世界最美丽的花（剪辑版）', '@开心每一天💋创作的原声', 'Wrap Me In Plastic', '嘴巴嘟嘟-剪辑版', 'Delícia Tchu Tcha Tcha - Dj King Remix', '狂浪-花姐', '皮一下很开心(剪辑版)', '用户创作的原声~3', '@天启体验教育创作的原声', 'Getaway ', '多想抱抱你', '小星星~1', '小白兔遇上卡布奇诺', '爱情让我心痛DJ版-赵小南', '确认过眼神（撩妹撩汉版）', '你个小坏坏', '阳光彩虹小白马', '像个孩子', '小奶狗', '她扒拉我', 'MyLove一起去旅行', '我不会唱歌_主歌版', '习惯你的好', '小棉袄儿（剪辑版）', '爱你三千遍（剪辑版）', '点歌的人-海来阿木', '歌曲来自柠檬啊', '@宾县＿＿仓少创作的原声', '忘情牛肉面', '皓皓最棒', '我带上你', '隔壁泰山(part.2)', '@Z小仙女创作的原声~1', '再见彩霞', '生日快乐~1', '就是这个范儿', '姐妹', '你打不过我吧']

class 7 [数量：4]：['Honey Honey', '@你不来💞我不老创作的原声', '@宇尚织发创作的原声', '@赢在未来创作的原声']

class 8 [数量：30]：['哦哈哟欧尼酱', '学猫叫', '我很可爱', '一晃就老了-秋裤大叔 剪切版', 'Canção da piscada', 'Pika Pika Pikachu', 'Sunday（VIP）', '54321', '偷偷', '@DIE创作的原声', 'Say wangwang', '大宝宝', '你会唱小星星吗？', '胖嘟嘟-小叉系', '汪汪呱，汪汪喵', '就吃一口', '千年等一回，全场都两块', '娃娃呼呼', '@叶·知创作的原声', '石头剪刀布', '原来是萝卜丫', '妈妈陪我一起长大-母亲节版', '@senter创作的原声', '人生一世不容易-剪辑版-梅朵', '路上的风景很美-剪辑版', '@杨梓琪呀创作的原声', '你看我可不可爱', '甜蜜蜜', '我和你', '心愿便利贴']

class 9 [数量：61]：['爱出发', 'DJ原声', 'Planet', 'LA LA LAND', 'Moshi Moshi', '少年', '我心里的秘密', '少年-剪辑版', '我的小宝贝-剪辑版', '过年啦', '用户创作的原声', 'Dance Monkey', '沙漠骆驼', '这条街最靓的仔（走起路一定要大摇大摆）', '@猜猜宝贝胖fufu创作的原声', '骑上我心爱的小摩托', '@苏皓创作的原声', '1234喜欢你', '美美哒', '为你祈祷', '桥边姑娘-宋小睿', '喵...汪汪汪汪汪汪', 'LA LA LAND (Part 1)', '山清水秀好风光-张曼', '用户创作的原声~2', '处处吻', '一百万个可能', 'Leyla', '爱的世界只有你', '有可能的夜晚', '有一个笑点低的朋友', 'Moshi Moshi (Part 1)', '寂寞的人伤心的歌', '非正常犬猫养殖户', '天天快乐', 'Asian Power (Part 1)', '不仅仅是喜欢', '大雨还在下 (新版)', "Ain't My Fault - R3hab Remix", '就是这么牛', '雨蝶', '画_網易雲戴羽彤', '这扯淡的人生（主歌剪辑版）', '就这_', 'Chu Desu!', '网友顽童_创作原声', '倒数', '@用户创作的原声~2', '雪糕', '恭喜发财', '宝贝宝贝', '无奈的思绪', '@小红书服装店.创作的原声', '夜', '山那边', '卖萌进行曲-郝美', '《幺妹住在十三寨》', '@简单就好创作的原声', '一生回味一面', '美美哒(剪辑版）', '佛系少女']
```











**以下是格式还未处理的**

---

### 音乐音调变化图(这部分貌似写分析方法的时候可以用上 所以我没删 然后midi文件生成应该是用来获取和弦还是啥的吧 不知道放在哪块我就没动)

由于wav文件是模仿音乐的波形，是时域上的变化，因此想要想要得到频域上的变化，就需要利用傅里叶变换，将波形拆分成多个正弦曲线。不同的正弦曲线，代表的不同音调的波形，所以我们无法获取绝对应高，只能比较时间维度上，相对的音调变化。我们认为，频率高且声音响度大的正弦曲线代表了高音，而频率低且声音响度低的曲线代表了低音。于是，我们对音乐帧进行分段，8000个帧为一小段，进行快速傅里叶变换，并对分解的曲线进行对应其响度上的加权，最终获得该小段上的相对音高。最终将这些音高做成曲线。


### midi文件生成

根据读取wav文件的帧，对其进行傅里叶变换获取音高信息，最终导入mid文件

山妖原音乐

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96.wav"> </audio>
山妖mid文件音乐

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96mp32mid.mp3"> </audio>




---

**以上是格式还未处理的**







##  4. 案例分析

#### 4.1 数据爬取以及处理

​	以蝉妈妈数据榜单前50首为例，通过对header的解析，模拟浏览器向服务器发出请求，最终保存音乐以及音乐信息至本地。

​	

#### 4.2 特征提取

##### 4.2.1 音乐频谱图 

将mp3文件转成wav文件后，提取文件中所有的帧的信息。若文件为单通道，则直接将所有帧形成一维矩阵，若为双通道，则提取左声道的帧形成一维矩阵。最后将一维矩阵归一化，再将离散的点连线作图。任取十首，示范如下：

**图例**

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/DancingWithYourGhost.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/%E5%B1%B1%E5%A6%96.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/%E6%83%B3%E8%A7%81%E4%BD%A0%E6%83%B3%E8%A7%81%E4%BD%A0%E6%83%B3%E8%A7%81%E4%BD%A0.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%A2%91%E8%B0%B1%E5%9B%BE/%E7%88%B1%EF%BC%8C%E5%AD%98%E5%9C%A8.wav.png" width="400"/> 
</figure>
**4.2.2** **音乐语谱图**

将音乐频谱图中得到的一维矩阵，将该矩阵形成谱图。任取4首效果如下:

图例：

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E4%BD%A0%E5%95%8A%E4%BD%A0%E5%95%8A.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E5%B1%B1%E5%A6%96.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E5%BE%AE%E5%BE%AE.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E7%88%B1%EF%BC%8C%E5%AD%98%E5%9C%A8.wav.png" width="400"/> 
</figure>

**4.2.3 音乐音调变化图**

利用傅里叶变换，将波形拆分成多个正弦曲线。不同的正弦曲线，代表的不同音调的波形，所以我们无法获取绝对应高，只能比较时间维度上，相对的音调变化。对音乐帧进行分段，8000个帧为一小段，进行快速傅里叶变换，并对分解的曲线进行对应其响度上的加权，最终获得该小段上的相对音高。最终将这些音高做成曲线。任意取四首效果如下

**图例**

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/DancingWithYourGhost.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E4%B8%80%E5%8D%83%E9%9B%B6%E4%B8%80%E6%AC%A1%E6%88%91%E7%88%B1%E4%BD%A0.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E5%8F%AB%E6%88%91baby.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E6%97%A7%E6%A2%A6%E4%B8%80%E5%9C%BA.wav.png" width="400"/> 
</figure>

**4.2.4 音乐自相似矩阵图**

读取midi文件,根据其音调信息生成谱图

**图例**

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%98%B5/DancingWithYourGhost.mid.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%98%B5/%E4%BD%A0%E5%95%8A%E4%BD%A0%E5%95%8A.mid.png" width="400"/> 
</figure>







## 5. 结论

## 6. 参考文献

[R语言中的遗传算法](http://blog.fens.me/algorithm-ga-r/)

[sklearn聚类算法官网](https://scikit-learn.org/stable/modules/clustering.html#)

 [基于神经网络的音乐流派分类](https://medium.com/@navdeepsingh_2336/identifying-the-genre-of-a-song-with-neural-networks-851db89c42f0)



## 7. 不足以及展望

只研究了声学三要素相关的知识，没有研究音乐三要素（旋律，和声，节奏）相关的内容，并没有直接的midi文件，没有生成音乐