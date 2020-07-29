# 抖音热门音乐特征分析

### TODO:

wy:

* 3.2 音频处理部分增加midi生成的代码 以及解释
* 3.3.1.2音调提取部分的解释以及代码 以及4.2.3后面的补充一下案例实践
* 4.5 案件事例的midi文件生成 就是融合那里 然后可以解释一下 做的效果不好什么的
* 把自相似矩阵和midi那里 就是连贯一下 要能说通 就是为什么要用这个 是为了后面干嘛干嘛
* 3.3.1.1的波形提取部分的解释以及代码 包括后面的4.2 的波形分析 的音乐波形图和频谱图（也算是语谱图）部分的实例图片 运行图片 这样 
* 我想不到了



zzn：

* 聚类算法以及后面的补充



qxy：

* 结论部分
* ....





其他的 你们看到能加的就加吧

然后

# 奥利给









**项目地址： https://github.com/Cynyard999/NJUSEDouyinDataAnalysis**
<!-- TOC -->

- [抖音热门音乐特征分析](#抖音热门音乐特征分析)
- [成员](#成员)
- [1. 摘要](#1-摘要)
- [2. 引言](#2-引言)
- [3. 研究方法](#3-研究方法)
    - [3.1 数据获取](#31-数据获取)
        - [3.1.1 获取当下最热门的400首音乐](#311-获取当下最热门的400首音乐)
    - [3.2 音频处理](#32-音频处理)
        - [3.2.1 音乐格式转换与时长处理](#321-音乐格式转换与时长处理)
        - [3.2.2 midi生成](#322-midi生成)
    - [3.3 音频特征](#33-音频特征)
        - [3.3.1 基础音频信息](#331-基础音频信息)
            - [3.3.1.1 波形提取](#3311-波形提取)
            - [3.3.1.2 音调提取](#3312-音调提取)
            - [3.3.1.3 过零率](#3313-过零率)
            - [3.3.1.4 光谱质心变化](#3314-光谱质心变化)
            - [3.3.1.5 光谱衰减](#3315-光谱衰减)
            - [3.3.1.6 梅尔频率倒谱系数](#3316-梅尔频率倒谱系数)
        - [3.3.2 统计特征提取](#332-统计特征提取)
            - [3.3.2.1 时域特征（waveform）](#3321-时域特征waveform)
            - [3.3.2.2 频域特征（spectrogram）](#3322-频域特征spectrogram)
            - [3.3.2.3 差分](#3323-差分)
            - [3.3.2.4 快速傅里叶变换（FFT）](#3324-快速傅里叶变换fft)
            - [3.3.2.5 平滑滤波（加窗）](#3325-平滑滤波加窗)
    - [3.4 数据降维及归一化](#34-数据降维及归一化)
        - [3.4.1 PCA主成分分析](#341-pca主成分分析)
        - [3.4.2 数据归一化](#342-数据归一化)
    - [3.5 聚类算法](#35-聚类算法)
        - [3.5.1 K-Means聚类](#351-k-means聚类)
        - [3.5.2 二分K-均值(bisecting K-means)](#352-二分k-均值bisecting-k-means)
        - [3.5.3 MiniBatch k-Means](#353-minibatch-k-means)
    - [3.6 分类算法](#36-分类算法)
        - [3.6.1 HMM算法](#361-hmm算法)
- [4. 案例实践](#4-案例实践)
    - [4.1 数据爬取以及处理](#41-数据爬取以及处理)
    - [4.2 波形分析](#42-波形分析)
        - [4.2.1 音乐频谱图](#421-音乐频谱图)
        - [4.2.2 音乐语谱图](#422-音乐语谱图)
        - [4.2.3 音乐音调变化图](#423-音乐音调变化图)
        - [4.2.4 音乐自相似矩阵图](#424-音乐自相似矩阵图)
    - [4.3 音乐特征分析聚类](#43-音乐特征分析聚类)
        - [4.3.1 统计特征提取](#431-统计特征提取)
        - [4.3.2 特征选择](#432-特征选择)
        - [4.3.3 归一化和降维处理](#433-归一化和降维处理)
        - [4.3.4 K-Means聚类](#434-k-means聚类)
            - [4.3.4.1 传统K-Means算法](#4341-传统k-means算法)
            - [4.3.4.2 MiniBatchKMeans算法](#4342-minibatchkmeans算法)
            - [4.3.4.3 选择K值](#4343-选择k值)
            - [4.3.4.4 聚类结果](#4344-聚类结果)
    - [4.4 音乐分类](#44-音乐分类)
    - [4.5 midi文件生成](#45-midi文件生成)
- [5. 结论](#5-结论)
- [6. 反思与不足](#6-反思与不足)
- [7. 参考文献](#7-参考文献)

<!-- /TOC -->
> 最后写完了再加toc

# 成员

| 姓名   | 学号      | GitHub账户               | 主要分工 |
| ------ | --------- | ------------------------ | -------- |
| 张卓楠 | 181830249 | github.com/sunflower-zzn | 数据处理 |
| 巫夷   | 181250153 | github.com/RickyWu9      | 数据提取 |
| 邱星曜 | 181830154 | github.com/Cynyard999    | 数据建模 |

# 1. 摘要

本次大作业选取了抖音当下最热门的400首音乐，通过一系列方法提取每首歌的波形特征，再经过降维以及机器学习等手段，进行无监督学习对音乐数据进行聚类的同时训练并使用监督学习分类器进行音乐流派分类，并通过可视化方法呈现分类聚类效果。

**关键词**：特征提取，PCA主成分分析，Normalization归一化，sklearn机器学习，pytorch神经网络，k-means聚类，Librosa音频处理，HMM隐马尔可夫模型，midi音序

# 2. 引言

随着移动网络与数字多媒体技术的飞速发展，基于快餐文化而快速崛起的短视频平台已经充斥在人们生活的各个角落，而随着人们的生活进行的“越来越快”，人们的时间貌似也越来越值钱，原本十几分钟才能讲完的事情，被浓缩到几分钟，甚至是十几秒就要讲完。

文本变为视频，无疑是满足了人们对于外界认知的获取速率的要求，但短视频平台产生的海量，庞大的视频数据确实大大超出了受众的需求和接收能力，因此，在一个月活五亿的平台上脱颖而出，抓住观众的感官，让视频观看量达到成千上万甚至达到百万级，千万级是所有短视频创作者的第一要务。

正如抖音名字所呈现的，音乐是抖音短视频的灵魂，背景音乐的选用是否恰当直接关系到作品的人们程度，因此，究竟什么样的音乐才能成为爆款，推动视频的传播，值得深入研究。

# 3. 研究方法

## 3.1 数据获取

### 3.1.1 获取当下最热门的400首音乐

​	由于数据量要求过大，仅仅通过人工获取数据明显是一个不现实的手段，我们选择利用爬虫工具对数据进行大批量的获取。但由于抖音短视频官网做了相当严密的反爬虫机制，很难从抖音官网获取视频信息，以及得到其背景音乐，所以我们选择使用第三方抖音数据分析网站来获取我们需要的热门音乐数据。

​	我们调查研究了*新榜、抖查查、66榜、飞瓜数据、卡思数据、蝉妈妈*等近十个分析网站，最终选取了蝉妈妈作为音乐数据来源。

​	注册会员后，通过点击音乐榜单，在f12控制台的network窗口监听到浏览器发出的http请求以及服务器返回的数据，经过筛选得到获取音乐榜单的请求：

`search?keyword=&page=1&size=50&orderby=user_count&incr_type=7d&order=desc` 

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

​	由于下载的音乐格式均为*mp3*，而后续所需的所有格式均为*wav*，所以先进行进一步的*mp3*到*wav*的转化:

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

### 3.2.2 midi生成

​	由于我们要获取尽可能更多的纯音乐信息并设法生成我们的wav音乐，所以我们要从wav文件得到其midi文件的映射，从而来研究音乐的音调变化

```python
import os
import subprocess
import time
dir='..\\wav\\'
filename= os.listdir(dir)
exe='wav2midi.exe'
print(filename)
for file in filename:
    p = subprocess.Popen(exe+' '+dir+file)
    time.sleep(10)
    p.kill()
```

​	利用wave to midi映射程序，得到我们想要的midi文件（该文件在降噪处理后，利用傅里叶变换获取频率信息，再根据频率对应的音高得到音调）

## 3.3 音频特征

> 音频信号是（Audio）带有语音、音乐和音效的有规律的声波的频率、幅度变化信息载体。 根据声波的特征，可把音频信息分类为规则音频和不规则声音。其中规则音频又能够分为语音、音乐和音效。规则音频是一种连续变化的模拟信号，可用一条连续的曲线来表示，称为声波。声音的三个要素是音调、音强和音色。声波或正弦波有三个重要参数：频率 、幅度和相位 ，这也就决定了音频信号的特征。即**不一样频率和相位的正弦波的一个叠加**

### 3.3.1 基础音频信息

这里主要介绍波形结构和音频基础信息，即直接读取wav文件就可以获取的数据

#### 3.3.1.1 波形提取

提取文件中所有的帧的信息。若文件为单通道，则直接将所有帧形成一维矩阵，若为双通道，则提取左声道的帧形成一维矩阵。最后将一维矩阵归一化，再将离散的点连线作图。

```python
f=wave.open(filepath+file,'rb')#打开文件
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
waveData = np.reshape(waveData,[nframes,nchannels])
# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)
plt.subplot(3,1,1)
plt.plot(time,waveData[:,0])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Double channel wavedata")
plt.grid('on')#标尺，on：有，off:无。
plt.subplot(3,1,3)
plt.plot(time,waveData[:,1])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Double channel wavedata")
plt.grid('on')#标尺，on：有，off:无。
plt.savefig('..\\波形图\\' + file + '.png')
plt.close()
print(file, "波形图已保存")
```



#### 3.3.1.2 音调提取

利用傅里叶变换，将波形拆分成多个正弦曲线。不同的正弦曲线，代表的不同音调的波形，所以我们无法获取绝对应高，只能比较时间维度上，相对的音调变化。对音乐帧进行分段，8000个帧为一小段，进行快速傅里叶变换，并对分解的曲线进行对应其响度上的加权，最终获得该小段上的相对音高。最终将这些音高做成曲线。

```python
def pitch(list):
    ans=0
    for i in range(len(list)):
        ans+=(i+1)*abs(list[i])
    ans=ans/len(list)
    return  ans
f = wave.open(filepath+file, 'rb')#读取文件
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
time = nframes / framerate
strData = f.readframes(nframes)  # 读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
data = waveData[0::2]
interval = 8000
extra = len(data) % interval
data = data[0:len(data) - extra]
anslist = []
for i in range(0, len(data), interval):
    list = data[i:i + interval]
    list = np.fft.rfft(list)
    anslist.append(int(pitch(list)))
```



#### 3.3.1.3 过零率

#### 3.3.1.4 光谱质心变化

#### 3.3.1.5 光谱衰减

#### 3.3.1.6 梅尔频率倒谱系数

### 3.3.2 统计特征提取

#### 3.3.2.1 时域特征（waveform）

**含量纲的时域特征**

音频信号中含量纲的时域特征，常用的有十个，其中包括最大值(maximum)、最小值(minimum)、极差(range)、均值(mean)、中位数(media)、众数(mode)、标准差(standard deviation)、均方根值(root mean square/rms)、均方值(mean square/ms)、k阶中心/原点矩。

**无量纲的时域特征**

音频信号中无量纲的时域特征，分别为偏度(skewness)，峰度(kurtosis)，峰度因子(kurtosis factor)、波形因子(waveform factor)、脉冲因子(pulse factor)、裕度因子(margin factor)。

本次作业中我们选用其中的*均值*、*标准差*、*偏度*和*峰度*四项时域特征：
- 均值：
    - 均值描述的是样本集合的中间点
    - ![](https://math.jianshu.com/math?formula=%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bj%3D0%7D%5E%7Bn-1%7Dz_%7Bij%7D)
- 标准差：
    - 标准差描述的是样本集合的各个样本点到均值的距离之平均
    - ![](https://math.jianshu.com/math?formula=%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bj%3D0%7D%5E%7Bn-1%7D(z_%7Bij%7D-%5Coverline%7Bz_i%7D)%5E2%7D)
- 偏度：
    - 偏度是三阶标准矩
    - ![](https://math.jianshu.com/math?formula=E%5B(%5Cfrac%7Bz_%7Bij%7D-%5Cmu%7D%7B%5Csigma%7D)%5E3%5D)
- 峰度：
    - 峰度是四阶标准矩
    - ![](https://math.jianshu.com/math?formula=E%5B(%5Cfrac%7Bz_%7Bij%7D-%5Cmu%7D%7B%5Csigma%7D)%5E4%5D)

#### 3.3.2.2 频域特征（spectrogram）

频域（频率域）——自变量是频率,即横轴是频率,纵轴是该频率信号的幅度,也就是通常说的频谱图。频谱图描述了信号的频率结构及频率与该频率信号幅度的关系。

对信号进行时域分析时，有时一些信号的时域参数相同，但并不能说明信号就完全相同。因为信号不仅随时间变化，还与频率、相位等信息有关，这就需要进一步分析信号的频率结构，并在频率域中对信号进行描述。动态信号从时间域变换到频率域主要通过傅立叶级数和傅立叶变换实现。周期信号靠傅立叶级数，非周期信号靠傅立叶变换。

下图为我们选用的10个频域特征的公式详情：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E9%A2%91%E5%9F%9F%E7%89%B9%E5%BE%81.png)

#### 3.3.2.3 差分

通常情况下，我们接收到的所有信号都是非平稳信号，比如我们的语音信号就是典型的非平稳信号。而平稳信号指在不同时间得到的采样值的统计特性(比如期望、方差等)是相同的，非平稳信号则与之相反，其特性会随时间变化。在信号处理中，这个特性通常指频率。对于非平稳信号，由于频率特性会随时间变化，为了捕获这一短时时变特性，我们需要对信号进行时频分析。

我们可以使用差分的方式来减轻数据之间的不规律波动，使其波动曲线更平稳，消除线性的趋势因素，得到平稳序列。通俗来说当间距相等时，用下一个数值，减去上一个数值 ，就叫“一阶差分”，做两次相同的动作，即再在一阶差分的基础上用后一个数值再减上一个数值一次，就叫“二阶差分"。

#### 3.3.2.4 快速傅里叶变换（FFT）

傅立叶原理表明：任何连续测量的时序或信号，都可以表示为不同频率的正弦波信号的无限叠加。而根据该原理创立的傅立叶变换算法利用直接测量到的原始信号，以累加方式来计算该信号中不同正弦波信号的频率、振幅和相位。

对于一个输入信号，假如我们不能确定该输入信号的频率组成，我们对其进行FFT处理之后，便可以很轻松的看出其频率分量，并且可以通过简单的计算来获知该信号的幅值信息等。在信号分析过程中，傅里叶变换的作用就是将组成这个回波信号的所有输入源在频域中按照频率的大小来表示出来。傅里叶变换之后，信号的幅度谱可表示对应频率的能量，而相位谱可表示对应频率的相位特征。经过傅立叶变换可以在频率中很容易的找出杂乱信号中各频率分量的幅度谱和相位谱，然后根据需求，进行高通或者低通滤波处理，最终得到所需要频率域的回波。

FFT（Fast Fourier Transformation），中文名快速傅里叶变换，是离散傅氏变换的快速算法，它是根据离散傅氏变换的奇、偶、虚、实等特性，对离散傅立叶变换的算法进行改进获得的。

#### 3.3.2.5 平滑滤波（加窗）

FFT提供了观察信号的新视角，但是FFT也有各种限制，可通过加窗增加信号的清晰度。使用FFT分析信号的频率成分时，分析的是有限的数据集合。 FFT认为波形是一组有限数据的集合，一个连续的波形是由若干段小波形组成的。 对于FFT而言，时域和频域都是环形的拓扑结构。时间上，波形的前后两个端点是相连的。 如测量的信号是周期信号，采集时间内刚好有整数个周期，那么FFT的上述假设合理。

平滑，也可叫滤波，或者合在一起叫平滑滤波，平滑滤波是低频增强的空间域滤波技术。它的目的有两类：一类是模糊；另一类是消除噪音。空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。

我们主要通过平滑窗的选用来降低音频信号的噪声，并且显示音频信号在不同时间尺度上的特征值表现。

## 3.4 数据降维及归一化

### 3.4.1 PCA主成分分析

### 3.4.2 数据归一化

## 3.5 聚类算法

聚类(Clustering)是按照某个特定标准(如距离)把一个数据集分割成不同的类或簇，使得同一个簇内的数据对象的相似性尽可能大，同时不在同一个簇中的数据对象的差异性也尽可能地大。也即聚类后同一类的数据尽可能聚集到一起，不同类数据尽量分离。

聚类属于无监督学习，目的是将具有相关性的数据聚合在一起而并不关心这些数据的标签。

### 3.5.1 K-Means聚类

K-Means聚类算法属于最基本的划分式聚类方法，需要事先指定簇类的数目或者聚类中心，通过反复迭代，直至最后达到"簇内的点足够近，簇间的点足够远"的目标。基于原型的、划分的距离技术，它试图发现用户指定个数(K)的簇。

算法原理：
```
选择K个点作为初始质心  
repeat  
    将每个点指派到最近的质心，形成K个簇  
    重新计算每个簇的质心  
until 簇不发生变化或达到最大迭代次数  
```

特点：
- 需要提前确定k值
- 对初始质心点敏感
- 对异常数据敏感

k均值算法非常简单且使用广泛，但是其也有一些缺陷：
1. K值需要预先给定，属于预先知识，很多情况下K值的估计是非常困难的，对于像计算全部微信用户的交往圈这样的场景就完全的没办法用K-Means进行。对于可以确定K值不会太大但不明确精确的K值的场景，可以进行迭代运算，然后找出Cost Function最小时所对应的K值，这个值往往能较好的描述有多少个簇类。
2. K-Means算法对初始选取的聚类中心点是敏感的，不同的随机种子点得到的聚类结果完全不同
3. K均值算法并不是很所有的数据类型。它不能处理非球形簇、不同尺寸和不同密度的簇，银冠指定足够大的簇的个数是他通常可以发现纯子簇。
4. 对离群点的数据进行聚类时，K均值也有问题，这种情况下，离群点检测和删除有很大的帮助。

### 3.5.2 二分K-均值(bisecting K-means)

为了克服K-Means算法收敛于局部最小值的问题，优化出了二分k-Means算法：一种度量聚类效果的指标是SSE(Sum of Squared Error)，他表示聚类后的簇离该簇的聚类中心的平方和，SSE越小，表示聚类效果越好。 bi-kmeans是针对kmeans算法会陷入局部最优的缺陷进行的改进算法。该算法基于SSE最小化的原理，首先将所有的数据点视为一个簇，然后将该簇一分为二，之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否能最大程度的降低SSE的值。

### 3.5.3 MiniBatch k-Means

在原始的K-means算法中，每一次的划分所有的样本都要参与运算，如果数据量非常大的话，这个时间是非常高的，因此有了一种分批处理的改进算法。
使用Mini Batch（分批处理）的方法对数据点之间的距离进行计算。
Mini Batch的好处：不必使用所有的数据样本，而是从不同类别的样本中抽取一部分样本来代表各自类型进行计算。n 由于计算样本量少，所以会相应的减少运行时间n 但另一方面抽样也必然会带来准确度的下降。


## 3.6 分类算法

### 3.6.1 HMM算法



#  4. 案例实践

## 4.1 数据爬取以及处理

​	以蝉妈妈数据榜单前50首为例，通过对header的解析，模拟浏览器向服务器发出请求，最终保存音乐以及音乐信息至本地。

​	

## 4.2 波形分析

### 4.2.1 音乐波形图 

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
![](https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E6%B3%A2%E5%BD%A2.png)

### 4.2.2 音乐语谱图

将音乐频谱图中得到的一维矩阵，将该矩阵形成谱图。任取4首效果如下:

**图例**

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E4%BD%A0%E5%95%8A%E4%BD%A0%E5%95%8A.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E5%B1%B1%E5%A6%96.wav.png" width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E5%BE%AE%E5%BE%AE.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1%E5%9B%BE/%E7%88%B1%EF%BC%8C%E5%AD%98%E5%9C%A8.wav.png" width="400"/> 
</figure>
![](https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%AF%AD%E8%B0%B1.png)

### 4.2.3 音乐音调变化图

利用傅里叶变换，将波形拆分成多个正弦曲线。不同的正弦曲线，代表的不同音调的波形，所以我们无法获取绝对应高，只能比较时间维度上，相对的音调变化。对音乐帧进行分段，8000个帧为一小段，进行快速傅里叶变换，并对分解的曲线进行对应其响度上的加权，最终获得该小段上的相对音高。最终将这些音高做成曲线。任意取四首效果如下

**图例**

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/Crying%20Over%20You.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/DancingWithYourGhost.wav.png"width="400"/> 
</figure>

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E5%8F%AE%E5%8F%AE%E5%8F%AE.wav.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83%E5%9B%BE/%E6%88%91%E5%BF%83%E9%87%8C%E7%9A%84%E7%A7%98%E5%AF%86.wav.png" width="400"/> 
</figure>

![](https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E8%B0%83.png)

### 4.2.4 音乐自相似矩阵图

读取midi文件,根据其音调信息生成谱图

根据谱图对角线方块颜色的变化，我们可以看出音乐音调变化相似性，从自相似矩阵中，我们可以验证音乐本身所具有的节奏感，旋律感。

**图例**

<figure class="half">     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%98%B5/DancingWithYourGhost.mid.png" width="400"/ >     
    <img src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%98%B5/%E4%BD%A0%E5%95%8A%E4%BD%A0%E5%95%8A.mid.png" width="400"/> 
</figure>
![](https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E8%87%AA%E7%9B%B8%E4%BC%BC.png)

## 4.3 音乐特征分析聚类

> ​	在之前的波形特征提取部分，我们研究了不同音乐间的歌的**波形结构**，但我们发现即使同一名歌手演唱的同一首歌曲的不同段落音频波形结构之间的差异也很大，而我们想要探讨这些抖音热歌之间的共同点以及与非热门歌曲之间的区别，为此我们想到了提取音乐的统计特征并进行**聚类**来探讨他们之间的**关联性**。

### 4.3.1 统计特征提取

我们首先需要做的是波形的特征提取，我们从原始的WAV文件中提取统计要素，我们人为地选取了42个音频特征值：

- **歌曲波形的统计矩**，包括*均值*、*标准差*、*偏态*和*峰态*，同时，我们通过平滑窗(递增平滑，长度分别为1,10,100,1000)来获取这些特征在不同时间尺度上的表现；
- 为了体现信号的短时变化，我们可以计算一下**波形一阶差分幅度的统计矩**，同样也通过平滑窗来获取这些特征(*均值*、*标准差*、*偏态*和*峰态*)在不同时间尺度上的表现；
- 最后，我们计算一下**波形的频域特征**，这里我们只计算歌曲在不同*频段*(将整个频段均分为10份)的能量占比，不过直接对歌曲的波形数据作快速傅里叶变换的话其计算量过于庞大了，因此先让波形数据通过长度为5的平滑窗再对其作快速傅里叶变换。

最终得出如下42个特征值：

![42个特征值示例](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/42%E7%89%B9%E5%BE%81%E5%80%BC%E7%A4%BA%E4%BE%8B.png)

```python
amp1mean amp1std amp1skew amp1kurt
amp1dmean amp1dstd amp1dskew amp1dkurt
amp10mean amp10std amp10skew amp10kurt
amp10dmean amp10dstd amp10dskew amp10dkurt
amp100mean amp100std amp100skew amp100kurt
amp100dmean amp100dstd amp100dskew amp100dkurt
amp1000mean amp1000std amp1000skew amp1000kurt
amp1000dmean amp1000dstd amp1000dskew amp1000dkurt
power1 power2 power3 power4 power5
power6 power7 power8 power9 power10
```

### 4.3.2 特征选择

然而在使用这些特征值的过程中我们发现，由于这些特征值是人为提取的，所以并不能很好地表现出歌曲特征，并且这些特征之间的相关系数是不为0的，也就是说存在冗余特征，因此我们需要对特征值进行筛选。

我们从每首歌曲中抽取两个15s 的样本，并尝试找到一种算法，该算法可以最好地将每首歌曲中的两个样本匹配在一起。

为了找到在所有歌曲中提供最佳平均匹配特征值子集，我们使用**遗传算法（R 中的 genalg 包）**来控制42 个特征值中的每个特征值。下面的图显示了遗传算法超过 100 代的目标函数（即歌曲的两个样本由最近的邻域分类器匹配在一起）的结果。

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95%E7%AD%9B%E9%80%89%E7%89%B9%E5%BE%81%E5%80%BC.png)

如果我们选用全部42个特征值，那么目标函数的错误率会达到275，而通过筛选上图中右侧的18个特征值，我们能够将目标函数的错误率在100次以上迭代中降低到90，这是一个重大的优化。

通过对42个特征值数组的提取重组，我们可以得到音乐的18个特征值向量(上图右侧)：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/18%E7%89%B9%E5%BE%81%E5%80%BC%E7%A4%BA%E4%BE%8B.png)

### 4.3.3 归一化和降维处理

为了最后服务于我们的歌曲聚类目标以及最后结果的可视化，我们需要将这些数据进行归一化处理和降维处理。

使用StandardScaler进行去均值和方差归一化，调用sklearn的preprocessing库进行数据的归一化。

使用PCA主成分分析方法进行数据降维，将18维数据降为2维数据，调用sklearn的decomposition进行主成分分析降维。

最终处理结束后绘制数据分布图：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE%E7%BB%98%E5%9B%BE.png)

### 4.3.4 K-Means聚类

为了分析这些音乐数据之间的关联关系，我们使用了无监督分类的方式试图找到他们之间的关联关系。

#### 4.3.4.1 传统K-Means算法

KMeans算法通过尝试在等方差组中分离样本来对数据进行聚类，从而最小化称为惯性或聚类内平方和的标准。此算法要求指定群集数。它可很好地扩展到大量的样品中，并已被广泛应用于许多不同的领域。k-means算法将一组样本划分为不相交的聚类，每个样本都用群集中样本的均值描述。这些手段通常称为聚类"中心"。

我们绘制了k=5-k=13时的聚类结果：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/KMeans.png)

#### 4.3.4.2 MiniBatchKMeans算法

MiniBatchKMeans是KMeans算法的变体，该算法使用小型批处理来缩短计算时间，同时仍在尝试优化相同的目标函数。小型批处理是输入数据的子集，在每个训练迭代中随机采样。这些小型批处理大大减少了收敛到本地解决方案所需的计算量。与其他减少 k-means 收敛时间的算法相比，小批 k-means 产生的结果通常只比标准算法稍差。

我们绘制了k=5-13时的聚类结果：
![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/MiniBatchKMeans.png)

#### 4.3.4.3 选择K值

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

#### 4.3.4.4 聚类结果
最后，我们可以根据聚类结果生成音乐信息分类情况，为我们后续的热门音乐流派分析提供对比集(以数量最多的一类为示例)：

```
class 6 [数量：76]：['来吧开整', 'waiting for love', '你长这样谁要', '叮叮叮', 'chuchacha', '将会面对什么样的结局', '花开的时候你就来看我', '酒醉的蝴蝶', '我怎么这么好看', '最美的期待', '学猫叫-完整剪辑版', '小可爱 （我的小可爱 今天有没有乖）', '123我爱你', '听我说谢谢你－李昕融', '就是不想长大！', '学猫叫-剪辑版', '火山用户创作的原声', '黄梅戏-慕容晓晓', '殇雪（剪辑版）', '三月里的小雨_王恰恰', 'Make Some TikTok', '生日快乐', '@德华饰品创作的原声', '一曲红尘', '@我是苏朋友创作的原声', '宝贝宝贝我爱你（剪辑版）', '天在下雨我在想你', '一路向北-剪辑版', '捉泥鳅', '@✨🌟淡然一笑🌟✨创作的原声', '金久哲 - 干就完了（剪切版）', '你会爱我到什么时候-剪辑版', '敖包相会', '男人要有担当', '小小的太陽', '千年等一回', '站着等你三千年-高潮版', '醉千年', '女人是世界最美丽的花（剪辑版）', '@开心每一天💋创作的原声', 'Wrap Me In Plastic', '嘴巴嘟嘟-剪辑版', 'Delícia Tchu Tcha Tcha - Dj King Remix', '狂浪-花姐', '皮一下很开心(剪辑版)', '用户创作的原声~3', '@天启体验教育创作的原声', 'Getaway ', '多想抱抱你', '小星星~1', '小白兔遇上卡布奇诺', '爱情让我心痛DJ版-赵小南', '确认过眼神（撩妹撩汉版）', '你个小坏坏', '阳光彩虹小白马', '像个孩子', '小奶狗', '她扒拉我', 'MyLove一起去旅行', '我不会唱歌_主歌版', '习惯你的好', '小棉袄儿（剪辑版）', '爱你三千遍（剪辑版）', '点歌的人-海来阿木', '歌曲来自柠檬啊', '@宾县＿＿仓少创作的原声', '忘情牛肉面', '皓皓最棒', '我带上你', '隔壁泰山(part.2)', '@Z小仙女创作的原声~1', '再见彩霞', '生日快乐~1', '就是这个范儿', '姐妹', '你打不过我吧']
```

## 4.4 音乐分类

音乐分类结果：

![](https://zzn-normal.oss-cn-beijing.aliyuncs.com/%E5%AD%A6%E4%B9%A0/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%9F%BA%E7%A1%80-%E6%8A%96%E9%9F%B3%E5%88%86%E6%9E%90/%E9%9F%B3%E4%B9%90%E5%88%86%E7%B1%BB.png)

## 4.5 midi文件生成

根据读取wav文件的帧，对其进行傅里叶变换获取音高信息，最终导入mid文件

​	**山妖原音乐**

​	**下载链接：**

https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96.wav

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96.wav"> </audio>
​	**山妖mid文件音乐**

​	**下载链接：**

https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96mp32mid.mp3

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E9%9F%B3%E4%B9%90/%E5%B1%B1%E5%A6%96mp32mid.mp3"> </audio>

根据不同的wav文件，我们综合其属性和特征，融合了属于自己的wav

​	**wav融合**

​	**下载链接：**

https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E7%BB%9D%E6%B4%BB%E7%BB%88%E6%9E%81.wav

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E7%BB%9D%E6%B4%BB%E7%BB%88%E6%9E%81.wav"> </audio>

但是令人遗憾的是，这一曲子太过嘈杂，我们认为，这是因为在形成文件时，我们著重音乐波形和音调的变换，而忽略音乐本身的旋律性，缺少了节奏。因此我们通过midi映射，生成对应音调变化文件。

 	**最终音调**

​	**下载链接：**

https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E6%9C%80%E7%BB%88.mp3

<audio id="audio" controls="" preload="none"> <source  src="https://umlpicture.oss-cn-shanghai.aliyuncs.com/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%A4%A7%E4%BD%9C%E4%B8%9A/%E6%9C%80%E7%BB%88.mp3"> </audio>

这就是最终生成的音调变化，我们期望以该旋律为基础，加上适当旋律和和声，生成一首该项目的抖音BGM

# 5. 结论

# 6. 反思与不足

只研究了声学三要素相关的知识，没有研究音乐三要素（旋律，和声，节奏）相关的内容，并没有直接的midi文件，没有生成音乐

# 7. 参考文献

[R语言中的遗传算法](http://blog.fens.me/algorithm-ga-r/)

[sklearn聚类算法官网](https://scikit-learn.org/stable/modules/clustering.html#)

[音乐收藏分析](https://www.christianpeccei.com/musicmap/)

 [基于神经网络的音乐流派分类](https://medium.com/@navdeepsingh_2336/identifying-the-genre-of-a-song-with-neural-networks-851db89c42f0)



