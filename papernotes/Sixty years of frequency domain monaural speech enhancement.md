---
title: paper note
author: xmj
---



>主要介绍了频率域单耳语音增强 60 年的发展。

## 简介

单耳语音增强的目的是提取干净的语音或通过从麦克风捕获的噪声混响语音信号中去除噪声和混响来提高语音与背景的比率。

单耳语音增强的方法有时域和频域方法，但是频域方法是最广泛研究的，原因如下：

1. 频谱分析可以利用 FFT 高效完成，即使傅里叶分析可能并不是语音增强的最优方法
2. 可以解耦语音的幅度和相位，然后可以根据硬件条件和预期性能进行处理
3. 人类的耳朵类似于频率分析器
4. 语音在频域稀疏，有助于消除非语音部分（非语音部分在时域并不稀疏）
5. 频域方法的效果比时域方法的效果更好

语音增强的深度学习方法可以分为幅度增强和复频谱增强。前者只估计干净语音的频谱幅度。利用噪声相位重构时域语音信号，后者直接估计干净语音的复频谱的实部和虚部，具有进一步提高语音质量的潜力。虽然深度学习方法效果更好，但是对于计算资源和储存的要求高，所以在一些资源受限的设备上，传统的语音增强方法还是大量应用。

## 信号模型与问题描述

单耳语音的信号模型
$$
y(t)=s(t)*h(t)+v(t)
$$
s(t) 为干净语音，h(t) 为从干净语音到麦克风的传输函数，v(t) 为噪声。当麦克风接收到语音时，信号包括直接和早反射和晚反射（经过多次反射）语音分量。通常认为，早反射有助于语音的可理解性，而晚反射会降低语音质量和可理解性。

通过短时傅里叶变换 STFT，得到信号模型的时频表示
$$
Y(k,l) = S(k,l)H(k,l) + V(k,l)
$$
k 表示频域索引，l 表示帧（时域）索引，Y(k,l) 的计算公式如下
$$
Y(k,l) = \sum\limits_{\mu  = 0}^{K - 1} {y(lR + \mu )w(\mu ){e^{ - j{{2\pi k\mu } \over K}}}}
$$
w(t) 是一个窗口，R 是帧偏移，K 是帧长。

基于假设晚混响语音分量与直接和早反射语音分量不相关，许多语音降噪方法已经扩展到实现语音反混响。当语速足够快时，这个假设是合理的，麦克风上给定的语音在该声音的晚混响到达麦克风之前就已经完成了。即使脉冲响应 h(t) 随时间变化，这个假设也可能是合理的。这样的话单耳语音信号模型中的 H(k,l) 可以视为 1。

对于频域单耳语音增强方法，目标是从 Y(k, l) 中估计出干净语音 $\hat S(k,l)$，时域信号 $\hat s(t)$ 可以通过逆 STFT 重建。有许多种方法用来获得 $\hat S(k,l)$，可以大致分为两类，间接和直接两类。几乎所有的传统方法和一些深度学习方法都属于第一类。对于间接方法，先估计一个频谱增益函数 G(k,l) 或者基于噪声观测的网络映射，再乘上 Y(k,l)，如
$$
\hat S(k,l) = G(k,l)Y(k,l)
$$
通常，这些方法只估计干净语音频谱的幅度，G(k,l) 的值域是 [0, 1]，噪声相位不变，用于时域语音重构。

最近的研究确定了最小化估计语音和干净语音之间相位差异的重要性，但是很难在低信噪比的情况下估计干净语音的相位。在近些年，一些直接方法被提出，使用深层复杂网络直接训练出 $\hat S(k,l)$ 的幅度或者实部和虚部。



## 传统方法

基于统计信号处理的单耳语音增强方法通常基于语音与噪声独立的假设，一些学者假设语音是一种随机—确定性的组合信号，而不是完全随机的信号，但是也有许多启发式方法，将语音假设为确定性信号。对于所有这些传统方法，通常使用五个关键模块，即噪声估计、先验信噪比估计、语音存在概率估计、频谱增益估计和相位估计。

### 噪声估计

噪声估计模块几乎在所有的传统频域语音增强方法中发挥重要作用，噪声估计的性能直接影响到噪声消除和语音失真。当噪声能量谱密度 PSD 欠估计时，会导致语音幅度失真，当噪声 PSD 过估计时，会导致语音衰减失真。

早期的噪声估计基于语音活动检测 VAD，将每个时间帧分成 speech-absent 和 speech-present 这两个状态，噪声的 PSD 在 speech-absent 状态更新：
$$
\sigma _v^2(k,l) = \left\{ \matrix{
  {\alpha _v}\sigma _v^2(k,l - 1) + (1 - {\alpha _v}){\left| {Y(k,l)} \right|^2}\quad SA \hfill \cr 
  \sigma _v^2(k,l - 1)\qquad SP \hfill \cr}  \right.
$$
$\alpha_v\in(0,1)$ 为平滑因子，SA 表示 speech absent，SP 表示 speech present。基于 VAD 的噪声 PSD 估计器存在一些缺点，噪声估计准确率严重依赖 VAD 性能，时间帧 speech-absent 和 speech-present 状态的错误区分会造成噪声的过估计。

如果噪声 PSD 通过 VAD 来估计，则假设噪声是似稳态的，PSD 随着时间缓慢变化。为了避免使用 VAD，还有一个假设是必要的，即 $E\left\{ {{{\left| {Y(k,l)} \right|}^2}} \right\} \ge E\left\{ {{{\left| {V(k,l)} \right|}^2}} \right\}$（这个假设始终正确）。鉴于此，学者提出了一种递归方法在不需要 VAD 的情况下估计噪声幅度
$$
{\sigma _v}(k,l) = \left\{ \matrix{
  {\alpha _v}{\sigma _v}(k,l - 1) + (1 - {\alpha _v})\left| {Y(k,l)} \right|\quad if\;\left| {Y(k,l)} \right| \le {\beta _v}{\sigma _v}(k,l - 1) \hfill \cr 
  \sigma _v^2(k,l - 1)\qquad {\rm{otherwise}} \hfill \cr}  \right.
$$
$\beta_v$​ 实际应用时取值在 1.5 到 2.5 之间。

还有更多的噪声估计参见原论文。



### 先验 SNR 估计

先验 SNR 使用最大似然方法进行估计
$$
{\xi _{ML}}(k,l) = \max \left\{ {\gamma (k,l) - {\beta _y},0} \right\}
$$
和
$$
{\xi _{DD}}(k,l) = {\alpha _\xi }{{{{\left| {\hat S(k,l - 1)} \right|}^2}} \over {\sigma _v^2(k,l)}} + (1 - {\alpha _\xi })\max \left\{ {\gamma (k,l) - {\beta _{\gamma}},0} \right\}
$$


其中，$\beta_{\gamma}$ 常设置为 1，${\hat S(k,l - 1)}$ 是上一帧估计出的复语音谱，平滑因子 ${\alpha _\xi }$ 非常接近 1。当  ${\alpha _\xi }=0$时，${\xi _{DD}}(k,l)$ 退化为 ${\xi _{ML}}(k,l)$，${\gamma (k,l)}$ 为后验 SNR
$$
\gamma (k,l) = {{{{\left| {Y(k,l)} \right|}^2}} \over {\sigma _v^2(k,l)}}
$$


上面这种方法称为判决引导方法，存在两个缺点，在语音开始时需要一帧的延迟来跟踪先验 SNR，从而导致语音失真。其次，需要一帧的延迟来跟踪语音偏移处的后验 SNR。为了解决这个这两个缺点，有学者提出了一个双阶段 SNR 估计器
$$
{\xi _{TSNR}}(k,l) = G_{{H_1}}^2(k,l)\gamma (k,l)
$$
其中 ${G_{{H_1}}}(k,l) = {\xi _{DD}}(k,l)/\left( {1 + {\xi _{DD}}(k,l)} \right)$​。在双阶段 SNR 估计器的第一阶段，使用判决引导方法粗略估计先验 SNR，然后使用估计的先验 SNR 计算维纳滤波器增益，以对第二阶段的后验信噪比进行加权。

上述方法都需要对后验 SNR 进行初始估计，如果后验 SNR 估计错误，那么先验 SNR 也会估计错误。有两种方法，一是使用数据驱动方法提高噪声 PSD 的估计精度，而是使用数据驱动方法直接估计先验 SNR。





### 语音存在概率估计

语音存在概率 SPP 依靠三个参数：后验 SNR $\gamma(k,l)$，先验SNR $\xi(k,l)$ 和语音缺失的先验概率 q(k,l)，需要对每个频率段和帧进行估计这三个参数。

语音缺失的先验概率定义为 $q(k,l)=P(H_0(k,l))$，一般来说每个时频块的 q(k,l) 都不同，但在初始化的时候，都是设置为一个常数，q(k,l) 的估计如下
$$
q(k,l) = {a_q}q(k,l - 1) + \left( {1 - {a_q}} \right)I(k,l)
$$
其中，$a_q$ 是平滑因子，当 $H_0(k,l)$ 为真时，$I(k,l)$ 为 1，当 $H_1(k,l)$ 为真时，$I(k,l)$ 为 0。SPP 可由下式估计
$$
p(k,l) = {\left( {1 + {{q(k,l)} \over {1 - q(k,l)}}\left( {1 + \xi (k,l)} \right)\exp \left( { - {{\xi (k,l)\gamma (k,l)} \over {1 + \xi (k,l)}}} \right)} \right)^{ - 1}}
$$




### 谱增益估计

谱增益估计共有三类方法，确定的、随机的和随机-确定的。



**确定方法**：第一类方法基于语音与噪声独立推导的谱增益。由于语音高度非平稳，导致相应的噪声语音具有非平稳特征，因此只需使用非常有限的帧数来估计 $E\left\{ {{{\left| {S(k,l)} \right|}^2}} \right\}$ 和 $E\left\{ {{{\left| {Y(k,l)} \right|}^2}} \right\}$。另一方面，噪声通过假定为平稳或者似稳定的，所以噪声的 PSD 不会快速变换，所以有更多的帧数可以用来估计 $E\left\{ {{{\left| {V(k,l)} \right|}^2}} \right\}$，可以得到下面的近似等式
$$
{\left| {Y(k,l)} \right|^{{\alpha _g}}} \approx {\left| {S(k,l)} \right|^{{\alpha _g}}} + \sigma _v^{{\alpha _g}}(k,l)
$$
使用上面的等式，可以推导
$$
\left| {S(k,l)} \right| = {\left( {\max \left\{ {{{\left| {Y(k,l)} \right|}^{{\alpha _g}}} - {\beta _g}\sigma _v^{{\alpha _g}}(k,l),0} \right\}} \right)^{1/{\alpha _g}}}
$$
$\alpha_g$ 的不同取值有不同的效果，详见原论文。借助上式可以将谱增益写为
$$
G(k,l) = {{\max \left\{ {{{\left| {Y(k,l)} \right|}^{{\alpha _g}}} - {\beta _g}\sigma _v^{{\alpha _g}}(k,l),0} \right\}} \over {{{\left| {Y(k,l)} \right|}^{{\alpha _g}}}}} = {{\max \left\{ {{\gamma ^{{\alpha _g}/2}}(k,l) - {\beta _g},0} \right\}} \over {{\gamma ^{{\alpha _g}/2}}(k,l)}}
$$


**随机方法**：假设语音和噪声是统计独立的，将语音和噪声的实部和虚部的复频域建模为统计独立的高斯随机变量。频谱增益给定为
$$
{G_{{H_1}}}(k,l) = \Gamma (1.5){{\sqrt {v(k,l)} } \over {\gamma (k,l)}}M( - 0.5;1, - v(k,l))
$$


**确定-随机方法**：语音可以线性预测为
$$
s(t) = \sum\limits_{{t_0} = 1}^{{t_P}} {a({t_0})s(t - {t_0}) + e(t)} 
$$
e(t) 是激励信号，对于语音段，e(t) 是一个周期性脉冲或者锯齿波，非语音段，e(t) 可以被建模成高斯随机信号。由于 s(t) 不是一个完全随机的信号，对语音使用这种确定性-随机模型可能会更好。



### 相位处理

从带噪声的语音中估计干净语音的相位是一个困难的事情，特别是 SNR 很低时。Gerkmann等人(2015)对基于相位处理的单自然语音增强方法进行了全面概述。





### 讨论

对于绝大多数传统频域语音增强算法，有四条潜在假设

1. 语音和噪声在统计上是独立的
2. 噪声比语音更加平稳
3. 每个时频块与其它时频块都是统计独立的
4. 语音的幅度比语音的相位更加重要

除了第一条，剩下三条并不是特别有道理。





## 深度学习方法



### 特征提取

MFCC、GFCC（gammatone frequency cepstral coefficients）、LOG-AMP（log-amplitude spectral features）（感觉可以直接使用 STFT 后生成语谱图）

<img src="https://cdn.jsdelivr.net/gh/Xiang-M-J/MyPic@img/img/image-20240406155523532.png" alt="image-20240406155523532" style="zoom:67%;" />

使用短时目标可理解性（short-time objective intelligibility，STOI）分数作为目标性能指标。Gammatone 特征比 LOG-MAP，对数梅尔滤波器特征和 MFCC 有更高的 STOI 分数，但是计算量较大。



一些容易提取的特征已经用于深层神经网络，如 LOG-AMP，频谱幅度，取幂运算后的频谱幅度（幂取小于 1 的数，可以看成一种幅度压缩），频谱幅度的三次方根一般会有更好的性能，可能是因为三次方根减少了语音的动态范围，有利于训练。还有学者将复频谱的实部和虚部作为输入特征，对应干净语音复频谱的实部和虚部，这样的训练效果更好，因为隐含了相位信息。



STFT 域特征的提取在计算时是高效的，所以它们的计算负担相对于深度神经网络本身的负担来说通常是很小的。然而，随着帧长的增加，特征的数量也会增加，而为了便于区分语音和噪声，良好的频率分辨率需要较大的帧长。普遍认为，人类听觉系统的频率分辨率可以用 ERB-Number 频率标度来表征，其中听觉滤波器带宽随着中心频率的增加而增加。Bark尺度具有类似的特性，但在低频分辨率上与 ERB-Number 尺度不同。这些基于心理声学的尺度已被用于减少stft域特征的数量。如使用 22 维的 Bark-frequency cepstral coefficients（BFCC）加上前 6 个 BFCCs 的一阶导和二阶导和前 6 个波段的优势周期性强度作为输入特征，这样可以大幅减少特征维度。

使用基于感知的特征的另一个优点出现在要增强全频带语音时。对于全频带语音，在线性频率尺度上提取特征时，输入特征的数量会显著增加，导致计算复杂度大大增加。输入特征的数量可以通过在对数频率尺度上提取它们来减少，或者通过使用基于 ERB 或基于 bark 的滤波器组来提取输入特征。



基于 F0 的特征被广泛用于分离多个说话人同时说话的场景，但在低信噪比下，由于对基于 F0 的特征估计不准确，它们在噪声中增强语音的效果很差。



虽然有许多种可选的特征，但是幅度压缩和相位被认为是重要的特征，所以压缩的复频谱应用广泛。复幅度的压缩可以表示为
$$
\left| {{Y_{cp}}(k,l)} \right| = {\left| {Y(k,l)} \right|^{{\alpha _{cp}}}}
$$
其中 $\alpha_{cp} \in (0,1]$，一般为 1/2 或者 1/3。



### 网络架构

早期的在时频域上操作的深度学习语音增强方法只增强幅度谱而不改变相谱，基于RNN的模型可以更好捕捉时间层面上的特征，卷积递归网络（CRNs）是一种非常受欢迎的处理语音的结构，如 GCRNs 使用两个 CRNs 来分别估计复频谱的实部和虚部。DPCRN 将 RNN 换成双路径 RNN，这种双路径 RNN 包含一个块内 RNN，用于模拟单个时间帧的频谱，以及一个块间 RNN，用于建模频谱随时间的变化。



UNet，SDDNet



### 训练目标

训练目标可以分为两类，基于 masking 和基于 mapping。

一种基于 masking 的目标是 ideal binary mask（IBM），对于每个时频块，IBM 的值为 0 或 1。值为1表示该块的估计信噪比大于预定义的阈值，反之值为 0 表示该块的估计信噪比小于预定义的阈值。IBM 应用到帧上，可以高效选择需要保留的时频块
$$
IBM(k,l) = \left\{ \matrix{
  1\quad \left| {S(k,l)} \right| \ge {\theta _{th}}\left| {V(k,l)} \right| \hfill \cr 
  0\quad {\rm{othervise}} \hfill \cr}  \right.
$$
这里的 $\theta_{th}$ 是门限，一般为 0.5 到 1。IBM 将每个时频块标记为目标主导或噪声主导，使用 IBM 可以获得良好的语音清晰度，但语音质量一般。除了 IBM 这种硬门限，可以使用 ideal ratio mask（IRM）对每个时频块应用一个衰减，随着信道估计信噪比的降低而增加，IRM 定义为
$$
IRM(k,l) = {\left( {{{{{\left| {S(k,l)} \right|}^{{\alpha _{irm}}}}} \over {{{\left| {S(k,l)} \right|}^{{\alpha _{irm}}}} + {{\left| {V(k,l)} \right|}^{{\alpha _{irm}}}}}}} \right)^{{\beta _{irm}}}}
$$
这里的 ${{\alpha _{irm}}}\ge 0$ 和 ${{\beta _{irm}}} \ge 0$ 是可调参数，${{\beta _{irm}}}$ 一般取 0.5，IRM 相比 IBM 能够获得更好的语音质量。

IBM 和 IRM 都应用于幅度，phase-sensitive mask（PSM）考虑了相位
$$
PSM(k,l) = {{\left| {S(k,l)} \right|} \over {\left| {Y(k,l)} \right|}}\cos {\Phi _\Delta }
$$
$\Phi_{\Delta}$​ 表示带噪语音和干净语音之间的相位差异，在训练目标中引入相位差异可以让增强语音有更高的 SNR。

将 IRM 扩展到复数域，可以得到 cIRM
$$
cIRM(k,l) = {M_r}(k,l) + i{M_i}(k,l)
$$
这里的 $M_r(k,l)$ 和 $M_i(k,l)$ 分别为
$$
{M_r}(k,l) = {{{Y_r}(k,l){S_r}(k,l) + {Y_i}(k,l){S_i}(k,l)} \over {Y_r^2(k,l) + Y_i^2(k,l)}}
$$

$$
{M_i}(k,l) = {{{Y_r}(k,l){S_i}(k,l) - {Y_i}(k,l){S_r}(k,l)} \over {Y_r^2(k,l) + Y_i^2(k,l)}}
$$





基于 mapping 的目标，使用预训练的深度学习模型直接映射干净语音的频谱。将频谱 S(k,l) 映射为压缩复频谱 $S_{cp}(k,l)$。



### 损失函数

损失函数可以分成三类：频域、时域和基于感知



对于频域，早期的工作使用一个神经网络来映射 IBM，每个时频块经过 IBM 映射后变为 0 或 1，然后使用二元交叉熵衡量预测值与真实值之间的差异。但是语音增强更类似回归任务，可以使用 KL 散度和 MSE（效果更好），所以很多模型通过最小化干净语音频谱和预测频谱之间的MSE来优化模型。通过信号近似可以获得更好的度量指标，如基于频谱幅度的 MSE，相位敏感的 MSE 和基于复频谱的 MSE。基于谱幅度的 MSE 和基于复频谱的 MSE 损失函数可表示为
$$
{L_{Mag}} = \left\| {\left| S \right| - \left| {\hat S} \right|} \right\|_F^2
$$
和
$$
{L_{RI}} = \left\| {{S_r} - {{\hat S}_r}} \right\|_F^2 + \left\| {{S_i} - {{\hat S}_i}} \right\|_F^2
$$
其中，$\left\|  \cdot  \right\|_F$ 表示 Frobenius 范数（平方和的平方根），还可以将这两个损失函数结合起来。基于假设每个 STFT 块的实部和虚部服从拉普拉斯分布，可以使用估计值和干净声音幅度之间的平均绝对误差（MAE）作为复频谱的距离度量。由于人耳的声音感知大致符合对数尺度，因此还有学者使用 log-spectral MSE 损失函数，还有 power-law-compressed spectral MSE。



时域的损失函数一般用于分离多个说话人语音，已经被拓展到语音增强，并且有效。除了时域 MAE，还有信号能量损失函数如 SDR 和 scale invariant SDR。此外，还有学者引入受限 SNR 损失来保持干净语音、中间语音和输入带噪语音的相对值。



基于感知的损失函数有 PESQ，cepstral distance 和 STOI。当单独使用时，这些损失函数在与用于优化的度量一起评估时通常会获得更好的性能，但对于其他度量则不会，而将这些损失函数结合使用通常会同时提高多个目标度量分数。



## 混合方法

传统方法不能完全抑制非平稳噪声，因为噪声 PSD 不能准确跟踪，特别是噪声 PSD 快速波动。一种直接方法使用数据驱动的方法来提升噪声跟踪性能，另一种方式是联合估计语音 PSD 和噪声 PSD。混合方法使用 DNN 来替换传统方法中的关键模块，如将先验 SNR 使用 Deep Xi DNN 替代，将其集成在 MMSE-STSA 估计器。另一种方法是结合 DNNs 和 NMF（非负矩阵分解）。





## 评估

### 数据集

WSJ + DNS：WSJ0-SI84 和 Interspeech 2020 DNS-Challenge 噪声数据集，包含了 150,000 条语音与噪声的混合数据及对应的干净语音。干净语音来自 WSJ0-SI84 数据集，噪声片段包括了 20,000 种噪声类型，来自 Interspeech 2020 DNS-Challenge 数据集。



VoiceBank + DEMAND ：可以用于训练语音增强算法和 TTS 模型

kaggle数据集：[VoiceBank_DEMAND_16k (kaggle.com)](https://www.kaggle.com/datasets/jweiqi/voicebank-demand-16k?select=clean_trainset_28spk_wav)

官方数据集：[Noisy speech database for training speech enhancement algorithms and TTS models (ed.ac.uk)](https://datashare.ed.ac.uk/handle/10283/2791)



#### 参数值

发音片段重采样为 16 kHz，窗口大小为 20ms，重叠长度为 10 ms，FFT 长度为 320。使用 Adam 优化器（$\beta_1=0.9, \beta_2=0.999$），学习率初始为0.001，如果验证集损失连续三次上升，学习率降为一半，如果验证集损失连续 5 次上升，停止训练。一共训练 60 个epoch，batch size 设为 16。幅度压缩的幂 $\alpha_{cp}=0.5$



#### 方法评估

用于评估的 DNNs 分为三类，基于幅度谱，基于复谱和解耦，所有的模型均使用带噪语音的 stft 谱或者压缩后的谱作为输入，spectral MSE 损失作为损失函数。

+ 基于幅度谱：LSTM，FullSubNet（两个RNN），CRN（卷积+RNN）
+ 基于复谱：GCRN（CNN+RNN+CNN）、DPCRN（CNN+RNN+CNN）、Uformer、DCCRN、DCCRN(SNR)
+ 解耦：CTSNet、GaGNet、TaylorSENet



选择了 6 种代表性的传统方法，包括 MMSE-STSA，MMSE-LSA、β-order MMSE-STSA（β=0.5）、magnitude spectral subtraction（MSS）、power spectral subtraction（PSS）和 MSS 的平方根（SQ-MSS）。



选择了 2 种混合的方法

+ DeepXi-LSA：使用 Deep Xi 框架估计先验 SNR，后验 SNR 则为估计的先验 SNR 加 1
+ DeepXi-STSA，



#### 评价指标

使用了四种指标评价对于正常听力人群的语音质量，即 PESQ、扩展 STOI（ESTOI）、SDR 和 DNSMOS

使用两种指标来评估对于正常听力和听力受损人群的语音质量和清晰度，即 hearing-aid speech quality index (HASQI) version 2 和 hearing-aid speech perception index (HASPI) version 2



PESQ：通常与人类对语音质量的估计高度相关，包括窄带版本（NB-PESQ，取值范围为-0.5-4.5）和宽带版本（WB-PESQ，取值范围为1.0-4.5）

ESTOI：另一种被广泛用于评估语音清晰度的方法，取值范围为 0-1

SDR：时域指标，被广泛应用于盲语音分离，也可以用于语音质量评估

DNSMOS：稳健语音质量指标，由三个次指标组成，即 DNS-OVL，DNS-SIG 和 DNS-BAK，取值范围为 1-5。

> 上述指标越高表示性能越好



当给定模拟听者的听力阈值时，HASQI 和 HASPI 可分别用于评估模拟正常听力和听力受损听者的语音质量和语音可理解性。对于听力正常的听者，在 HASQI 和 HASPI 要求的所有频率（250、500、1000、2000、4000 和 6000 Hz）下，听力阈值设置为 0 dB HL。对于听力受损的听者，将听力学阈值指定为大于 0 dB HL，并修改听觉模型，以考虑听力损失的一些典型后果，如频率选择性降低和耳蜗压缩减少。HASQI 和 HASPI 的值以百分比表示，范围从0%到100%，分数越高表示性能越好。

pesq 可以通过 pesq 包 

```python
from scipy.io import wavfile
from pesq import pesq

rate, ref = wavfile.read("./audio/speech.wav")
rate, deg = wavfile.read("./audio/speech_bab_0dB.wav")

print(pesq(rate, ref, deg, 'wb'))
print(pesq(rate, ref, deg, 'nb'))

# pesq_batch
```



stoi 可以通过 pystoi 包

```python
import soundfile as sf
from pystoi import stoi

clean, fs = sf.read('path/to/clean/audio')
denoised, fs = sf.read('path/to/denoised/audio')

# Clean and den should have the same length, and be 1D
d = stoi(clean, denoised, fs, extended=False)
```







### 实验

#### CRN





#### DPCRN

