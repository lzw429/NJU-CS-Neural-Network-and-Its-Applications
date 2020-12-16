# 第八讲作业 随机神经网络

舒意恒 MF20330067 计算机科学与技术系

yhshu@smail.nju.edu.cn

[TOC]

## 优化问题

<img src="img/question.png" alt="img.png" style="zoom:67%;" />

## 参数设置

- T_max：最大温度
- T_min：最小温度
- L：链长
- max_stay_counter：冷却耗时



## 模型设置

### 模型 SA

算法概述如下：

```
    u ~ Uniform(0, 1, size = d)
    y = sgn(u - 0.5) * T * ((1 + 1/T)**abs(2*u - 1) - 1.0)

    xc = y * (upper - lower)
    x_new = x_old + xc

    c = n * exp(-n * quench)
    T_new = T0 * exp(-c * k**quench)
```

###  模型 SABoltzmann

算法概述如下：

```
    std = minimum(sqrt(T) * ones(d), (upper - lower) / (3*learn_rate))
    y ~ Normal(0, std, size = d)
    x_new = x_old + learn_rate * y

    T_new = T0 / log(1 + k)
```

其中学习率取 0.5.



## 实验效果

以下实验中，当 x1、x2 没有明显变化时，y 仍然有明显变化。这是因为此时 x 的变化相对于整张图而言较为微小。

### 第一组实验

本组实验关注于最高温度与最低温度两个参数。

**实验 #1 设置与结果**

模型 SA，T_max = 100, T_min = 1e-7, L = 300, max_stay_counter = 150

<img src="img.png" alt="img.png" style="zoom: 40%;" />
<img src="img_1.png" alt="img_1.png" style="zoom:40%;" />

**实验 #2 设置与结果**

模型 SA，T_max = 1, T_min = 1e-7, L = 300, max_stay_counter = 150

<img src="img_2.png" alt="img_2.png" style="zoom:40%;" />
<img src="img_3.png" alt="img_3.png" style="zoom:40%;" />

**实验 #3 设置与结果**

模型 SA，T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 150

<img src="img_4.png" alt="img_4.png" style="zoom:40%;" />
<img src="img_5.png" alt="img_5.png" style="zoom:40%;" />


由第一组实验，`T_max = 1, T_min = 1e-9` 导致了较快的收敛，是相对合适的参数，后续实验中将使用这组参数。



### 第二组实验

本组实验关注于链长和冷却耗时两个参数。

**实验 #4 设置与结果**

模型 SA，T_max = 1, T_min = 1e-9, L = 200, max_stay_counter = 150

<img src="img_6.png" alt="img_6.png" style="zoom:40%;" />
<img src="img_7.png" alt="img_7.png" style="zoom:40%;" />


**实验 #5 设置与结果**

模型 SA，T_max = 1, T_min = 1e-9, L = 100, max_stay_counter = 150

<img src="img_10.png" alt="img_10.png" style="zoom:40%;" />
<img src="img_11.png" alt="img_11.png" style="zoom:40%;" />


**实验 #6 设置与结果**

模型 SA，T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 100

<img src="img_12.png" alt="img_12.png" style="zoom:40%;" />
<img src="img_13.png" alt="img_13.png" style="zoom:40%;" />


**实验 #7 设置与结果**

模型 SA，T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 200

<img src="img_14.png" alt="img_14.png" style="zoom:40%;" />
<img src="img_15.png" alt="img_15.png" style="zoom:40%;" />


由第二组实验，与实验 #3 相比，`L = 300, max_stay_counter = 150` 导致了较快的收敛，是相对合适的参数，后续实验将使用这组参数。



### 第三组实验

本组实验关注于 `SA` 与 `SABoltzmann` 两种模型的选择。

**实验 #8 设置与结果**

模型 SABoltzmann，T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 150

<img src="img_16.png" alt="img_16.png" style="zoom:40%;" />
<img src="img_17.png" alt="img_17.png" style="zoom:40%;" />

**实验 #9 设置与结果**

模型 SABoltzmann，T_max = 100, T_min = 1e-7, L = 300, max_stay_counter = 150

<img src="img_18.png" alt="img_18.png" style="zoom:40%;" />
<img src="img_19.png" alt="img_19.png" style="zoom:40%;" />


**实验 #10 设置与结果**

模型 SABoltzmann，T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 200

<img src="img_20.png" alt="img_20.png" style="zoom:40%;" />
<img src="img_21.png" alt="img_21.png" style="zoom:40%;" />


**实验 #11 设置与结果**

模型 SABoltzmann，T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 100

<img src="img_22.png" alt="img_22.png" style="zoom:40%;" />
<img src="img_23.png" alt="img_23.png" style="zoom:40%;" />

**实验 #12 设置与结果**

模型 SABoltzmann，T_max = 1, T_min = 1e-9, L = 400, max_stay_counter = 150

<img src="img_24.png" alt="img_24.png" style="zoom:40%;" />
<img src="img_25.png" alt="img_25.png" style="zoom:40%;" />

**实验 #13 设置与结果**

模型 SABoltzmann，T_max = 1, T_min = 1e-9, L = 200, max_stay_counter = 150

<img src="img_26.png" alt="img_26.png" style="zoom:40%;" />
<img src="img_27.png" alt="img_27.png" style="zoom:40%;" />



由第三组实验与之前实验对比可知，SABoltzmann 模型显著优于 SA 模型。本实验中，`T_max = 1, T_min = 1e-9, L = 300, max_stay_counter = 150` 这一组参数相比其他参数可快速收敛。