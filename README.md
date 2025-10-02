<div align="center">

<img src="https://free.picui.cn/free/2025/08/11/6899bb5b91222.png" alt="logo" width="200" height="200">

# AntiCAP

<strong>Version:3.2.6</strong>

<strong>识别率随缘，全靠内置模型和算法</strong>

| 类型         | 状态 | 描述                                         |
|------------|-|--------------------------------------------|
| `OCR识别`    |✅| 返回图片字符串                                    |
| `数学计算`     |✅| 返回计算结果                                     |
| `缺口滑块`     |✅| 返回坐标                                       |
| `阴影滑块`     |✅| 返回坐标                                       |
| `图标点选`     |✅| 侦测图标位置 或 按序返回坐标                            |
| `文字点选`     |✅| 侦测文字位置 或 按序返回坐标                            |
| `相似对比`     |✅| 图片中文字的相似度对比                                |
| `双图旋转验证码`  |✅| 返回角度                                       |
| `单图旋转验证码`  |✅| 返回角度                                       |
| `WebApi服务` | ✅ | https://github.com/81NewArk/AntiCAP-WebApi |


</div>

---
<div align="center">

## 📑 目录

</div>

- [🧑‍⚖️ 免责声明](#免责声明)
- [📄 AntiCAP 文档](#anticap-文档)
  - [🌍 环境说明](#环境说明)
  - [📁 安装](#安装)
- [🤖 调用说明](#调用说明)
  - [1. 通用OCR识别](#1-通用ocr识别)
  - [2. 算术验证码识别](#2-算术验证码识别)
  - [3. 图标侦测](#3-图标侦测)
  - [4. 文字侦测](#4-文字侦测)
  - [5. 图标点选类](#5-图标点选类)
  - [6. 文字点选类](#6-文字点选类)
  - [7. 缺口滑块类](#7-缺口滑块类)
  - [8. 阴影滑块类](#8-阴影滑块类)
  - [9. 相似度对比](#9-相似度对比)
  - [10. 双图旋转类验证码](#10-双图旋转类验证码)
  - [11. 单图旋转类验证码](#11-单图旋转类验证码)
- [🧬 自定义模型](#自定义模型)
- [🐧 QQ交流群](#qq交流群)
- [💪🏼 模型训练](#模型训练)
- [🚬 请作者抽一包香香软软的利群](#请作者抽一包香香软软的利群)
- [😚 致谢名单](#致谢名单)
- [📚 参考文献](#参考文献)




---



<h2 id="免责声明" align="center">🧑‍⚖️ 免责声明</h2>




本项目基于 MIT 开源协议 发布，欢迎自由使用、修改和分发，但必须遵守中华人民共和国法律法规。

使用本项目即表示您已阅读并同意以下条款：

1. **合法使用：** 不得将本项目用于任何违法、违规或侵犯他人权益的行为，包括但不限于网络攻击、诈骗、绕过身份验证、未经授权的数据抓取等。

2. **风险自负：** 任何因使用本项目而产生的法律责任、技术风险或经济损失，由使用者自行承担，项目作者不承担任何形式的责任。

3. **禁止滥用：** 不得将本项目用于违法牟利、黑产活动或其他不当商业用途。

### 使用视为同意上述条款,即"谁使用，谁负责"。如不同意，请立即停止使用并删除本项目。




<br>



<h1 id="anticap-文档" align="center">📄 AntiCAP 文档</h1>



## 🌍环境说明

```
python >=3.8  64bit
```

## 📁 安装


###  Pypi下载
```
pip install AntiCAP -i https://pypi.tuna.tsinghua.edu.cn/simple
```


<h2 id="调用说明" align="center">🤖 调用说明</h2>



###  1. 通用OCR识别
#### 参考例图 (数字、大小写字母、汉字)
<img src="https://free.picui.cn/free/2025/07/30/68896c92e18f0.jpg">


```python
# example.py

import base64
import AntiCAP


with open("captcha.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)
result = Atc.OCR(img_base64=img_base64) #传入图片Base64编码字符串

print(result) # 返回字符串 jepy
```

---



###  2. 算术验证码识别
#### 参考例图 (加减乘除) 
<img src="https://free.picui.cn/free/2025/07/30/6889718adee8f.jpg">


```python
# example.py

import base64
import AntiCAP


with open("captcha.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)
result = Atc.Math(img_base64=img_base64) #传入图片Base64编码字符串

print(result) #返回计算结果 8

```


---



###  3. 图标侦测
#### 参考例图
<img src="https://free.picui.cn/free/2025/07/30/688972d69d3c1.jpg" width="200" height="200">


```python
# example.py

import base64
import AntiCAP


with open("captcha.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)
result = Atc.Detection_Icon(img_base64=img_base64) #传入图片Base64编码字符串

print(result)

# [{'class': 'icon', 'box': [9.12, 105.4, 111.73, 223.02]}...]
# box分别为 [x1, y1, x2, y2] 左上角和右下角坐标

```


---



###  4. 文字侦测
#### 参考例图
<img src="https://free.picui.cn/free/2025/07/30/688974085b38e.jpg" width="200" height="200">


```python
# example.py

import base64
import AntiCAP


with open("captcha.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)
result = Atc.Detection_Text(img_base64=img_base64) #传入图片Base64编码字符串

print(result)
# [{'class': 'Text', 'box': [145.71, 19.21, 223.99, 95.7]}...]
# box分别为 [x1, y1, x2, y2] 左上角和右下角坐标
```


---




###  5. 图标点选类
#### 提示图
<img src="https://free.picui.cn/free/2025/07/30/688975c92514c.jpg" width="200" height="50">

#### 目标图片
<img src="https://free.picui.cn/free/2025/07/30/688972d69d3c1.jpg" width="200" height="200">


```python
# example.py

import base64
import AntiCAP

with open("order_image.jpg", "rb") as f:
    order_img_base64 = base64.b64encode(f.read()).decode('utf-8')

# 读取目标图（所有图标）并转为 base64
with open("target_image.jpg", "rb") as f:
    target_img_base64 = base64.b64encode(f.read()).decode('utf-8')

Atc = AntiCAP.Handler(show_banner=True)
result = Atc.ClickIcon_Order(
    order_img_base64=order_img_base64,
    target_img_base64=target_img_base64
)

print(result)
```



---



###  6. 文字点选类
#### 提示图
<img src="https://free.picui.cn/free/2025/07/30/6889773219292.jpg" width="200" height="50">

#### 目标图片
<img src="https://free.picui.cn/free/2025/07/30/688974085b38e.jpg" width="200" height="200">


```python
# example.py

import base64
import AntiCAP

with open("order_image.jpg", "rb") as f:
    order_img_base64 = base64.b64encode(f.read()).decode('utf-8')

# 读取目标图（所有图标）并转为 base64
with open("target_image.jpg", "rb") as f:
    target_img_base64 = base64.b64encode(f.read()).decode('utf-8')

Atc = AntiCAP.Handler(show_banner=True)
result = Atc.ClickText_Order(
    order_img_base64=order_img_base64,
    target_img_base64=target_img_base64
)

print(result)
```



---



###  7. 缺口滑块类
#### 缺口图
<img src="https://free.picui.cn/free/2025/07/30/68897881c804c.png" width="50" height="120">

#### 背景图
<img src="https://free.picui.cn/free/2025/07/30/688978834962d.jpg" width="400" height="200">

```python
# example.py

import base64
import AntiCAP

# 读取滑块图片（小块）
with open("slider.png", "rb") as f:
    target_base64 = base64.b64encode(f.read()).decode('utf-8')

# 读取背景图片（带缺口的大图）
with open("background.jpg", "rb") as f:
    background_base64 = base64.b64encode(f.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)

result = Atc.Slider_Match(target_base64=target_base64,
                          background_base64=background_base64
)

print(result)
```



---




###  8. 阴影滑块类
#### 目标图片
<img src="https://free.picui.cn/free/2025/07/30/68897997591b9.jpg" width="400" height="200">

#### 背景图片
<img src="https://free.picui.cn/free/2025/07/30/68897997a65d7.jpg" width="400" height="200">

```python
# example.py

import base64
import AntiCAP

# 读取滑块图片（小块）
with open("target.jpg", "rb") as f:
    target_base64 = base64.b64encode(f.read()).decode('utf-8')

# 读取背景图片（带缺口的大图）
with open("background.jpg", "rb") as f:
    background_base64 = base64.b64encode(f.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)

result = Atc.Slider_Match(target_base64=target_base64,
                          background_base64=background_base64
)

print(result)
```



---




###  9. 相似度对比
#### 图片1
<img src="https://free.picui.cn/free/2025/07/30/68897a1a09ecc.jpg" width="100" height="100">

#### 图片2
<img src="https://free.picui.cn/free/2025/07/30/68897a1a0b5b7.jpg" width="100" height="100">

```python
# example.py
import base64
import AntiCAP

with open("image1.jpg", "rb") as f:
    image1_base64 = base64.b64encode(f.read()).decode('utf-8')

with open("image2.jpg", "rb") as f:
    image2_base64 = base64.b64encode(f.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)

result = Atc.Compare_Image_Similarity(image1_base64=image1_base64, image2_base64=image2_base64)

print("相似度结果:", result)

```


###  10. 双图旋转类验证码
#### 内圈
<img src="https://free.picui.cn/free/2025/09/02/68b6642405a61.png" width="100" height="100">

#### 外圈
<img src="https://free.picui.cn/free/2025/09/02/68b6642400acd.png" width="300" height="150">

```python
# example.py

import base64
import AntiCAP

with open("inside.jpg", "rb") as f:
    inside_base64 = base64.b64encode(f.read()).decode('utf-8')

with open("outside.jpg", "rb") as f:
    outside_base64 = base64.b64encode(f.read()).decode('utf-8')


Atc = AntiCAP.Handler(show_banner=True)

result= Atc.Double_Rotate(inside_base64=inside_base64, outside_base64=outside_base64)

print(result)

# {'similarity': 0.6651270985603333, 'inner_angle': 75.5, 'raw_angle': 151}

```


###  11. 单图旋转类验证码

#### 图片
<img src="https://free.picui.cn/free/2025/09/10/68c1141e02a9f.jpg" width="200" height="200">

```python
# example.py

import base64
import AntiCAP

with open("rotate_image.jpg", "rb") as f:
    rotate_base64 = base64.b64encode(f.read()).decode('utf-8')




Atc = AntiCAP.Handler(show_banner=True)

result= Atc.Single_Rotate(image_base64=rotate_base64)

print(result)

# 229  返回旋转角度

```





<h2 id="自定义模型" align="center">🧬 自定义模型</h2>



## 使用模型的函数均支持自定义模型



```python

import AntiCAP

Atc = AntiCAP.Handler(show_banner=True)


result = Atc.Math(img_base64="",
                  # math_model_path 参数为 .pt格式的yolo侦测模型
                  math_model_path=r"C:\AntiCAP\model\yuor_math_model.pt")


result = Atc.Detection_Icon(img_base64="",
                            # text_model_path 参数为 .pt格式的yolo侦测模型
                            detectionIcon_model_path=r"C:\AntiCAP\model\yuor_detectionIcon_model.pt")

result = Atc.Detection_Text(img_base64="",
                            # detectionText_model_path 参数为 .pt格式的yolo侦测模型
                            detectionText_model_path=r"C:\AntiCAP\model\yuor_detectionText_model.pt")


result = Atc.ClickText_Order(order_img_base64="",
                             target_img_base64="",
                             # text_model_path 参数为 .pt格式的yolo侦测模型
                             detectionText_model_path=r"C:\AntiCAP\model\yuor_detectionTcon_model.pt",
                             # sim_onnx_model_path为 .onnx格式的孪生神经网络模型
                             sim_onnx_model_path=r"C:\AntiCAP\model\your_sim_onnx_model.onnx") 


result = Atc.ClickIcon_Order(order_img_base64="",
                             target_img_base64="",
                             # text_model_path 参数为 .pt格式的yolo侦测模型
                             detectionIcon_model_path=r"C:\AntiCAP\model\yuor_detectionIcon_model.pt",
                             # sim_onnx_model_path为 .onnx格式的孪生神经网络模型
                             sim_onnx_model_path=r"C:\AntiCAP\model\your_sim_onnx_model.onnx") 


result = Atc.Compare_Image_Similarity(image1_base64="",
                                      image2_base64="",
                                      # sim_onnx_model_path为 .onnx格式的孪生神经网络模型
                                      sim_onnx_model_path=r"C:\AntiCAP\model\your_sim_onnx_model.onnx")


result = Atc.Single_Rotate(image_base64="",
                            # rotate_onnx_modex_path为 .onnx格式的旋转模型
                            rotate_onnx_modex_path=r"C:\AntiCAP\model\your_rotate_onnx_model.onnx")

```


<h1 id="qq交流群" >🐧 QQ交流群</h1>

<br>

<div align="center">



<img src="https://free.picui.cn/free/2025/07/04/6867f1907d1a0.png" alt="QQGroup" width="200" height="200">

</div>






<h1 id="模型训练" >💪🏼 模型训练</h1>

<br>

<div align="center">

### 知识星球逐步更新
### https://t.zsxq.com/Mwjpr

### 星球收入用于项目维护及更新。 
### 为避免星球收20%手续费 可以私我VX拉进去。


</div>


<h1 id="请作者抽一包香香软软的利群" >🚬 请作者抽一包香香软软的利群</h1>

<br>

<div align="center">

<img src="https://free.picui.cn/free/2025/07/04/6867efd0bd67e.png" alt="Ali" width="200" height="200">
<img src="https://free.picui.cn/free/2025/07/04/6867efd0d7cbb.png" alt="Wx" width="200" height="200">

</div>

<h1 id="致谢名单">😚 致谢名单</h1>


<strong>这份荣光我不会独享</strong>


[1] Ddddocr作者 网名:sml2h3


[2] 微信公众号 OneByOne 网名:十一姐


[3] 苏州大学,苏州大学文正学院 计算机科学与技术学院 张文哲教授


[4] 苏州大学,苏州大学文正学院 计算机科学与技术学院 王辉教授


[5] 苏州市职业大学,苏州大学文正学院 计算机科学与技术学院 陆公正副教授


[6] 武汉科锐软件安全教育机构 钱林松讲师 网名:Backer



<h1 id="参考文献">📚 参考文献</h1>

[1] Github. 2025.03.28 https://github.com/sml2h3


[2] Github. 2025.03.28 https://github.com/2833844911/


[3] Bilibili. 2025.03.28 https://space.bilibili.com/308704191


[4] Bilibili. 2025.03.28 https://space.bilibili.com/472467171


[5] Ultralytics. 2025.03.28 https://docs.ultralytics.com/modes/train/


[6] YRL's Blog. 2025.03.28 https://blog.2zxz.com/archives/icondetection





