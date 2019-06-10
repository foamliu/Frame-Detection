# Frame-Detection

## 数据集
共计 9935 个方案，373471个照片。

## 依赖
- Python 3.6.7
- PyTorch 1.0.0

## 如何使用

### 数据准备
预处理:
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

可视化训练过程：
```bash
$ tensorboard --logdir=runs
```

## 效果演示
从验证集中随机抽取10张检视识别效果：
```bash
$ python demo.py
```

原图|模型识别|GT|
|---|---|---|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_true.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_img.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_out.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_true.jpg)|


