# 在manim中用傅里叶绘图
1 项目的主要功能是将图像轮廓扫描成顺序点集，再将这些顺序的点集路径计算成傅里叶参数，最后用一系列不同频率的信号(本轮)合成原始图像轮廓。
其中用来来将图像轮廓顺序化的程序是借鉴了本项目https://github.com/biran0079/circle
源程序是用matplotlib检测图像轮廓，在本程序中是用opencv检测图像轮廓的，并用opencv中的逼近多边形对路径进行优化。也对源程序的ui进行了改进。得到图像轮廓的顺序点集后在manim进行绘制，运用了数值积分方法对离散点进行傅里叶计算，计算了-300~300频率范围的初始参数。最后用这些计算的初始参数在一秒内合成原始的数据点。
## 效果
![图示](https://github.com/user-attachments/assets/32211890-8d6b-440b-8e4f-b7734e7188bf)
        
## 界面
     
![0bc85c7e-fb3f-4798-9138-5ebd2181dbf7](https://github.com/user-attachments/assets/0ca26ba4-5ace-4a0c-80c6-d27b478f0040)


## 动画

![db77b87a-831c-431d-8986-dbf608ccc46e](https://github.com/user-attachments/assets/bd6360eb-d03e-4fd9-a288-3ce31527f965)

## manim渲染


![dffc8a0a-e5cb-49b6-bb61-9e67ab8c6600](https://github.com/user-attachments/assets/347505c9-416a-4093-8632-25abc0f1cf58)
