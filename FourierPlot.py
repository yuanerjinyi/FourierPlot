import math
import sys
import os
import json
import cv2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button ,RangeSlider,RadioButtons,TextBox
from scipy.spatial import Delaunay
from scipy.fftpack import fft
import heapq
import threading
import tkinter
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from functools import partial
import copy
from manim import *
from pathlib import Path

class FourierRender(Scene):

    def construct(self):
        self.init()
        path=self.dft_path()
        factor = 6 / np.max(path[:, 1])
        x_Offset = -(np.sum(path[:, 0] * factor) / len(path))
        y_Offset = -3
        fx = fft((path[:, 0]*factor+x_Offset) + (path[:, 1]*factor+y_Offset) * 1j)
        fr = np.arange(len(fx))
        N=len(fx)
        srot = np.flipud(np.argsort(np.abs(fx)))
        fx = fx[srot]
        fr = fr[srot]
        rf=np.array([np.dot(fx,[np.exp(1j*2*np.pi*n*k/N) for k in fr])/N for n in range(N)])
        print("path路径")
        print(path[:20,:])
        print("fft路径")
        print(rf[:20])
        print(fx.ndim, "\t", fx.shape)
        print(fx[0])
        #将数据归一化到0-6
        factor=6/np.max(path[:,1])
        x_Offset=-(np.sum(path[:,0]*factor)/len(path))
        y_Offset=-3
        c1=np.array([factor*x+x_Offset+(factor*y+y_Offset)*1j for x,y in path])
        print(c1.size)
        self.dotxyz=[0,0,0]
        #傅里叶参数，运用的数值积分的方式
        fre = np.linspace(-self.frequency_range,self.frequency_range, 2*self.frequency_range+1)
        kwr = np.array(
            [np.dot(c1,[np.exp(-2*np.pi*n*t*1j)/c1.size for t in np.linspace(0,1,c1.size+1)[:-1]]) for n in fre])
        #按照每个本轮的模长排序
        srot=np.flipud(np.argsort(np.abs(kwr)))
        kwrs=[]
        fres=[]
        for n in srot:
            kwrs.append(kwr[n])
            fres.append(fre[n])
        kwr=np.array(kwrs)
        fre=np.array(fres)

        def guij(dec):
            if self.model:
                lisc = np.dot(fx, [np.exp(1j * 2 * np.pi * dec * k / N) for k in fr]) / N
                return [lisc.real, lisc.imag, 0]
            else:
                t=dec/self.frame_length
                lisc=np.dot(kwr,[np.exp(2*np.pi*fre[n]*t*1j) for n in range(fre.size)])
                return [lisc.real,lisc.imag,0]
        def guij1(dec):
            lisc = np.dot(fx,[np.exp(1j*2*np.pi*dec*k/N) for k in fr])/N
            return [lisc.real, lisc.imag, 0]

        def lincirs(dec):
            if self.model:
                lisc = fx * np.array([np.exp(1j * 2 * np.pi * dec * k / N) for k in fr]) / N
                xyz = [0, 0, 0]
                xyz1 = [0, 0, 0]
                vg = VGroup()
                for n in range(len(fr)):
                    xyz[0] += lisc[n].real
                    xyz[1] += lisc[n].imag
                    if xyz[0] != xyz1[0] and xyz[1] != xyz1[1]:
                        if np.abs(fx[n]) / N > 0.01 and n > 0:
                            vg.add(Circle(np.abs(fx[n]) / N, stroke_width=2).move_to(xyz1))
                        vg.add(Line(xyz1, xyz, color=YELLOW, stroke_width=2))
                    xyz1[0] = xyz[0]
                    xyz1[1] = xyz[1]
                self.dotxyz = copy.deepcopy(xyz)
                return vg
            else:
                t=dec/self.frame_length
                lisc=kwr*np.array([np.exp(2*np.pi*fre[n]*t*1j) for n in range(fre.size)])
                xyz=[0,0,0]
                xyz1=[0,0,0]
                vg = VGroup()
                for n in range(fre.size):
                    xyz[0]+=lisc[n].real
                    xyz[1]+=lisc[n].imag
                    if xyz[0]!=xyz1[0] or xyz[1]!=xyz1[1]:
                        if np.abs(kwr[n])>0.1 and n>0:
                            vg.add(Circle(np.abs(kwr[n]),stroke_width=2).move_to(xyz1))
                        vg.add(Line(xyz1,xyz,color=YELLOW,stroke_width=2))
                    xyz1[0]=xyz[0]
                    xyz1[1]=xyz[1]
                self.dotxyz=copy.deepcopy(xyz)
                return vg

        def lincirs1(dec):
            lisc=fx*np.array([np.exp(1j*2*np.pi*dec*k/N) for k in fr])/N
            xyz=[0,0,0]
            xyz1=[0,0,0]
            vg = VGroup()
            for n in range(len(fr)):
                xyz[0]+=lisc[n].real
                xyz[1]+=lisc[n].imag
                if xyz[0]!=xyz1[0] and xyz[1]!=xyz1[1]:
                    if np.abs(fx[n])/N>0.1 and n>0:
                        vg.add(Circle(np.abs(fx[n])/N,stroke_width=2).move_to(xyz1))
                    vg.add(Line(xyz1,xyz,color=YELLOW,stroke_width=2))
                xyz1[0]=xyz[0]
                xyz1[1]=xyz[1]
            self.dotxyz=copy.deepcopy(xyz)
            return vg

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": False,"close_new_points":False},
            tips=False,
            x_axis_config={
                "stroke_color": BLACK,
                "font_size": 12,
                "tick_size": 0.05,
            },
            y_axis_config={
                "stroke_color": BLACK,
                "font_size": 12,
                "tick_size": 0.05,
            },
        )
        decimal = DecimalNumber(-1,color=BLACK).move_to([-6,0,0])
        dot=Dot(radius=0.01).move_to(guij(0))
        pdot1=TracedPath(dot.get_center)
        alpha = ValueTracker(0)

        decimal.add_updater(lambda d:d.set_value(d.get_value()+1))
        dot.add_updater(lambda d:d.move_to(self.dotxyz))

        Gcirlin=always_redraw(lambda:lincirs(decimal.get_value()))

        self.add(axes,decimal,Gcirlin,pdot1,dot)
        self.play(alpha.animate.set_value(1),rate_func=linear,run_time=self.get_time())


    def dft_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = "points.npy"
        file_path = os.path.join(current_dir, file_name)
        loaded_point = np.load(file_path)

        ncontour = loaded_point.copy()
        max_xy = 1 / np.max((np.max(ncontour[:, 0]), np.max(ncontour[:, 1])))
        ncontour = ncontour.astype(np.float64)
        ncontour *= max_xy
        return ncontour

    def init(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "config.json")
        with open(config_file, "r") as f:
            loaded_config = json.load(f)
        self.length=loaded_config['length']
        self.frame_rate=loaded_config['frame_rate']
        self.frequency_range=loaded_config['frequency_range']
        self.frame_length=loaded_config['frame_length']
        self.model=loaded_config['model']

    def get_time(self):
        if self.model:
            time=math.ceil(self.length/self.frame_rate)
            return time
        else:
            time=math.ceil(self.frame_length/self.frame_rate)
            return time

    def save_data(self):
        config = {
            "length": self.length,
            "frame_rate": self.frame_rate,
            "frequency_range": self.frequency_range,
            "frame_length": self.frame_length,
            "model": self.model
        }
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        print(f"配置已保存到 {config_file}")




class Parameters:

    def __init__(self,length):
        self.length = length
        self.quality = "l"
        self.frame_rate = 30
        self.frequency_range = 300
        self.frame_length = 600
        self.model = False

    def save_data(self):
        config = {
            "length": self.length,
            "frame_rate": self.frame_rate,
            "frequency_range": self.frequency_range,
            "frame_length": self.frame_length,
            "model": self.model
        }
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        print(f"配置已保存到 {config_file}")

    def video_parameters(self):

        return (self.quality,self.frame_rate)

    def show_ui(self):
        # 创建主图形和子图
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.3, bottom=0.4)  # 调整布局
        self.ax.set_axis_off()  # 关闭坐标轴显示

        # 初始化变量
        self.current_method = "method1"  # 当前选择的计算方式
        self.current_quality = "Medium"  # 当前选择的画质
        self.result = 0  # 计算结果

        # 创建单选按钮组：计算方式
        self.radio_method_ax = plt.axes([0.1, 0.5, 0.2, 0.2])  # 定义按钮位置
        self.radio_method = RadioButtons(self.radio_method_ax, ["FT", "DFT"])
        self.radio_method.on_clicked(self.switch_method)

        # 创建单选按钮组：画质
        self.radio_quality_ax = plt.axes([0.1, 0.2, 0.2, 0.25])  # 定义按钮位置
        self.radio_quality = RadioButtons(self.radio_quality_ax, ["Low", "Medium", "High"])
        self.radio_quality.on_clicked(self.switch_quality)
        #提示文字
        self.ax.text(-0.34, 0.66, 'Calculation method',ha="left", va="center", color="blue", fontsize=10)
        self.ax.text(-0.34, 0.15, 'Video quality',ha="left", va="center", color="blue", fontsize=10)
        # 创建文本输入框（方式 1 参数）
        self.textbox_framerate_ax = plt.axes([0.6, 0.4, 0.2, 0.05])  # 定义帧率输入框位置
        self.textbox_framerate = TextBox(self.textbox_framerate_ax, "Frame Rate:",initial="30")

        self.textbox_frequency_ax = plt.axes([0.6, 0.3, 0.2, 0.05])  # 定义频率输入框位置
        self.textbox_frequency = TextBox(self.textbox_frequency_ax, "Frequency:",initial="100")

        self.textbox_length_ax = plt.axes([0.6, 0.2, 0.2, 0.05])  # 定义帧长度输入框位置
        self.textbox_length = TextBox(self.textbox_length_ax, "Frame Length:",initial="600")

        # 创建计算按钮
        self.calc_button_ax = plt.axes([0.4, 0.05, 0.2, 0.1])  # 定义按钮位置
        self.calc_button = Button(self.calc_button_ax, "Calculate")
        self.calc_button.on_clicked(self.calculate)

        # 创建显示结果的文本
        self.result_text = self.ax.text(0.4, 0.8, "Total Duration: Not calculated",
                                        ha="center", va="center", fontsize=12,color="green")

        # 初始状态
        self.update_inputs_visibility()
        plt.show()

    def switch_method(self, label):
        """切换计算方式"""
        self.current_method = "method1" if label == "FT" else "method2"
        self.update_inputs_visibility()

    def switch_quality(self, label):
        """切换画质等级"""
        self.current_quality = label

    def update_inputs_visibility(self):
        """更新输入框的可见性"""
        if self.current_method == "method1":
            # 方式 1 显示所有参数
            self.textbox_framerate_ax.set_visible(True)
            self.textbox_frequency_ax.set_visible(True)
            self.textbox_length_ax.set_visible(True)
        else:
            # 方式 2 隐藏不必要的参数
            self.textbox_framerate_ax.set_visible(True)
            self.textbox_frequency_ax.set_visible(False)
            self.textbox_length_ax.set_visible(False)
        self.fig.canvas.draw_idle()

    def calculate(self, event):
        """执行计算逻辑"""
        try:
            # 根据画质等级设置权重
            quality_weights = {"Low":"l", "Medium": "m", "High": "h"}
            quality_factor = quality_weights[self.current_quality]

            if self.current_method == "method1":
                # 方式 1 的计算公式
                framerate = self.textbox_framerate.text
                frequency = self.textbox_frequency.text
                frame_length = self.textbox_length.text
                if framerate.isdigit() and frequency.isdigit() and frame_length.isdigit():
                    framerate = int(framerate)
                    frequency = int(frequency)
                    frame_length = int(frame_length)
                    self.model=False
                    self.quality=quality_factor
                    self.frame_rate=framerate
                    self.frequency_range=frequency
                    self.frame_length=frame_length
                    self.result = frame_length/framerate
            else:
                # 方式 2 的计算公式（示例公式）
                framerate = self.textbox_framerate.text
                if framerate.isdigit():
                    framerate = int(framerate)
                    self.model=True
                    self.frame_rate = framerate
                    self.quality = quality_factor
                    self.result = self.length/framerate
            # 更新结果文本
            self.result_text.set_text(f"Total Duration: {self.result:.2f}")
        except ValueError:
            # 输入无效时显示错误提示
            self.result_text.set_text("Invalid input. Please check!")
        self.fig.canvas.draw_idle()

class CalculatePath:

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.withdraw()
        self.sample_ratio=0.8
        self.smooth_window=11
        self.epsilon=4.0
        self.kernel_size=(3,3)
        self.lower_threshold=50
        self.upper_threshold=150

    def _init_image(self):
        def format_zh(string):
            for ch in string:
                if u'\u4e00' <= ch <= u'\u9fff':
                    return True
            return False

        self.file_name = askopenfilename(title="Select file", filetypes=[
            ("image files", "*.jpg *.gif *.png *.jpeg *.webp")])
        if self.file_name and not format_zh(self.file_name):
            img = cv2.imread(self.file_name)
            # 将图片转换为8位灰度图
            self.imgL = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("无法识别中文路径")
            sys.exit()

    def _find_contours(self):
        # 对灰度图做高斯滤波去噪
        dst = cv2.GaussianBlur(self.imgL,self.kernel_size, 0, 0)
        # 用Canny进行图片边缘检测
        r1 = cv2.Canny(dst,self.lower_threshold,self.upper_threshold, apertureSize=3)
        # 轮廓提取
        contours, hierarchy = cv2.findContours(r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ncontour = contours.pop(0)
        for contour in contours:
            ncontour = np.concatenate((ncontour, contour), axis=0)
        ncontour = np.reshape(ncontour, (len(ncontour), 2))
        ncontour[:, 1] = -ncontour[:, 1] + r1.shape[0]
        xmin=np.min(ncontour[:,0])
        ymin=np.min(ncontour[:,1])
        ncontour[:,0]-=xmin
        ncontour[:,1]-=ymin

        t, rst = cv2.threshold(r1, 10, 255, cv2.THRESH_BINARY)
        self.canny=rst
        self.raw_data=ncontour



    def _random_sampling(self):
        indices = np.random.choice(len(self.raw_data), int(
            len(self.raw_data) * self.sample_ratio), replace=False)
        self.samples = self.raw_data[indices].copy()

    def _smooth_path(self, path):
        if self.smooth_window == 1:
            return path
        print('smoothing path with savgol_filter')
        before_len = sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path)))
        from scipy.signal import savgol_filter
        path = np.array(path)
        path[:, 0] = savgol_filter(path[:, 0], self.smooth_window, 2)
        path[:, 1] = savgol_filter(path[:, 1], self.smooth_window, 2)
        after_len = sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path)))
        print(f'path length after smoothing: {before_len} -> {after_len}')
        return path


    def _rdp_opencv(self,path):
        path=np.reshape(path,(len(path),1,2))
        approx = cv2.approxPolyDP(path,self.epsilon, False)
        return np.reshape(approx,(len(approx),2))

    def _compute_path(self):
        def do_work(self):
            if len(self.samples) <= 1:
                self.path = np.array([])
                return
            print(f'searching mst for {len(self.samples)} points')
            g = self._mst()
            print('searching st, ed')
            st, ed = self._find_farthest_leaf_pair(g)
            print('rearanging children order')
            self._rearange_children_order(g, st, ed)
            path = self._generate_path(g, st, ed)
            # connect start and end if not too far apart
            max_dis = max([np.linalg.norm(path[i - 1] - path[i])
                           for i in range(1, len(path))])
            print(f'max dis on path: {max_dis}')
            if self._dis(st, ed) < 2 * max_dis:
                print('connecting end and start')
                path.append(self.samples[st])
            path = self._smooth_path(path)
            self.path=self._rdp_opencv(path)
        # Workaround stack size limit on windows.
        # https://stackoverflow.com/questions/2917210/python-what-is-the-hard-recursion-limit-for-linux-mac-and-windows/2918118#2918118
        threading.stack_size(100 * 1024 * 1024)
        max_rec_depth = len(self.samples) + 100
        sys.setrecursionlimit(max_rec_depth)
        thread = threading.Thread(target=partial(do_work, self))
        thread.start()
        thread.join()


    def _dis(self, i, j):
        return np.linalg.norm(self.samples[i] - self.samples[j])

    def _find_farthest_leaf_pair(self, g):
        def dfs(i, parent):
            """
            Return
                - farthest leaf id in thissubtree and distance to root i
                - farthest leave pair in this subtree and distance between them
            """
            farthest_leaf = i
            farthest_leaf_dis = 0
            farthest_leaf_pair = None
            farthest_leaf_pair_dis = -1
            leave_dis = []
            for j, _ in g[i]:
                if j == parent:
                    continue
                l, ld, pair, pair_dis = dfs(j, i)
                leave_dis.append((ld + 1, l))
                if ld + 1 > farthest_leaf_dis:
                    farthest_leaf_dis = ld + 1
                    farthest_leaf = l
                if farthest_leaf_pair_dis < pair_dis:
                    farthest_leaf_pair = pair
                    farthest_leaf_pair_dis = pair_dis
            if len(leave_dis) >= 2:
                (d1, l1), (d2, l2) = sorted(leave_dis)[-2:]
                if d1 + d2 > farthest_leaf_pair_dis:
                    farthest_leaf_pair_dis = d1 + d2
                    farthest_leaf_pair = l1, l2
            return farthest_leaf, farthest_leaf_dis, farthest_leaf_pair, farthest_leaf_pair_dis

        for i in range(len(g)):
            if len(g[i]):
                l, ld, pair, pair_dis = dfs(i, -1)
                if len(g[i]) == 1 and ld > pair_dis:
                    # root is a leave
                    return i, l
                return pair

    def _rearange_children_order(self, g, st, ed):
        # reagange children list order to make sure ed is the last node to visit
        # when starting from st
        vis = set()

        def dfs(i):
            vis.add(i)
            if i == ed:
                return True
            for j in range(len(g[i])):
                if g[i][j][0] not in vis:
                    if dfs(g[i][j][0]):
                        g[i][j], g[i][-1] = g[i][-1], g[i][j]
                        return True
            return False

        dfs(st)
        return st, ed

    def _generate_path(self, g, st, ed):
        res = []
        vis = set()

        def dfs(i):
            vis.add(i)
            res.append(self.samples[i])
            if i == ed:
                return True
            leaf = True
            for j, _ in g[i]:
                if j not in vis:
                    leaf = False
                    if dfs(j):
                        return True
            if not leaf:
                # don't visit leaf twice
                res.append(self.samples[i])
            return False

        dfs(st)
        return res

    def _mst(self):
        print('running Delaunay triangulation')
        n = len(self.samples)
        tri = Delaunay(self.samples)
        g = [[] for i in range(n)]

        edges = {}
        nodes = set()
        for simplex in tri.simplices:
            nodes |= set(simplex)
            for k in range(3):
                i, j = simplex[k - 1], simplex[k]
                edge = min(i, j), max(i, j)
                if edge not in edges:
                    edges[edge] = self._dis(i, j)
        pq = [(d, i, j) for ((i, j), d) in edges.items()]
        heapq.heapify(pq)
        p = list(range(n))

        def union(i, j):
            p[find(i)] = find(j)

        def find(i):
            if p[i] == i:
                return i
            p[i] = find(p[i])
            return p[i]

        print('running kruskal')
        # nodes may not contain all points as some points close to each other are treated as single points
        cc = len(nodes)
        while cc > 1:
            d, i, j = heapq.heappop(pq)
            if find(i) != find(j):
                union(i, j)
                g[i].append((j, d))
                g[j].append((i, d))
                cc -= 1
        return g

    def _base_name(self,fname, with_extension=False):
        fname = os.path.basename(fname)
        if not with_extension:
            fname = os.path.splitext(fname)[0]
        return fname

    def _updata_first(self):
        self._find_contours()
        self._random_sampling()
        self._compute_path()

        self.image_ax.imshow(self.imgL, cmap='gray', vmin=0, vmax=255)
        self.image_ax.set_title("Grayscale")
        self.image_ax.set_axis_off()

        self.canny_ax.imshow(self.canny, cmap='gray', vmin=0, vmax=255)
        self.canny_ax.set_title("Canny")
        self.canny_ax.set_axis_off()

        self.sample_ax.plot(self.samples[:,0],self.samples[:,1], marker='o',linestyle='none',color='b', markersize=1)
        self.sample_ax.set_title(f"{len(self.samples)} Canny")
        self.sample_ax.set_axis_off()
        self.sample_ax.set_aspect('equal', adjustable='datalim')

        self.rdp_ax.plot(self.path[:,0], self.path[:,1], marker='o',linestyle='-', color='b', markersize=1)
        self.rdp_ax.set_title(f"{len(self.path)} Rdp")
        self.rdp_ax.set_axis_off()
        self.rdp_ax.set_aspect('equal', adjustable='datalim')

    def _updata_file(self,value):
        self._init_image()
        self._find_contours()
        self._random_sampling()
        self._compute_path()

        self.image_ax.clear()
        self.image_ax.imshow(self.imgL, cmap='gray', vmin=0, vmax=255)
        self.image_ax.set_title("Grayscale")
        self.image_ax.set_axis_off()

        self.canny_ax.clear()
        self.canny_ax.imshow(self.canny, cmap='gray', vmin=0, vmax=255)
        self.canny_ax.set_title("Canny")
        self.canny_ax.set_axis_off()

        self.sample_ax.clear()
        self.sample_ax.plot(self.samples[:, 0], self.samples[:, 1], marker='o', linestyle='none', color='b',markersize=1)
        self.sample_ax.set_title(f"{len(self.samples)} Canny")
        self.sample_ax.set_axis_off()
        self.sample_ax.set_aspect('equal', adjustable='datalim')

        self.rdp_ax.clear()
        self.rdp_ax.plot(self.path[:, 0], self.path[:, 1], marker='o', linestyle='-', color='b', markersize=1)
        self.rdp_ax.set_title(f"{len(self.path)} Rdp")
        self.rdp_ax.set_axis_off()
        self.rdp_ax.set_aspect('equal', adjustable='datalim')
        plt.draw()

    def _updata_kernel(self,value):
        self.kernel_size=(value,value)
        self._find_contours()
        self._random_sampling()
        self._compute_path()

        self.canny_ax.clear()
        self.canny_ax.imshow(self.canny, cmap='gray', vmin=0, vmax=255)
        self.canny_ax.set_title("Canny")
        self.canny_ax.set_axis_off()

        self.sample_ax.clear()
        self.sample_ax.plot(self.samples[:, 0], self.samples[:, 1], marker='o', linestyle='none', color='b',
                            markersize=1)
        self.sample_ax.set_title(f"{len(self.samples)} Canny")
        self.sample_ax.set_axis_off()
        self.sample_ax.set_aspect('equal', adjustable='datalim')

        self.rdp_ax.clear()
        self.rdp_ax.plot(self.path[:, 0], self.path[:, 1], marker='o', linestyle='-', color='b', markersize=1)
        self.rdp_ax.set_title(f"{len(self.path)} Rdp")
        self.rdp_ax.set_axis_off()
        self.rdp_ax.set_aspect('equal', adjustable='datalim')
        plt.draw()

    def _updata_threshold(self,value):
        self.lower_threshold=value[0]
        self.upper_threshold=value[1]

        self._find_contours()
        self._random_sampling()
        self._compute_path()

        self.canny_ax.clear()
        self.canny_ax.imshow(self.canny, cmap='gray', vmin=0, vmax=255)
        self.canny_ax.set_title("Canny")
        self.canny_ax.set_axis_off()

        self.sample_ax.clear()
        self.sample_ax.plot(self.samples[:, 0], self.samples[:, 1], marker='o', linestyle='none', color='b',
                            markersize=1)
        self.sample_ax.set_title(f"{len(self.samples)} Canny")
        self.sample_ax.set_axis_off()
        self.sample_ax.set_aspect('equal', adjustable='datalim')

        self.rdp_ax.clear()
        self.rdp_ax.plot(self.path[:, 0], self.path[:, 1], marker='o', linestyle='-', color='b', markersize=1)
        self.rdp_ax.set_title(f"{len(self.path)} Rdp")
        self.rdp_ax.set_axis_off()
        self.rdp_ax.set_aspect('equal', adjustable='datalim')
        plt.draw()

    def _updata_sampling(self,value):
        self.sample_ratio=value

        self._random_sampling()
        self._compute_path()

        self.sample_ax.clear()
        self.sample_ax.plot(self.samples[:, 0], self.samples[:, 1], marker='o', linestyle='none', color='b',
                            markersize=1)
        self.sample_ax.set_title(f"{len(self.samples)} Canny")
        self.sample_ax.set_axis_off()
        self.sample_ax.set_aspect('equal', adjustable='datalim')

        self.rdp_ax.clear()
        self.rdp_ax.plot(self.path[:, 0], self.path[:, 1], marker='o', linestyle='-', color='b', markersize=1)
        self.rdp_ax.set_title(f"{len(self.path)} Rdp")
        self.rdp_ax.set_axis_off()
        self.rdp_ax.set_aspect('equal', adjustable='datalim')
        plt.draw()

    def _updata_rdp(self,value):
        self.epsilon=value

        self._compute_path()

        self.rdp_ax.clear()
        self.rdp_ax.plot(self.path[:, 0], self.path[:, 1], marker='o', linestyle='-', color='b', markersize=1)
        self.rdp_ax.set_title(f"{len(self.path)} Rdp")
        self.rdp_ax.set_axis_off()
        self.rdp_ax.set_aspect('equal', adjustable='datalim')
        plt.draw()

    def animation(self):
        x = self.path[:, 0]
        y = self.path[:, 1]
        x_min, x_max = 0, np.max(x) + 1  # 横坐标范围，留一点边距
        y_min, y_max = 0, np.max(y) + 1  # 纵坐标范围，留一点边距
        # 创建图形

        fig, ax = plt.subplots()
        line, = ax.plot([], [], marker='o', linestyle='-', color='b', markersize=1)
        # 设置固定坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_axis_off()
        #fig.tight_layout()
        #ax.set_aspect('equal', adjustable='datalim')

        def init():
            line.set_data([], [])
            return line,
        # 动画更新函数
        def update(frame):
            line.set_data(x[:frame + 1], y[:frame + 1])
            return line,
        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(self.path), init_func=init, blit=True, interval=10)
        # 显示动画
        plt.show()

    def get_length(self):

        return len(self.path)

    def get_path(self):
        ncontour=self.path.copy()
        max_xy=1/np.max((np.max(ncontour[:,0]),np.max(ncontour[:,1])))
        ncontour=ncontour.astype(np.float64)
        ncontour*=max_xy
        return ncontour

    def save(self):
        file_name_first_10 = os.path.splitext(os.path.basename(self.file_name))[0][:10]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        #file_name=file_name_first_10+".txt"
        file_name = "points.npy"
        file_path = os.path.join(current_dir, file_name)
        #np.savetxt(file_path, self.path, header="x y", fmt="%d", delimiter="\t")
        np.save(file_path, self.path)

    def render(self):
        self._init_image()

        fig = plt.figure(figsize=(8, 8))  # 创建 Figure 对象
        ((self.image_ax, self.canny_ax), (self.sample_ax,self.rdp_ax))=fig.subplots(2, 2)

        # 调整子图布局，给按钮和滑动条腾出空间
        fig.subplots_adjust(bottom=0.2)  # 留出底部空间
        fig.canvas.manager.set_window_title("Convert graphic outline to path")

        # 使用 Matplotlib 的 add_axes 添加按钮
        button_ax = fig.add_axes([0.4, 0.94, 0.2, 0.05])  # 在 Figure 底部添加按钮轴
        mpl_button = Button(button_ax, "Select File", color="white", hovercolor="lightgreen")  # 创建 Matplotlib 按钮
        # mpl_button.label.set_rotation(90)
        mpl_button.on_clicked(self._updata_file)  # 绑定回调函数

        # 添加四个滑动条

        nuclear_ax = fig.add_axes([0.1, 0.02 * 4, 0.7, 0.01])
        slider_nuclear = Slider(nuclear_ax, "Kernel size",1,45, valinit=self.kernel_size[0],valstep=2)
        slider_nuclear.on_changed(self._updata_kernel)  # 绑定回调函数

        threshold_ax = fig.add_axes([0.1, 0.02*3, 0.7, 0.01])  # 范围滑动条放置位置
        slider_threshold = RangeSlider(threshold_ax, label="Canny threshold", valmin=0, valmax=255, valinit=(self.lower_threshold, self.upper_threshold),valstep=1)
        slider_threshold.on_changed(self._updata_threshold)  # 绑定回调函数


        sampling_ax = fig.add_axes([0.1, 0.02*2, 0.7, 0.01])
        slider_sampling = Slider(sampling_ax, "Sampling accuracy",0.0,1.0, valinit=self.sample_ratio,valstep=0.01)
        slider_sampling.on_changed(self._updata_sampling)  # 绑定回调函数

        rdp_ax = fig.add_axes([0.1, 0.02, 0.7, 0.01])
        slider_rdp = Slider(rdp_ax, "Rdp epsilon",0,30.0, valinit=self.epsilon,valstep=0.1)
        slider_rdp.on_changed(self._updata_rdp)  # 绑定回调函数

        self._updata_first()
        # 显示图形
        plt.show()



if __name__ == "__main__":
    param_computer = CalculatePath()
    param_computer.render()
    param_computer.animation()
    param_computer.save()

    parameter=Parameters(param_computer.get_length())
    parameter.show_ui()
    parameter.save_data()
    quality,rate=parameter.video_parameters()

    filename=Path(__file__).resolve().name
    os.system(f"manim -pq{quality} --fps {rate} {filename} FourierRender")