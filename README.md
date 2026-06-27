三摆仿真与可视化
中文说明
项目简介
这是一个基于数值积分的三摆系统仿真与动画生成脚本。程序会先计算整个时间区间内的运动轨迹，再按设定的播放速度和帧率生成动画，支持导出 GIF 和 MP4，并提供多种颜色方案、轨迹尾迹和参数面板。

特性
三摆动力学数值仿真，采用拉格朗日形式建模
支持 RK4 和半隐式欧拉两种积分方式
先计算、后渲染，便于导出和复现
支持 GIF 和 MP4 输出
支持多种颜色映射方案
支持轨迹尾迹显示、平滑和长度控制
支持进度条，计算和渲染过程都有反馈
支持右侧参数面板，主图更简洁
输出文件默认保存在脚本所在目录
默认不自动弹出窗口，适合批量运行和服务器环境
环境要求
Python 3.10 或更高版本
依赖库：
numpy
matplotlib
pillow
tqdm
如果需要导出 MP4，还需要系统可用的 ffmpeg
安装依赖
安装基础依赖：

pip install numpy matplotlib pillow tqdm

如果需要导出 MP4，请确保系统已安装 ffmpeg，并且命令行可以直接调用。

运行方式
直接运行主脚本即可。

python 主脚本.py

程序会先完成仿真计算，再生成动画文件。

可配置项
脚本通常会把这些参数集中放在文件顶部，便于修改：

初始角度：可手动输入，也可使用默认值
总时长：控制模拟覆盖的时间范围
时间步长：影响数值精度和计算速度
帧率：影响最终动画播放节奏
播放速度：用于快放或慢放
积分器：RK4 或半隐式欧拉
颜色模式：不同方式把角度、相位或能量映射到颜色
颜色动态开关：是否让颜色随时间变化
颜色平滑：减少颜色跳变
尾迹开关：是否显示摆球轨迹
尾迹长度：保留多少历史轨迹
尾迹平滑：让轨迹更顺
信息面板开关：是否显示参数说明
网格开关：是否显示辅助网格
输出格式：GIF、MP4，或两者都输出
输出分辨率：通过画布尺寸和 dpi 控制
窗口显示：是否在结束后弹出预览窗口
颜色模式说明
sin_smooth：对角度做平滑正弦映射，颜色变化连续
cos_phase：以相位感更强的方式显示变化
atan_soft：压缩极端值，过渡更柔和
linear_wrap：线性环绕映射，直观但可能有跳变
hsv_cycle：循环色相方案，适合强调周期性
triad_soft：三通道柔和混色，视觉更均衡
cmap_theta1：按第一个角度选取色图
energy_heat：按能量变化着色，适合观察系统状态
输出说明
输出文件会保存在脚本所在目录，而不是当前工作目录。

常见输出包括：

GIF 动画
MP4 动画
可能的中间缓存文件
物理模型说明
这个程序模拟的是经典三摆系统。它不是逐帧手工绘制，而是基于动力学方程进行数值求解。

核心流程如下：

用三个广义坐标描述三个摆角
建立系统的质量矩阵、重力项和速度相关项
通过数值积分推进状态
再把状态序列转换成动画帧
因此，动画效果来自真实的动力学计算，而不是预设路径。

需要注意的是，结果仍然受数值积分步长、积分器和浮点误差影响。

推荐使用方式
如果你想要稳定、清晰、适合演示的结果，建议：

使用 RK4
时间步长不要设得太大
颜色模式优先选平滑型
开启尾迹，但不要把尾迹长度设得过长
输出分辨率可以适当提高
常见问题
如果动画太慢：

降低总时长
增大时间步长
降低输出帧率
关闭不必要的尾迹和信息面板
如果颜色变化太突兀：

换成更平滑的颜色模式
打开颜色平滑
避免过于线性的环绕映射
如果 MP4 导出失败：

检查 ffmpeg 是否已安装
确认系统命令行能直接调用 ffmpeg
如果窗口不弹出：

这是默认行为
这样更适合批量运行和无图形环境
适用场景
教学演示
混沌系统可视化
数值积分实验
动画生成
物理建模展示
进一步改造成交互式程序的基础
English Documentation
Project Overview
This is a numerical simulation and animation generator for a triple pendulum system. The program first computes the full trajectory over time, then renders the animation according to the configured playback speed and frame rate. It supports GIF and MP4 export, multiple color schemes, trajectory trails, and an information panel.

Features
Numerical triple pendulum simulation based on Lagrangian mechanics
Supports both RK4 and semi-implicit Euler integration
Precompute first, then render for better export and reproducibility
Supports GIF and MP4 output
Supports multiple color mapping modes
Supports trail display, smoothing, and length control
Includes progress bars for both simulation and rendering
Includes a side panel for parameters, keeping the main plot clean
Output files are saved in the script directory by default
Does not auto-open a window by default, which is suitable for batch jobs and server usage
Requirements
Python 3.10 or newer
Dependencies:
numpy
matplotlib
pillow
tqdm
If you want MP4 export, ffmpeg must also be available on your system
Installation
Install the basic dependencies:

pip install numpy matplotlib pillow tqdm

For MP4 export, make sure ffmpeg is installed and accessible from the command line.

How to Run
Run the main script directly.

python 主脚本.py

The program will first simulate the motion and then generate the animation files.

Configurable Options
These parameters are usually placed near the top of the script for easy editing:

Initial angles: can be entered manually or use defaults
Total duration: controls the simulated time span
Time step: affects numerical accuracy and performance
Frame rate: affects the final animation playback
Playback speed: used for fast or slow motion
Integrator: RK4 or semi-implicit Euler
Color mode: different ways to map angles, phase, or energy to colors
Dynamic color toggle: whether colors change over time
Color smoothing: reduces abrupt color jumps
Trail toggle: whether to show the pendulum trail
Trail length: how much history is kept
Trail smoothing: makes the trail smoother
Info panel toggle: whether to display parameter information
Grid toggle: whether to show a guide grid
Output format: GIF, MP4, or both
Output resolution: controlled by figure size and dpi
Window display: whether to open a preview window at the end
Color Mode Guide
sin_smooth: smooth sine mapping of angles with continuous transitions
cos_phase: phase-like visualization with smoother cyclical feel
atan_soft: compresses extreme values for softer transitions
linear_wrap: linear wrapped mapping, intuitive but may jump at boundaries
hsv_cycle: cyclic HSV color scheme, good for periodic behavior
triad_soft: soft tri-channel mixing with balanced appearance
cmap_theta1: uses the first angle to index the colormap
energy_heat: colors by energy, useful for observing system state
Output
Output files are saved in the script directory instead of the current working directory.

Common outputs include:

GIF animation
MP4 animation
Possible intermediate cache files
Physics Model
This program simulates a classical triple pendulum. It is not manually animated frame by frame; instead, it numerically solves the governing equations of motion.

The core workflow is:

Use three generalized coordinates to represent the three angles
Build the mass matrix, gravity term, and velocity-dependent term
Advance the state using numerical integration
Convert the resulting state sequence into animation frames
So the animation comes from actual dynamics rather than a predefined path.

Note that the result is still affected by the time step size, integrator choice, and floating-point error.

Recommended Settings
For stable and presentation-friendly results, it is recommended to:

Use RK4
Keep the time step reasonably small
Prefer smoother color modes
Enable trails, but do not make them too long
Increase output resolution if needed
FAQ
If the animation looks too slow:

Reduce total duration
Increase the time step
Lower the output frame rate
Disable unnecessary trails and panels
If colors change too abruptly:

Use a smoother color mode
Enable color smoothing
Avoid overly sharp wrapped mappings
If MP4 export fails:

Check whether ffmpeg is installed
Make sure the terminal can run ffmpeg directly
If no window appears:

This is expected
It is intended for batch jobs and headless environments
Use Cases
Teaching demonstrations
Chaotic system visualization
Numerical integration experiments
Animation generation
Physics modeling presentations
A base for future interactive applications
