三摆仿真与可视化 / Triple Pendulum Simulation and Visualization
这是一个基于数值积分的三摆系统仿真脚本。程序会先计算整个时间区间内的运动轨迹，再按设定的播放速度和帧率生成动画，支持导出 GIF 和 MP4，并提供多种颜色方案、轨迹尾迹和参数面板。

This project is a numerical simulation and visualization tool for a triple pendulum system. It first computes the full trajectory over time, then renders animation frames according to the configured playback speed and frame rate. It supports GIF and MP4 export, multiple color schemes, trajectory trails, and an information panel.

特性 / Features
三摆动力学数值仿真，采用拉格朗日形式建模
Numerical triple pendulum simulation based on Lagrangian mechanics
支持 RK4 和半隐式欧拉两种积分方式
Supports both RK4 and semi-implicit Euler integration
先计算、后渲染，便于导出和复现
Precompute first, then render, which is good for export and reproducibility
支持 GIF 和 MP4 输出
Supports GIF and MP4 export
支持多种颜色映射方案
Supports multiple color mapping modes
支持轨迹尾迹显示、平滑和长度控制
Supports trail display, smoothing, and length control
支持进度条，计算和渲染过程都有反馈
Includes progress bars for both simulation and rendering
支持右侧参数面板，主图更简洁
Includes a side panel for parameters, keeping the main plot clean
输出文件默认保存在脚本所在目录
Output files are saved in the script directory by default
默认不自动弹出窗口，适合批量运行和服务器环境
Does not auto-open a window by default, which is suitable for batch jobs and server usage
环境要求 / Requirements
Python 3.10 或更高版本
Python 3.10 or newer
依赖库 / Dependencies:
numpy
matplotlib
pillow
tqdm
如果需要导出 MP4，还需要系统可用的 ffmpeg
If you want MP4 export, ffmpeg must also be available on your system
安装依赖 / Installation
安装基础依赖：

pip install numpy matplotlib pillow tqdm

如果需要导出 MP4，请确保系统已安装 ffmpeg，并且命令行可以直接调用。

For MP4 export, make sure ffmpeg is installed and accessible from the command line.

运行方式 / How to Run
直接运行主脚本即可：

python 主脚本.py

程序会先完成仿真计算，再生成动画文件。

Run the main script directly.

The program will first simulate the motion and then generate the animation files.

可配置项 / Configurable Options
脚本通常会把这些参数集中放在文件顶部，便于修改：

初始角度：可手动输入，也可使用默认值
Initial angles: can be entered manually or use defaults
总时长：控制模拟覆盖的时间范围
Total duration: controls the simulated time span
时间步长：影响数值精度和计算速度
Time step: affects numerical accuracy and performance
帧率：影响最终动画播放节奏
Frame rate: affects the final animation playback
播放速度：用于快放或慢放
Playback speed: used for fast or slow motion
积分器：RK4 或半隐式欧拉
Integrator: RK4 or semi-implicit Euler
颜色模式：不同方式把角度、相位或能量映射到颜色
Color mode: different ways to map angles, phase, or energy to colors
颜色动态开关：是否让颜色随时间变化
Dynamic color toggle: whether colors change over time
颜色平滑：减少颜色跳变
Color smoothing: reduces abrupt color jumps
尾迹开关：是否显示摆球轨迹
Trail toggle: whether to show the pendulum trail
尾迹长度：保留多少历史轨迹
Trail length: how much history is kept
尾迹平滑：让轨迹更顺
Trail smoothing: makes the trail smoother
信息面板开关：是否显示参数说明
Info panel toggle: whether to display parameter information
网格开关：是否显示辅助网格
Grid toggle: whether to show a guide grid
输出格式：GIF、MP4，或两者都输出
Output format: GIF, MP4, or both
输出分辨率：通过画布尺寸和 dpi 控制
Output resolution: controlled by figure size and dpi
窗口显示：是否在结束后弹出预览窗口
Window display: whether to open a preview window at the end
颜色模式说明 / Color Mode Guide
sin_smooth：对角度做平滑正弦映射，颜色变化连续
Smooth sine mapping of angles with continuous transitions
cos_phase：以相位感更强的方式显示变化
Phase-like visualization with smoother cyclical feel
atan_soft：压缩极端值，过渡更柔和
Compresses extreme values for softer transitions
linear_wrap：线性环绕映射，直观但可能有跳变
Linear wrapped mapping, intuitive but may jump at boundaries
hsv_cycle：循环色相方案，适合强调周期性
Cyclic HSV color scheme, good for periodic behavior
triad_soft：三通道柔和混色，视觉更均衡
Soft tri-channel mixing with balanced appearance
cmap_theta1：按第一个角度选取色图
Uses the first angle to index the colormap
energy_heat：按能量变化着色，适合观察系统状态
Colors by energy, useful for observing system state
输出说明 / Output
输出文件会保存在脚本所在目录，而不是当前工作目录。

This avoids confusion when the script is launched from different locations.

常见输出包括：

GIF 动画
GIF animation
MP4 动画
MP4 animation
可能的中间缓存文件
Possible intermediate cache files
物理模型说明 / Physics Model
这个程序模拟的是经典三摆系统。它不是逐帧手工绘制，而是基于动力学方程进行数值求解。

This script simulates a classical triple pendulum. It is not manually animated frame by frame; instead, it numerically solves the governing equations of motion.

核心流程如下：

用三个广义坐标描述三个摆角
Three generalized coordinates describe the three angles
建立系统的质量矩阵、重力项和速度相关项
Build the mass matrix, gravity term, and velocity-dependent term
通过数值积分推进状态
Advance the state using numerical integration
再把状态序列转换成动画帧
Convert the state sequence into animation frames
因此，动画效果来自真实的动力学计算，而不是预设路径。

The resulting animation comes from actual dynamics, not from a prebuilt path.

需要注意的是，结果仍然受数值积分步长、积分器和浮点误差影响。

Note that the result is still affected by time step size, integrator choice, and floating-point error.

推荐使用方式 / Recommended Settings
如果你想要稳定、清晰、适合演示的结果，建议：

使用 RK4
Use RK4
时间步长不要设得太大
Keep the time step reasonably small
颜色模式优先选平滑型
Prefer smoother color modes
开启尾迹，但不要把尾迹长度设得过长
Enable trails, but do not make them too long
输出分辨率可以适当提高
Increase output resolution if needed
For stable and presentation-friendly results:

Use RK4
Keep the time step moderate
Prefer smooth color modes
Enable trails, but keep them reasonably short
Raise output resolution when needed
常见问题 / FAQ
如果动画太慢：

降低总时长
Reduce total duration
增大时间步长
Increase the time step
降低输出帧率
Lower the output frame rate
关闭不必要的尾迹和信息面板
Disable unnecessary trails and info panels
If the animation looks too slow:

Reduce total duration
Increase the time step
Lower the output frame rate
Disable unnecessary trails and panels
如果颜色变化太突兀：

换成更平滑的颜色模式
Switch to a smoother color mode
打开颜色平滑
Enable color smoothing
避免过于线性的环绕映射
Avoid overly linear wrapped mappings
If colors change too abruptly:

Use a smoother color mode
Enable color smoothing
Avoid overly sharp wrapped mappings
如果 MP4 导出失败：

检查 ffmpeg 是否已安装
Check whether ffmpeg is installed
确认系统命令行能直接调用 ffmpeg
Make sure ffmpeg can be invoked from the terminal
If MP4 export fails:

Check whether ffmpeg is installed
Make sure the terminal can run ffmpeg directly
如果窗口不弹出：

这是默认行为
This is the default behavior
这样更适合批量运行和无图形环境
It is better suited for batch jobs and headless environments
If no window appears:

This is expected
It is intended for batch jobs and headless environments
适用场景 / Use Cases
教学演示
Teaching demonstrations
混沌系统可视化
Chaotic system visualization
数值积分实验
Numerical integration experiments
动画生成
Animation generation
物理建模展示
Physics modeling presentations
进一步改造成交互式程序的基础
A base for future interactive applications
