# 创建GUI窗口打开图像 并显示在窗口中

from PIL import Image, ImageTk # 导入图像处理函数库
import tkinter as tk           # 导入GUI界面函数库

# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('图像显示界面')
window.geometry('1000x800')
global img_png           # 定义全局变量 图像的
var = tk.StringVar()    # 这时文字变量储存器

# 创建打开图像和显示图像函数
def Open_Img():
    global img_png
    var.set('已打开')
    Img = Image.open('D:\\学习资料\\test\\practice_Img\\agricultural00.jpg')
    img_png = ImageTk.PhotoImage(Img)

def Show_Img():
    global img_png
    var.set('已显示')   # 设置标签的文字为 'you hit me'
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack()

# 创建文本窗口，显示当前操作状态
Label_Show = tk.Label(window,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.pack()
# 创建打开图像按钮
btn_Open = tk.Button(window,
    text='打开图像',      # 显示在按钮上的文字
    width=15, height=2,
    command=Open_Img)     # 点击按钮式执行的命令
btn_Open.pack()    # 按钮位置
# 创建显示图像按钮
btn_Show = tk.Button(window,
    text='显示图像',      # 显示在按钮上的文字
    width=15, height=2,
    command=Show_Img)     # 点击按钮式执行的命令
btn_Show.pack()    # 按钮位置

# 运行整体窗口
window.mainloop()

