
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage.morphology import distance_transform_edt
from torch.utils.data.dataloader import default_collate

import torch
import utils
import helpers
import my_custom_transforms as mtr
from model.general.sync_batchnorm import patch_replication_callback

import inference

# from model.hrcnet_base.my_net import MyNet as CutNet

########################################[ Interface ]########################################

def init_model(p):
    model = CutNet(output_stride=p['output_stride'],if_sync_bn=p['sync_bn'],special_lr=p['special_lr'],size=abs(p['ref_size']),aux_parameter=p['aux_parameter'])
    if not p['cpu']: model=model.cuda()
    model.eval()

    if p['resume'] is not None:     # 如果要恢复之前状态
        state_dict=torch.load(p['resume'])  if (not p['cpu']) else  torch.load(p['resume'] ,map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print('load from [{}]!'.format(p['resume'] ))
    return model

class Annotator(object):
    def __init__(self,p):
        self.p=p
        self.save_path=p['output']
        self.if_cuda=not p['cpu']
        self.model=init_model(p)
        self.file = Path(p['image']).name
        self.img = np.array(Image.open(p['image']).convert('RGB'))
        self.__reset()

    # pred表示模型的预测结果(二值化的掩码)（1是前景，0是背景），img表示原始图像，clicks表示点击次数，r表示标记点半径大小，
    # cb表示标记点边框粗细，b表示检测边缘线的宽度，if_first表示是否为第一个标记点。
    def __gene_merge(self,pred,img,clicks,r=0,cb=2,b=0,if_first=False):     # 该方法的主要作用是将预测结果与原始图像融合，形成一张可视化的预测结果图。
        # cv2.merge()函数返回一个三通道的图像，其中第一和第二个通道都是相同的pred值(0或255)，第三个通道的所有像素值都是0。这个三通道图像用于融合到原始图像上，以显示模型的预测结果的掩码部分，同时保留原始图像的颜色。
        pred_mask=cv2.merge([pred*255,pred*255,np.zeros_like(pred)])
        # 使用img*0.7对原始图像进行缩放，以保留一部分原始图像的颜色信息，同时使预测掩码更加突出。然后，用pred_mask*0.3对预测掩码进行缩放并加权，使得预测掩码在融合后能够以一定的透明度显示。
        result= np.uint8(np.clip(img*0.7+pred_mask*0.3,0,255))      # 通过np.clip()函数对结果进行裁剪，确保所有像素值在0-255范围内。最后，使用np.uint8()将像素值转换为整数类型，以便正确显示预测结果图。
        if b>0:
            # findContours()函数寻找所有连通区域，函数返回两个值：所有连通区域的轮廓列表(contours)以及层级信息(hierarchy)。由于我们不需要使用到层级信息，因此使用占位符_来忽略这个返回值。
            # 所有的轮廓(contours)都以numpy数组的形式存储，每个轮廓包含一组(x,y)坐标。
            contours,_=cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)     # cv2.RETR_TREE参数表示提取所有轮廓并建立完整的层级关系。cv2.CHAIN_APPROX_SIMPLE参数表示同时存储轮廓的顶点坐标，但是只保留转角点，因此可以减少存储空间。
            # result是融合后的结果图。contours是所有的连通区域的轮廓列表，作为需要绘制的对象。第三个参数-1表示绘制所有轮廓线。
            # (255, 255, 255)表示绘制轮廓线的颜色，这里选择白色。b表示线条的宽度，也即边缘线的宽度。绘制出来的轮廓线会自动添加到result中。
            cv2.drawContours(result,contours,-1,(255,255,255),b)        # drawContours()函数在融合后的结果图上绘制连通区域的轮廓线。（result上的二值化掩码和预测结果的边缘线都被绘制上了）

        if r>0:
            for pt in clicks:
                # result是融合后的结果图，将圆形绘制在这个图像上。tuple(pt[:2])表示圆心坐标，即(x,y)。pt是一个含有三个元素的列表，前两个元素表示圆心坐标，第三个元素表示圆是否被检测到。
                # tuple()函数用于将列表转换成元组，以便作为cv2.circle()函数的输入参数。r：半径。(255, 0, 0)蓝色和(0, 0, 255)红色：如果第三个元素pt[2]为1，表示圆被检测到，则圆的颜色为蓝色；否则为红色。-1表示绘制实心圆。
                cv2.circle(result,tuple(pt[:2]),r,(255,0,0) if pt[2]==1 else (0,0,255),-1)      # circle()函数在融合后的结果图上绘制圆形。圆形的大小和位置对应着模型对目标的检测结果。
                cv2.circle(result,tuple(pt[:2]),r,(255,255,255),cb)         # cb表示线条的宽度，也即圆环的宽度。
        
            if if_first and len(clicks)!=0:
                cv2.circle(result,tuple(clicks[0,:2]),r,(0,255,0),cb)       # 绿色
        return result

    def __update(self):     # 用于更新并显示当前图像界面
        self.ax.imshow(self.merge)      # 通过 imshow 将当前合并图像显示在 ax 上。（merge是多个二维数组组成的图像）
        for t in self.point_show:t.remove()     # self.point_show 是一个存储在类属性中的列表，用于存储当前界面上显示的点。这行代码的作用是将之前在图像上标记的所有点清除，为后续绘制新的点做准备。
        self.point_show=[]      # 清空列表，为后续绘制新的点做准备
        if len(self.clicks)>0:      # 如果点击次数大于0
            # self.clicks 是一个二维数组，保存了界面上所有的点击信息，每行对应一次点击。每次点击都伴随着一个数字，代表该点击在图像上所对应的位置。每行点击信息的第三列记录该次点击是正样本还是负样本，其中 1 表示正样本，0 则表示负样本。
            pos_clicks=self.clicks[self.clicks[:,2]==1,:]       # 从 self.clicks 数组中提取所有的正样本点击信息（筛选出所有第三列值为1的行（前景点）），并将其赋值给 pos_clicks 变量
            neg_clicks=self.clicks[self.clicks[:,2]==0,:]       # 背景点=0
            if len(pos_clicks)>0:self.point_show.append(self.ax.scatter(pos_clicks[:,0],pos_clicks[:,1],color='red'))       # 所有行的第一列和第二列（即坐标信息：横轴和纵轴），画散点图，加到列表中
            if len(neg_clicks)>0:self.point_show.append(self.ax.scatter(neg_clicks[:,0],neg_clicks[:,1],color='green'))     # 同上，不过是背景点
        self.fig.canvas.draw()      # 刷新 matplotlib 库中的绘图窗口，并显示最新的更新内容

    def __reset(self):      # 重置当前界面和数据
        self.clicks = np.empty([0,3],dtype=np.int64)        #创建空的二维数组，存放点击信息
        # .shape[:2] 表示获取原始图像的宽度和高度，:2 表示去除可能存在的第三个通道数（如彩色图像的通道数为 3），因此返回的是一个包含两个值的元组，分别代表图像的宽度和高度。
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)     # 创建一个和原始图像大小相同的二维数组 self.pred，并将其所有元素值设置为 0（uint是无符号整型）
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)   # 该方法的目的是将当前的预测结果和原始图像进行叠加，同时在合适的位置上绘制出所有已经标注的点。合并过程中，标注点的颜色会根据其所属的类别（正样本或负样本）而有所不同
        # 需要注意的是，在每次重置界面时，都会将当前的样本信息存储在 self.sample_backup 属性中，以便于在之后进行撤销操作时能够快速地恢复到之前的状态。
        self.sample={'img':self.img,'pre_pred':self.pred}   # 该字典的目的是用于存储每次更新界面后生成的样本信息。在这里，样本信息中包含原始图像和对应的预测结果
        # 在字典对象 self.sample 中添加了一个名为 'meta' 的键值对，该键值对的值是另一个包含三个键值对的字典对象。
        # 'source_size' 的值是一个元组，其中包含了原始图像的宽度和高度，以数组的形式存储，并使用 [::-1] 对其进行反转，以使其符合预期的格式。
        self.sample['meta']={'id': str(Path(self.p['image']).stem),'img_path' : self.p['image'], 'source_size':np.array(self.img.shape[:2][::-1])}  # .stem 属性获取文件名（不包含扩展名）
        self.sample_backup=[]       #self.sample_backup 属性用于存储每次更新界面时生成的样本信息，以便于在之后进行撤销操作时能够快速地恢复到之前的状态。
        self.hr_points=[]
        self.point_show=[]

    def __predict(self):
        if len(self.clicks)==1 and self.p['zoom_in']==0: inference.predict_wo(self.p,self.model,self.sample,self.clicks)    # 一次点击且不用方法，进行一次全局预测
        pred_tmp,result_tmp = inference.predict_wo(self.p,self.model,self.sample,self.clicks)       # 无论满不满足，都进行预测

        if len(self.clicks)>1 and self.p['model'].startswith('hrcnet') and self.p['hr_val']:
            # 这个方法会根据点击位置是否在填充区域内以及填充区域的大小来判断是否需要进行扩展区域进行局部检测，从而计算出需要扩展的半径 expand_r 和是否需要进行局部率检测的标识 if_hr，并将其返回。
            expand_r,if_hr=inference.cal_expand_r_new_final(self.clicks[-1],self.pred,pred_tmp)
            if if_hr: 
                hr_point={'point_num':len(self.clicks),'pt_hr':self.clicks[-1],'expand_r':expand_r,'pre_pred_hr':None,'seq_points_hr':None,'hr_result_src':None,'hr_result_count_src':None}
                self.hr_points.append(hr_point)

        # 预测局部区域
        self.pred= inference.predict_hr_new_final(self.p,self.model,self.sample,self.clicks,self.hr_points,pred=pred_tmp,result=result_tmp) if len(self.hr_points)>0 else pred_tmp

        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)       # __gene_merge() 方法会根据这些参数生成一个能够覆盖用户点击位置的区域，并将其赋值给 self.merge 变量。
        self.__update()     # 代码调用了 __update() 方法，用于更新一些内部状态，包括清空之前保存的局部像素点信息和填充区域等，为下一次交互做准备。

    def __on_key_press(self,event):     # 处理按键事件
        if event.key=='ctrl+z':     # 如果是 Ctrl+Z 组合键，则会根据当前 self.clicks 数组中存储的点击坐标恢复上一次操作
            if len(self.clicks)<=1:     # 如果 self.clicks 中只有一个坐标（即尚未进行过任何操作）
                self.__reset()      # 调用 __reset 方法，重置当前界面和数据
                self.__update()     # 用于更新并显示当前图像界面
            else:
                self.clicks=self.clicks[:-1,:]      # （将最新的一次点击坐标从 self.clicks 中删除）self.clicks 是一个二维数组，其中每行存储着一个点击坐标的 x 和 y 坐标值。删除最后一行数据的操作即删除了最新的一个点击坐标，意味着回到了上一步操作的状态。
                self.sample_backup.pop()        # 从 self.sample_backup 列表中弹出一个元素，以便回滚到之前的状态
                self.sample=deepcopy(self.sample_backup[-1])        # 将 self.sample 重置为回滚后的状态
                self.__predict()        # 重新调用 __predict 方法执行预测和更新图像。

        elif event.key=='ctrl+r':       # 如果按下 Ctrl+R 键，则会同样调用 __reset 方法重置所有状态并重新显示原图像
            self.__reset()
            self.__update()
        elif event.key=='escape':       # 如果按下 Esc 键，则会关闭当前 matplotlib 窗口
            plt.close()
        elif event.key=='enter':        # 如果按下 Enter 键，则会将当前预测得到的二值化图像存储到指定的文件路径（保存图像的方式可能因环境而异）
            if self.save_path is not None:      # 它的作用是将当前预测得到的二值化图像以指定的保存路径保存为一个图像文件
                Image.fromarray(self.pred*255).save(self.save_path)     # 使用 Image.fromarray 方法将其转换为了一个 PIL.Image 类型的对象，并通过 *255 操作将值域扩展到 [0, 255] 的范围内
                print('save mask in [{}]!'.format(self.save_path))      # 当存储完成后，会打印一条消息提示保存的路径，并关闭窗口。
            plt.close()

    def __on_button_press(self,event):      # 处理鼠标事件
        if (event.xdata is None) or (event.ydata is None):return        # 通过 event.xdata 和 event.ydata 判断当前的鼠标点击位置是否在图像范围内，如果不在，则什么也不做
        if event.button==1 or  event.button==3:     # 判断触发事件的是左键（event.button==1）还是右键（event.button==3）
            x,y= int(event.xdata+0.5), int(event.ydata+0.5)     # 获取当前点击位置的 x 和 y 坐标，并将其存储到 self.clicks 数组中（self.clicks=np.append(self.clicks,np.array([[x,y,(3-event.button)/2]],dtype=np.int64),axis=0)）
            self.clicks=np.append(self.clicks,np.array([[x,y,(3-event.button)/2]],dtype=np.int64),axis=0)       # 将当前点击位置信息存储到 self.clicks 数组中
            self.__predict()        # 将当前点击位置信息存储到 self.clicks 数组中
            self.sample_backup.append(deepcopy(self.sample))     # 将 self.sample 列表的最后一个元素深度复制一份，并将其添加到 self.sample_backup 列表中，以便在撤销操作时恢复到之前的状态。

    def main(self):
        self.fig = plt.figure('Annotator')      # 创建了一个名为 Annotator 的窗口，并将它赋值给 self.fig 成员变量,为该窗口绑定了两个事件处理函数 self.__on_key_press 和 self.__on_button_press，分别处理按键和鼠标点击事件
        self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.fig.canvas.mpl_connect("button_press_event",  self.__on_button_press)
        self.fig.suptitle('( file : {} )'.format(self.file),fontsize=16)        # 在窗口顶部设置了标题，显示当前打开文件名
        self.ax = self.fig.add_subplot(1,1,1)       # 创建了一个子图 self.ax 并将其设为无坐标系(该方法的参数 1,1,1 表示将当前图形分成 1 行 1 列，使用第 1 个子图来绘制)
        self.ax.axis('off')
        self.ax.imshow(self.merge)  # 显示了合并后的图像 self.merge。
        plt.show()




if __name__ == "__main__":
    p=utils.create_parser()
    exec('from model.{}.my_net import MyNet as CutNet'.format(p['model']))
    anno=Annotator(p)
    anno.main()
