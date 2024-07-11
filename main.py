#[General module]
import os
import time
import random
import shutil
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import ImageFile

#[Other module]
import torch
from PIL import Image

#[Personal module]
import utils
import helpers
import my_custom_transforms as mtr
from dataloader_cut import GeneralCutDataset
from model.general.sync_batchnorm import patch_replication_callback
import inference

from model.hrcnet_base.my_net import MyNet as CutNet
# from model.hrcnet_base_fca.fcanet import FCANet as CutNet

#[Basic setting]
TORCH_VERSION=torch.__version__
torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=3, suppress=True)

#[Trainer]
class Trainer(object):
    def __init__(self,p):               # 用于初始化一个对象（即一个类的实例）
        self.p=p

        self.transform_train = mtr.IIS(p)       # 进行一些transform操作（自己写的很多函数）

        self.train_set = GeneralCutDataset(os.path.join(p['dataset_path'],p['dataset_train']),list_file='train.txt',max_num=p['max_num'],           # 加载数据集的一些操作
                                           batch_size=-p['batch_size'], remove_small_obj=p['remove_small_obj'],gt_mode=p['gt_mode'],transform=self.transform_train, if_memory=p['if_memory'])

        self.val_robot_sets = [ GeneralCutDataset(os.path.join(p['dataset_path'],dataset_val),list_file='val.txt',max_num=p['max_num_robot_val'],batch_size=0,          # 加载val.txt
                                remove_small_obj=0,gt_mode=p['gt_mode'],transform=None, if_memory=p['if_memory']) for dataset_val in p['dataset_vals'] ]            # 验证期间并不需要数据增强

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=p['batch_size'], shuffle=True, num_workers=p['num_workers'])     # 加载数据集

        self.model = CutNet(output_stride=p['output_stride'],if_sync_bn=p['sync_bn'],special_lr=p['special_lr'],size=abs(p['ref_size']),aux_parameter=p['aux_parameter']).cuda()        # 加载模型

        if len(p['gpu_ids'])>1:     # 如果使用多个GPU，则将model变量封装在torch.nn.DataParallel中，以便在多个GPU上并行处理输入数据。
            self.model = (torch.nn.DataParallel(self.model, device_ids = p['gpu_ids']))     # torch.nn.DataParallel 是 PyTorch 中用于数据并行处理的模块。它可以自动将模型分配到多个 GPU 上，并使用多个 GPU 并行处理输入数据。使用是要求gpu数量必须大于等于2
            self.model_src=self.model.module
        else:
            self.model_src=self.model

        self.optimizer = utils.get_optimizer(p['optimizer'], self.model_src.get_train_params_lr(lr=p['learning_rate']))
        self.scheduler = utils.get_lr_scheduler(p['lr_scheduler'], self.optimizer)
        self.best_metric = None

        if p['resume'] is not None:         # 如果要恢复之前状态
            self.model.load_state_dict(torch.load(p['resume']))
            print('Load model from [{}]!'.format(p['resume']))

    def training(self, epoch):
        print('Training :')
        mtr.current_epoch = epoch
        loss_total,loss_show =0,'Loss: None'
        self.model.train()

        if self.p['hr_backbone_frozen']:self.model_src.freeze_main_bn()    # 冻结指定模块的bn层的参数
        tbar = tqdm(self.train_loader)              # 添加进度条
        for i, sample_batched in enumerate(tbar):
            self.optimizer.zero_grad()          # 梯度归零

            if self.p['seq_mode']:                                  # 这个代码段的主要作用是根据所选模型和参数设置，动态地运行不同类型的前向传递和损失计算。
                output,loss_items=self.model(sample_batched)        # 使用模型对批量数据进行前向传递，并计算损失值
            else:
                if self.p['model'].startswith('hrcnet'):            # 检查模型的名称是否以字符串'hrcnet'开头
                    output,loss_items=self.model(sample_batched)
                else:
                    output=self.model(sample_batched)               # 只进行前向传播
                    loss_items=self.model_src.get_loss_union(output,sample_batched=sample_batched)      # 计算由model_src对象提供的联合损失

            loss_total+=loss_items.mean(dim=0).cpu().numpy()            # 将一个数据批次中所有样本的平均损失加到一个累积的总损失中
            self.optimizer.step()           # 基于目前已经积累的梯度执行一步优化，更新模型的权重，以提高其在给定任务上的性能。

            loss_show='Loss: {:.3f}{}'.format(loss_total.sum()/(i + 1), '' if len(loss_total)==1 else loss_total / (i + 1))            # 将一个数据批次中所有样本的平均损失加到一个累积的总损失中
            tbar.set_description(loss_show)         # 设置/修改进度条描述。

            if (self.p['itis_pro']>0) and (not self.p['seq_mode']):     # 该语句块的功能是将模型输出的检测框结果进行处理，并将处理后的结果和其他信息存储在一个记录器对象mtr的属性中
                preds = np.uint8(self.model_src.get_result(output)>self.p['pred_tsh_itis'])     # 这里使用numpy中的广播机制，将比较结果转换为布尔值，并通过np.uint8()函数将其转换为0/1的数值类型，存储在preds数组中。
                for j in range(sample_batched['img'].shape[0]):         # 循环遍历输入 sample_batched['img'] 中的每个批次样本图像
                    id=sample_batched['meta']['id'][j]      # 从元数据字典中检索出相应的ID
                    mtr.record_itis['pred'][id]= helpers.encode_mask(preds[j,:,:])      # 使用helpers.encode_mask()函数，将preds中第j个样本的预测结果转换为可视化的掩码，并存储到mtr.record_itis字典中'pred'键下的id位置。
                    seq_points=sample_batched['seq_points'][j].numpy()      # 获取当前样本的检测框四点坐标，存储在seq_points变量中。
                    mtr.record_itis['seq_points'][id]= seq_points[seq_points[:,2]!=-1].tolist()     # 将检测框四点坐标中有效部分（第三维坐标为非-1）转换为Python列表，并存储到mtr.record_itis字典中'seq_points'键下的id位置。
                    mtr.record_itis['crop_bbox'][id]= list(sample_batched['meta']['crop_bbox'][j].numpy())      # 获取当前样本的裁剪框坐标，并将其转换为Python列表，存储到mtr.record_itis字典中'crop_bbox'键下的id位置。

                    if self.p['random_flip']:mtr.record_itis['if_flip'][id] = int(sample_batched['meta']['if_flip'][j])     # 如果训练过程中进行了随机翻转操作，则获取当前样本是否被翻转，并将其转换为整型数值，存储到mtr.record_itis字典中'if_flip'键下的id位置。
                    if self.p['random_rotate']:mtr.record_itis['rotate'][id] = int(sample_batched['meta']['rotate'][j])     # 如果训练过程中进行了随机旋转操作，则获取当前样本的旋转角度，并将其转换为整型数值，存储到mtr.record_itis字典中'rotate'键下的id位置。

        print(loss_show)

    def validation_robot(self, epoch, if_hrv=False):
        torch.backends.cudnn.benchmark = False      # 当我们需要保证程序运行时间的稳定性时，应该将 torch.backends.cudnn.benchmark 设置为 False；当我们希望尽可能地提高运行性能时，可以将其设置为 True。
        print('+'*79)
        print('if_hrv : ',if_hrv)
        self.model.eval()       # 这是因为在评估模式下，模型不进行梯度计算和参数更新，仅仅进行前向传播，这能够有效减少内存的占用，并且不会影响模型的学习效果。此外，评估模式下还会关闭一些具有随机性质的操作，如 dropout 和随机翻转等，从而确保预测结果的稳定性。
        for index, val_robot_set in enumerate(self.val_robot_sets):
            dataset=self.p['dataset_vals'][index]       #获取 self.p['dataset_vals'][index] 对应的值，并将其存储在 dataset 变量中。

            if dataset=='DAVIS':
                self.p['eval_size']=512
                self.p['hr_size']=512
            else:
                self.p['eval_size']=self.p['ref_size']      # 这个设置是384
                self.p['hr_size']=self.p['ref_size']

            print('Validation Robot: [{}]'.format(dataset))
            max_miou_target=max(self.p['miou_target'][index])

            record_other_metrics = {}

            record_dict={}

            ImageFile.LOAD_TRUNCATED_IMAGES = True

            for i, sample in enumerate(tqdm(val_robot_set)):        # 使用 enumerate() 函数将 val_robot_set 遍历出来并标上索引，每次从中取出一个样本   将val_robot_set传给 tqdm 函数，然后循环遍历这个可迭代对象。每当一个元素被处理完成时，tqdm 函数就会自动更新进度条的状态。
                id = sample['meta']['id']
                gt = np.array(Image.open(sample['meta']['gt_path']))        # 找到地址，打开图片，转为numoy格式
                pred = np.zeros_like(gt)        # 创建一张与 gt 相同大小的全零数组，并保存在 pred 变量中
                seq_points=np.empty([0,3],dtype=np.int64)       # 创建一个空的 ndarray 类型的 seq_points 变量，用于存储该样本的所有分割点坐标
                id_preds,id_ious=[helpers.encode_mask(pred)],[0.0]      # 分别将当前的 pred 编码为九位二进制数，并将其保存在 id_preds 列表中，并将当前的 iou 值 0.0 保存在 id_ious 列表中。

                id_other_metrics= {metric: [0.0] for metric in self.p['other_metric']}      # 首先通过 self.p['other_metric'] 获取到一个列表，该列表中存储了需要计算的指标名称。然后在字典生成式中遍历该列表，对于列表中的每个元素，将其作为键并初始化为 0.0，最终生成一个包含所有指标的字典对象 id_other_metrics。

                hr_points=[]        # 该列表用于存储局部率图片的那些分割点的坐标
                sample['pre_pred']=pred

                # predict_wo() 函数是进行分割推理的核心函数，它的作用是对一张图像进行分割，并返回分割后得到的结果。self.p：分割器的配置参数；self.model_src：模型的源路径；sample：要进行分割的图像样本；np.array([helpers.get_next_anno_point(np.zeros_like(gt), gt)],dtype=np.int64)：上一次预测得到的分割点的位置。
                # 函数名称中的 wo 表示该函数是在没有采用数据增强的情况下进行推断得到的结果。
                if self.p['zoom_in']==0:inference.predict_wo(self.p,self.model_src,sample,np.array([helpers.get_next_anno_point(np.zeros_like(gt), gt)],dtype=np.int64)) #add -wo

                # 在整个推理过程中，不断更新的 seq_points 列表记录了目前为止的所有预测点位置，而在每次迭代中前面的预测点都会被用来计算新的预测点位置。
                # 同时，对于 HRCNet 模型，在循环过程中还会记录符合条件的 HR 点信息，并使用它们对预测结果进行修正，以得到更准确的分割结果。
                for point_num in range(1, self.p['max_point_num']+1):       # 这段代码是一个分割推理的循环过程。该循环会在每次迭代中计算预测结果与真实分割结果之间的评估指标，并根据预测点数是否达到设定值或已经达到最大 IoU 值等条件来终止循环。
                    pt_next = helpers.get_next_anno_point(pred, gt, seq_points)     # 根据上一次预测得到的分割点位置 seq_points 和真实分割结果 gt，计算下一个预测点的位置 pt_next。
                    seq_points=np.append(seq_points,[pt_next],axis=0)       # 将新的预测点位置加入到 seq_points 中，并使用它们进行新一轮的分割推理，得到新的分割结果 pred_tmp。
                    pred_tmp,result_tmp = inference.predict_wo(self.p,self.model_src,sample,seq_points)
                    if point_num>1 and self.p['model'].startswith('hrcnet') and if_hrv and p['hr_val_setting']['pfs']!=0:       # 对于 HRCNet 模型，如果启用了水平翻转增强且当前处于第二个预测点之后，则计算当前预测点周围的扩展距离和 HR-Val 值，并将符合条件的点加入到 hr_points 列表中。
                        expand_r,if_hr=inference.cal_expand_r_new_final(pt_next,pred,pred_tmp)
                        if if_hr:
                            hr_point={'point_num':point_num,'pt_hr':pt_next,'expand_r':expand_r,'pre_pred_hr':None,'seq_points_hr':None,'hr_result_src':None,'hr_result_count_src':None,'img_hr':None,'pred_hr':None,'gt_hr':None}
                            hr_points.append(hr_point)

                    # 如果 hr_points 非空，则使用其中的 HR 点信息对预测结果进行修正，得到修正后的预测结果 pred。（局部视图相关）
                    pred= inference.predict_hr_new_final(self.p,self.model_src,sample,seq_points,hr_points,pred=pred_tmp,result=result_tmp) if len(hr_points)>0 else pred_tmp

                    for metric in id_other_metrics: id_other_metrics[metric].append(helpers.get_metric(pred,gt,metric))     # 根据预测结果和真实分割结果计算各个评估指标，并将其存储到 id_other_metrics 和 id_ious 列表中。

                    miou = ((pred==1)&(gt==1)).sum()/(((pred==1)|(gt==1))&(gt!=255)).sum()
                    id_ious.append(miou)
                    id_preds.append(helpers.encode_mask(pred))
                    if (np.array(id_ious)>=max_miou_target).any() and point_num>=self.p['record_point_num']:break       # 如果当前最大 IoU 值已经达到了预先设定的阈值 max_miou_target，并且当前预测点数已经达到了 self.p['record_point_num']，则跳出循环。

                record_dict[id]={'clicks':[None]+[tuple(pt) for pt in seq_points],'preds':id_preds,'ious':id_ious}
                record_other_metrics[id]=id_other_metrics

            # #[used for record result file]
            # if self.p['record_point_num']>5:
            #     np.save('{}/{}~{}~{}~infos.npy'.format(self.p['snapshot_path'],'FocusCut',dataset,'val'),record_dict,allow_pickle=True)


            # 这段代码是用于评估分割模型性能的函数。首先，将待评估的样本集按照不同的尺寸划分到不同的集合中，并针对每个集合计算出一些评价指标，包括 NoC-mIoU、mNoC 和 NoF 等。
            # 其中，NoC-mIoU 是不考虑数量级差异时得到的 mIoU，mNoC 指多个 IoU 阈值下的平均 NoC，NoF 则是指一个样本中被错误分类的前景点的数量。
            # 接着，根据设定的其它评价指标，计算出相应的平均值。最后，从所有集合中选取第一个集合的 mNoC 和 NoC-mIoU 作为当前的评价指标，并返回该指标。
            for size_key,size_ids in val_robot_set.get_size_div_ids(size_div=[0.045,0.230][:] if dataset in ['TBD'] else None).items():         #否则，不做尺寸限制
                print('[{}]({}):'.format(size_key,len(size_ids)))
                if len(size_ids)==0: print('N/A');continue
                noc_miou=helpers.get_noc_miou(record_dict,ids=size_ids,point_num=self.p['record_point_num'])        # record_dict：一个字典，包含预测的语义分割掩模和相应的真实标签。
                mnoc=helpers.get_mnoc(record_dict,ids=size_ids,iou_targets=self.p['miou_target'][index])
                print('NoC-mIoU : [{}]'.format(' '.join(['{:.3f}'.format(t) for t in noc_miou ])))
                print('mNoC : {}'.format('  '.join(['{:.3f} (@{:.2f})'.format(t1,t2) for t1,t2 in zip(mnoc,self.p['miou_target'][index])])))        #zip是打包成元组

                nof=helpers.get_nof(record_dict,ids=size_ids,iou_targets=self.p['miou_target'][index],max_point_num=self.p['max_point_num'])
                print('NoF : {}'.format('  '.join(['{} (@{:.2f})'.format(t1,t2) for t1,t2 in zip(nof,self.p['miou_target'][index])])))

            # 该方法似乎遍历了列表 "self.p['other_metric']" 中的所有指标，
            # 对于每个指标，它从 "record_other_metrics" 字典中获取相应键的值。然后使用NumPy的“mean”函数沿着第一个轴（即跨越该特定指标的所有值）取这些值的平均值，并将结果赋给变量“metric_mean”。
            for metric in self.p['other_metric']:
                metric_mean=np.array([v[metric][:self.p['record_point_num']+1] for v in record_other_metrics.values()]).mean(axis=0)
                print('{} : {}'.format(metric,metric_mean))

            if index==0:current_metric=[mnoc[0],noc_miou]           # 该语句确定当前的指标值是由 mnoc 和 noc_miou 两个指标共同决定的。也就是说，只有当两个指标都达到要求时，当前指标的值才被视为合格

        torch.backends.cudnn.benchmark = True
        return current_metric


if __name__ == "__main__":
    p=utils.create_parser()         # p是参数定义，他最后把它变成vars形式，会让他变成字典

    random.seed(p['seed'])          # 固定随机数的步骤（这样会让不管怎么训练，每次训练出来结果会是一样的）
    np.random.seed(p['seed'])
    torch.manual_seed(p['seed'])

    exec('from model.{}.my_net import MyNet as CutNet'.format(p['model']))      # exec()可以用来执行字符串类型的代码。这个函数是一个动态执行器，它可以接受一个字符串类型的参数，然后把这个字符串解析成可执行的代码，并执行这些代码。

    if p['clear_snapshot']:shutil.rmtree('./snapshot');exit()           # rmtree是递归删除文件树。如果在参数 p 中指定了清除快照的标志 clear_snapshot，则删除当前工作目录下名为 snapshot 的文件夹，并退出程序。
    os.makedirs(p['snapshot_path'],exist_ok=False)
    if p['backup_code']: shutil.copytree('.', '{}/code'.format(p['snapshot_path']), ignore=shutil.ignore_patterns('snapshot','__pycache__'))        # 递归复制文件树，如果在参数 p 中指定了备份代码的标志 backup_code，则将当前工作目录中除了 snapshot 和 __pycache__ 以外的所有文件和文件夹复制到名为 code 的文件夹中，并将该文件夹放入快照路径下。
    utils.set_log_file('{}/log.txt'.format(p['snapshot_path']))     # 将日志文件路径设置为 {快照路径}/log.txt。
    start_time=time.time()
    print('Start time : ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time)))
    print('---[   Note:({})    ]---'.format(p['note']))
    print('Using net : [{}]'.format(p['model']))
    print('-'*79,'\ninfos : ' , p, '\n'+'-'*79)

    mine =Trainer(p)        # 实例化

    if p['val']:        # 默认是false
        mine.validation_robot(0,p['hr_val'])
    else:
        if TORCH_VERSION[0]=='0': mine.scheduler.step()     # 这段代码是在训练过程中用于更新学习率的语句。具体来说，它会执行一次学习率调度器的 step() 方法，在训练过程中动态调整学习率。
        for epoch in range(p['epochs']):
            lr_str = ['{:.7f}'.format(i) for i in mine.scheduler.get_lr()]      # 它会调用学习率调度器的 get_lr() 方法，返回当前的学习率。
            print('-'*79+'\n'+'Epoch [{:03d}]=>  |-lr:{}-|  ({})\n'.format(epoch, lr_str,p['note']))

            #training
            if p['train_only_epochs']>=0:
                mine.training(epoch)

                # isinstance() 函数是 Python 的内置函数，用于判断一个对象是否属于指定的类型
                if (p['lr_scheduler_stop'] is None ) or (isinstance(p['lr_scheduler_stop'],int) and epoch<p['lr_scheduler_stop']) or (isinstance(p['lr_scheduler_stop'],float) and mine.scheduler.get_lr()[0]>p['lr_scheduler_stop']):
                    mine.scheduler.step()

            if epoch<p['train_only_epochs']: continue       # 跳过本轮次余下语句

            #validation-robot
            # if (epoch+1) % p['val_robot_interval']==0:
            # if epoch > -1:
            if epoch >= 29:
                if p['save_val_robot']=='each': torch.save(mine.model_src.state_dict(), '{}/model-epoch-{}.pth'.format(p['snapshot_path'],str(epoch).zfill(3)))     # str(epoch).zfill(3)表示将epoch数转换为三位数，并在左侧补零
                current_metric = mine.validation_robot(epoch, False)
                # mine.model_src.apply(lambda module: setattr(module, 'attention', True))     # 注意力相关
                # current_metric_with_attention = mine.validation_robot(epoch, True)
                current_metric = mine.validation_robot(epoch, True)       # 修改部分
                # mine.model_src.apply(lambda module: setattr(module, 'attention', False))        # 注意力相关
                if p['save_val_robot'] == 'best' and (mine.best_metric is None or current_metric[0] < mine.best_metric[0]):
                    mine.best_metric = current_metric
                    torch.save(mine.model_src.state_dict(), '{}/best.pth'.format(p['snapshot_path']))
                    print('Epoch [{:03d}]'.format(epoch))

    end_time=time.time()
    print('-'*79)
    print('End time : ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time)))
    delta_time=int(end_time-start_time)
    print('Delta time :   {}:{}:{}'.format(delta_time//3600,(delta_time%3600)//60,(delta_time%60)))
    print('Saved in [{}]({})!'.format(p['snapshot_path'],p['note']))




















