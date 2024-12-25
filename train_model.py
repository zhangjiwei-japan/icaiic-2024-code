import torch
import time
import copy
import argparse
import torch.nn as nn
from load_data_vegas_ave import *
from center_loss import CenterLoss
from evaluate import fx_calc_map_label
import numpy as np
import torch.optim as optim
from models.early_stopping import EarlyStopping
from v_a_model import CrossModal_NN
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate, vegas 0.01 for ave 0.001')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--dataset', default='vegas', help='dataset name: vegas or ave')
parser.add_argument('--Center_lr', default=0.05, type=float, help='learning rate, vegas 0.5 ave 0.05')
# parser.add_argument("--load_data", type=str, default= "ave_feature_norm.h5" , help="data_path")
parser.add_argument("--load_ave_data", type=str, default= "ave_feature_norm.h5" , help="data_path")
parser.add_argument("--load_vegas_data", type=str, default= "vegas_feature_norm.h5" , help="data_path")
args = parser.parse_args()

# load dataset path
dataset = args.dataset
print(".....Training the model on {} dataset.....".format(dataset))
base_dir = "./datasets/{}/".format(dataset)

early_stopping = EarlyStopping()
if dataset == 'vegas':
    batch_size = args.batch_size
    class_dim = 10
    num_epochs= 40
    Lr = args.lr
    alpha_ = 0.0001 
    beta_ = 0.0005
    test_size = 128
    save_threshold =0.87
    load_path =  base_dir + args.load_vegas_data 
    
elif dataset == 'ave':
    batch_size = args.batch_size
    class_dim = 15
    num_epochs= 120
    Lr = args.lr
    alpha_ = 0.0001 
    beta_ = 0.0001 
    test_size = 189
    save_threshold = 0.34
    load_path =  base_dir + args.load_ave_data 

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr
def Distance_loss (view1_feature, view2_feature):
    term = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()
    return term

def train_model(mid_dim,Lr, beta_, alpha_, batch_size, test_size, num_epochs):
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
 
    best_audio_2_img = 0.0
    best_img_2_audio = 0.0
    visual_feat_dim = 1024
    audio_fea_dim = 128
    best_acc = 0
   
    net = CrossModal_NN(img_input_dim=visual_feat_dim, img_output_dim=visual_feat_dim,
                        audio_input_dim=audio_fea_dim, audio_output_dim=visual_feat_dim, minus_one_dim= mid_dim, output_dim=class_dim).to(device)
    nllloss = nn.CrossEntropyLoss().to(device)
    center_loss = CenterLoss(class_dim,mid_dim).to(device)
    params_to_update = list(net.parameters())
    betas = (0.5, 0.999)
    optimizer = optim.Adam(params_to_update, Lr, betas=betas)
    optmizercenter = optim.SGD(center_loss.parameters(), lr=args.Center_lr)  # 0.05 设置weight_decay=5e-3，即设置较大的L2正则来降低过拟合。
    data_loader_visual,data_loader_audio = load_dataset_train(load_path,batch_size)
    train_losses = []
    for epoch in range(num_epochs):
        net.train()
        # current_lr = adjust_learning_rate(optimizer, epoch)
        train_loss,train_inter,train_nll,train_intra,train_dis,train_center,train_contra = 0,0,0,0,0,0,0
        for i, data in enumerate(zip(data_loader_visual, data_loader_audio)):
            optimizer.zero_grad()
            optmizercenter.zero_grad()
            if torch.cuda.is_available():
                inputs_visual = data[0][0].to(device)
                labels_visual = data[0][1].to(device)
                labels_visual = labels_visual.squeeze(1)
                inputs_audio = data[1][0].to(device)
                labels_audio = data[1][1].to(device)
                labels_audio  = labels_audio.squeeze(1)
            view1_feature, view2_feature, view1_predict, view2_predict= net(inputs_visual,inputs_audio)
            loss_id = nllloss(view1_predict,labels_visual.long()) + nllloss(view2_predict,labels_audio.long())
            loss_cent = center_loss(view1_feature,labels_visual.long()) + center_loss(view2_feature,labels_audio.long())
            loss_dis = Distance_loss(view1_feature, view2_feature)
            loss = loss_id + alpha_ *loss_dis + beta_ * loss_cent  

            train_loss += loss.item()
            train_center += loss_cent.item()
            train_dis += loss_dis.item()
            train_nll += loss_id.item()
            loss.backward()
            optimizer.step()
            optmizercenter.step()
        train_losses.append(train_loss / len(data_loader_visual))
        
        print("Epoch:{}/{} Loss:{:.2f} Nll:{:.2f} Center:{:.2f} Dist:{:.2f} Lr:{:.6f}".format(epoch,num_epochs, train_loss,
                 train_nll,train_center,train_dis,optimizer.param_groups[0]['lr']))
        if epoch > 0 and epoch%5==0:
             img_to_txt,txt_to_img,MAP,eval_loss = eval_model(net, mid_dim, beta_, alpha_ ,epoch, test_size)
             if MAP > best_acc:
                best_acc = MAP
                best_audio_2_img = txt_to_img
                best_img_2_audio = img_to_txt
                print("Best Acc: {}".format(best_acc))
                if best_acc >= save_threshold:
                    torch.save(net.state_dict(), './save_modal/audio_image_{}_best.pth'.format(args.dataset))
             early_stopping(eval_loss, net)
             if early_stopping.early_stop:
                print("Early stopping")
                break 
    return round(best_img_2_audio,4),round(best_audio_2_img,4),round(best_acc,4)

def eval_model(model,mid_dim, beta_, alpha_, epoch, test_size):
    local_time =  time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    model.eval()
    t_imgs, t_txts, t_labels,p_img,p_txt = [], [], [], [], []
    eval_loss = 0
    nllloss = nn.CrossEntropyLoss().to(device)
    center_loss = CenterLoss(class_dim,mid_dim).to(device)
    visual_test,audio_test = load_dataset_test(load_path,test_size)
    with torch.no_grad ():
        for i, data in enumerate(zip(visual_test, audio_test)):
            if torch.cuda.is_available():
                    inputs_visual = data[0][0].to(device)
                    labels_visual = data[0][1].to(device)
                    labels_visual = labels_visual.squeeze(1)
                    inputs_audio = data[1][0].to(device)
                    labels_audio = data[1][1].to(device)
                    labels_audio  = labels_audio.squeeze(1)
            t_view1_feature, t_view2_feature, predict_view1, predict_view2= model(inputs_visual,inputs_audio)
            loss_id = nllloss(predict_view1,labels_visual.long()) + nllloss(predict_view2,labels_audio.long())
            loss_cent = center_loss(t_view1_feature,labels_visual.long()) + center_loss(t_view2_feature,labels_audio.long())
            loss_dis = Distance_loss(t_view1_feature, t_view2_feature)
            loss = loss_id + alpha_ *loss_dis + beta_ * loss_cent  
            eval_loss += loss.item()
            labels_view1 = torch.argmax(predict_view1,dim=1).long()
            labels_view2 = torch.argmax(predict_view2,dim=1).long()

            t_imgs.append(t_view1_feature.cpu().detach().numpy())
            t_txts.append(t_view2_feature.cpu().detach().numpy())
            t_labels.append(labels_visual.cpu().detach().numpy())
            p_img.append(labels_view1.cpu().detach().numpy())
            p_txt.append(labels_view2.cpu().detach().numpy())
    print("Eval Loss:{:.2f}".format(eval_loss))
    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)
    p_img =  np.concatenate(p_img)
    p_txt =  np.concatenate(p_txt)
    
    img2audio = fx_calc_map_label(t_imgs, t_txts, t_labels)
    print('...Image to audio MAP = {}'.format(img2audio))
    txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('...audio to Image MAP = {}'.format(txt2img))
    Acc = (img2audio + txt2img) / 2.
    print('...Average MAP = {}'.format(Acc))


    return round(img2audio,4),round(txt2img,4),round(Acc,4), eval_loss


if __name__ == '__main__':
    mid_dim = 64
    img_to_txt,txt_to_img,MAP = train_model(mid_dim,Lr, beta_,alpha_,batch_size,test_size,num_epochs)
