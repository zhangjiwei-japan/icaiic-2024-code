import torch
import time
import copy
import argparse
import torch.nn as nn
from load_data_vegas_ave import *
from evaluate import fx_calc_map_label
import numpy as np
import torch.optim as optim
# from models.img_text_models import Cross_Modal_Net
from v_a_model import CrossModal_NN
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='vegas', help='dataset name: vegas or ave')
parser.add_argument("--load_ave_data", type=str, default= "ave_feature_norm.h5" , help="data_path")
parser.add_argument("--load_vegas_data", type=str, default= "vegas_feature_norm.h5" , help="data_path")
args = parser.parse_args()

print('...Data loading is beginning...')

# load dataset path
dataset = args.dataset
print(".....Testing the model on {} dataset.....".format(dataset))
base_dir = "./datasets/{}/".format(dataset)

if dataset == 'vegas':
    class_dim = 10
    test_size = 128
    save_path = './best_modal/vegas/audio_image_vegas_best.pth'
    load_path =  base_dir + args.load_vegas_data 
    
elif dataset == 'ave':
    class_dim = 15
    test_size = 189
    save_path = './best_modal/ave/audio_image_ave_best.pth'
    load_path =  base_dir + args.load_ave_data 

visual_feat_dim = 1024
word_vec_dim = 128
mid_dim = 64
net = CrossModal_NN(img_input_dim=visual_feat_dim, img_output_dim=visual_feat_dim,
                        audio_input_dim=word_vec_dim, audio_output_dim=visual_feat_dim, minus_one_dim= mid_dim, output_dim=class_dim).to(device)
def test_model(net,save_path,test_size):
    local_time =  time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    net.load_state_dict(torch.load(save_path))
    net.eval()
    t_imgs, t_txts, t_labels,p_img,p_txt = [], [], [], [], []
    visual_test,audio_test = load_dataset_test(load_path,test_size)
    with torch.no_grad():
        for i, data in enumerate(zip(visual_test, audio_test)):
            if torch.cuda.is_available():
                    inputs_visual = data[0][0].to(device)
                    labels_visual = data[0][1].to(device)
                    # print(labels_visual.shape)
                    labels_visual = labels_visual.squeeze(1)
                    inputs_audio = data[1][0].to(device)
                    labels_audio = data[1][1].to(device)
            t_view1_feature, t_view2_feature, predict_view1, predict_view2 = net(inputs_visual,inputs_audio)
            labels_view1 = torch.argmax(predict_view1,dim=1).long()
            labels_view2 = torch.argmax(predict_view2,dim=1).long()

            t_imgs.append(t_view1_feature.cpu().detach().numpy())
            t_txts.append(t_view2_feature.cpu().detach().numpy())
            t_labels.append(labels_visual.cpu().detach().numpy())
            p_img.append(labels_view1.cpu().detach().numpy())
            p_txt.append(labels_view2.cpu().detach().numpy())

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

    return round(img2audio,4),round(txt2img,4),round(Acc,4)


if __name__ == '__main__':
    test_model(net,save_path,test_size)
