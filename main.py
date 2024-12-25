from typing import ByteString
from v_a_model import *
import h5py
import torch as t
from save_acc_result import *
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from center_loss import CenterLoss
from evaluate import fx_calc_map_label
from load_dataset import load_dataset_train, load_dataset_test
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
local_time =  time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))  
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate, vegas 0.01 for ave 0.001')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--Center_lr', default=0.05, type=float, help='learning rate, vegas 0.5 ave 0.05')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--fig_attention_1', default= True, type=bool, help='fig_attention')
parser.add_argument('--fig_attention_2', default= True, type=bool, help='fig_attention')
parser.add_argument('--dataset', default='vegas', help='dataset name: vegas or ave]')
parser.add_argument('--l_id', default=1, type=float,help='loss paraerta')
parser.add_argument('--l_center', default=0.01, type=float,help='loss paraerta')
parser.add_argument('--l_dis', default=0.1, type=float,help='loss paraerta')
parser.add_argument('--save_acc_name', default="results/Train_Accuracy_{}".format(local_time), help='save_acc_name')
parser.add_argument("--load_ave_data", type=str, default= "dataset/AVE_feature_updated_squence.h5" , help="data_path")
parser.add_argument("--load_vegas_data", type=str, default= "dataset/vegas_feature.h5" , help="data_path")
args = parser.parse_args()

dataset = args.dataset
print(".....Training the model on {} dataset.....".format(dataset))
if dataset == 'vegas':
    data_path = args.load_vegas_data
    class_dim = 10
    num_epoch = 300
    batch_size = args.batch_size#128 使用较小的batch size 那么一个epoch就可以进行更多次的权值更新
    # lamada = args.l_id # best: lamada:1
    # alpha = args.l_center # best: alpha:0.001
    # beta = args.l_dis # best: beta:0.1
    show_feature_fig = False #True #False #True #  True 
    test_dataset_size = 5621
    save_threshold = 0.87
    save_acc_results = "results/"+ args.save_acc_name + "{}.log".format(dataset)
    
elif dataset == 'ave':
    #  LR: 0.01 Center_lr: 0.005 Batch_size: 128 Best_epoch: 165
    data_path = args.load_ave_data
    class_dim = 15
    num_epoch = 400
    batch_size = args.batch_size
    # lamada = args.l_id # best: lamada:1
    # alpha = args.l_center # best: alpha:0.001
    # beta = args.l_dis # best: beta:0.1
    show_feature_fig = False # True
    test_dataset_size = 189
    save_threshold = 0.34
    # lamada:1 alpha:0.001 beta:0.1 gamma:0.1 
    save_acc_results ="results/"+ args.save_acc_name + "{}.log".format(dataset)

def Distance_loss (view1_feature, view2_feature):
    term = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()
    return term

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim
visual_feat_dim = 1024
word_vec_dim = 128
mid_dim = 128
net = CrossModal_NN(img_input_dim=visual_feat_dim, img_output_dim=visual_feat_dim,
                        audio_input_dim=word_vec_dim, audio_output_dim=visual_feat_dim, minus_one_dim= mid_dim, output_dim=class_dim).to(device)
center_loss = CenterLoss(class_dim,mid_dim).to(device)
# nllloss = nn.NLLLoss().to(device)
distance_loss = nn.MSELoss().to(device)
nllloss = nn.CrossEntropyLoss().to(device)
def adjust_learning_rate(optimizer, epoch,num_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        lr = args.lr * (epoch + 1) / 20
    elif epoch >= 20 and epoch < 0.25*num_epoch:
        lr = args.lr
    elif epoch >=  0.25*num_epoch and epoch < 0.50*num_epoch:
        lr = args.lr * 0.1
    elif epoch >= 0.50*num_epoch and epoch < 0.75*num_epoch:
        lr = args.lr * 0.01
    elif epoch >= 0.75*num_epoch:
        lr = args.lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr
if args.optim == 'sgd':
    ignored_params =  list(map(id, net.audio_fine_grained_feature.parameters())) \
                    + list(map(id, net.visual_fine_grained_feature.parameters())) \
                    + list(map(id, net.out_layer.parameters())) \
                    + list(map(id, net.classifier_audio.parameters()))\
                    + list(map(id, net.classifier_visual.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.audio_fine_grained_feature.parameters(), 'lr': args.lr},
        {'params': net.visual_fine_grained_feature.parameters(), 'lr': args.lr},
        {'params': net.out_layer.parameters(), 'lr': args.lr},
        {'params': net.classifier_audio.parameters(), 'lr': args.lr},
        {'params': net.classifier_visual.parameters(), 'lr': args.lr}
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    optmizercenter = optim.SGD(center_loss.parameters(), lr=args.Center_lr)  # 0.05 设置weight_decay=5e-3，即设置较大的L2正则来降低过拟合。

def train_cross_modal(lamada,alpha,beta,load_path):
    print("Training model by LR: {} Center_lr: {} batch_size: {} lamada: {} alpha: {} beta: {} "\
        .format(args.lr,args.Center_lr,args.batch_size,lamada,alpha,beta))
    best_acc = 0
    best_epoch = 0
    best_t_2_i =0
    best_i_2_t =0

    data_loader_visual,data_loader_audio = load_dataset_train(load_path,batch_size)
    # optimizer = t.optim.SGD(net.parameters()
    #                        ,lr= args.lr, momentum=0.9, weight_decay=5e-3)
    # scheduler = t.optim.lr_scheduler.StepLR(optmizercenter, step_size=10, gamma=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optmizercenter, milestones=[int(num_epoch * 0.56), int(num_epoch * 0.78)], gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                milestones=[int(num_epoch * 0.56), int(num_epoch * 0.78)],
    #                gamma=0.1, last_epoch=-1)

    for epoch in range(num_epoch+1):
        current_lr = adjust_learning_rate(optimizer, epoch,num_epoch)
        train_loss,train_center,train_nll,train_dis = 0,0,0,0
        net.train()
        feat1 = []
        feat2 = []
        target = []
        for i, data in enumerate(zip(data_loader_visual, data_loader_audio)):
            optimizer.zero_grad()
            optmizercenter.zero_grad()

            visual_data_input = data[0][0]
            labels_visual = data[0][1]
            audio_data_input = data[1][0]
            labels_audio = data[1][1]

            label_input = Variable(labels_audio)
            label= label_input.to(device)

            audio_data_input = Variable(audio_data_input)
            inputs_audio= audio_data_input.to(device)   
            
            visual_data_input = Variable(visual_data_input)
            inputs_visual= visual_data_input.to(device)
        
            final_feature_v,final_feature_a,final_classifier_v,final_classifier_a = net(inputs_visual,inputs_audio) #output audio and visual features
            # print(final_feature_v.shape,final_feature_a.shape)
            loss_cent = center_loss(final_feature_v,label) + center_loss(final_feature_a,label)
            loss_id = nllloss(final_classifier_v,label) + nllloss(final_classifier_a,label)
            loss_dis = Distance_loss(final_feature_v,final_feature_a)
            # loss_dis = distance_loss(final_feature_v,final_feature_a)
            # loss_dis = trisim_loss(final_feature_v,final_feature_a)
            loss = lamada *loss_id + alpha* loss_cent + beta*loss_dis
            # with autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            optmizercenter.step()

            feat1.append(final_feature_a) 
            feat2.append(final_feature_v)    
            target.append(label)

            train_loss += loss.item()
            train_center += loss_cent.item()
            train_nll += loss_id.item()
            train_dis += loss_dis.item()
        # scheduler.step()
        # print("Epoch:{}/{} Loss:{:.2f} Nll:{:.2f} Dis:{:.2f} Center:{:.2f} Lr:{:.6f}/{:.4f}".format(epoch,num_epoch, train_loss,
        #          train_nll,train_dis,train_center,optimizer.param_groups[0]['lr'],optmizercenter.param_groups[0]['lr']))
        # print("Epoch:{}/{} Loss:{:.2f} Nll:{:.2f} Dis:{:.2f} Center:{:.2f} Lr:{:.6f}/{:.4f}".format(epoch,num_epoch, train_loss,
        #          train_nll,train_dis,train_center,optimizer.param_groups[0]['lr'],optmizercenter.param_groups[0]['lr']))
        print("Epoch:{}/{} Loss:{:.2f} Nll:{:.2f} Dis:{:.2f} Center:{:.2f} Lr:{:.6f}/{:.6f}".format(epoch,num_epoch, train_loss,
                 train_nll,train_dis,train_center,current_lr,optmizercenter.param_groups[0]['lr']))
        
        features1 = t.cat(feat1,0)
        features2 = t.cat(feat2,0)
        targets = t.cat(target,0)
        save_type1 = "audio"
        save_type2 = "visual"
        save_type3 = "visual_audio"
        if show_feature_fig :
            if epoch > 0 and epoch%5==0:
                print("....Show feature plot....")
                plot_tsne_castter_visual(out_class_size,features1.data.cpu().numpy(),targets.data.cpu().numpy(), epoch,"mscnn-cross-{}-{}-{}".format(save_type2,dataset,local_time),"results_image/show_image_new/")
                plot_tsne_castter_audio(out_class_size,features2.data.cpu().numpy(),targets.data.cpu().numpy(), epoch,"mscnn-cross-{}-{}-{}".format(save_type1,dataset,local_time),"results_image/show_image_new/")
                plot_tsne_castter_v_a(out_class_size,features1.data.cpu().numpy(),features2.data.cpu().numpy(),targets.data.cpu().numpy(), epoch,"mscnn-cross-{}-{}-{}".format(save_type3,dataset,local_time),"results_image/show_image_new/")
                # plot_umap_visual(features1.data.cpu().numpy(),targets.data.cpu().numpy(), epoch,"mscnn-cross-{}-{}-{}".format(save_type2,dataset,local_time),"results_image/show_image_new/")
                # plot_umap_audio(features2.data.cpu().numpy(),targets.data.cpu().numpy(), epoch,"mscnn-cross-{}-{}-{}".format(save_type1,dataset,local_time),"results_image/show_image_new/")
                # plot_umap_visual_audio(features1.data.cpu().numpy(),features2.data.cpu().numpy(),targets.data.cpu().numpy(), epoch,"mscnn-cross-{}-{}-{}".format(save_type3,dataset,local_time),"results_image/show_image_new/")
        else:
            if epoch > 0 and epoch%5==0:
                print('Test Epoch: {}'.format(epoch))
                img_to_txt,txt_to_img,Acc = test_cross_modal(epoch,net,data_path)
                if Acc > best_acc:  # not the real best for sysu-mm01
                    best_epoch = epoch
                    best_acc = Acc
                    best_t_2_i =txt_to_img
                    best_i_2_t =img_to_txt
                    print("BEST ACC:{} EPOCH:{}".format(best_acc,best_epoch))
                    if best_acc >= save_threshold:
                        torch.save(net.state_dict(), 'save_model/audio_visual_{}_{}_{}_{}_{}_best.pth'.format(args.lr,args.Center_lr,args.batch_size,best_acc,dataset)) 
    return best_i_2_t,best_t_2_i,best_acc,best_epoch

def test_cross_modal(epoch,net,load_path):
    visual_test,audio_test = load_dataset_test(load_path,test_dataset_size)
    print('...Evaluation on testing data...')
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(zip(visual_test,audio_test)):
            visual_data_input = data[0][0]
            labels_visual = data[0][1]
            audio_data_input = data[1][0]
            audio_label_input = data[1][1]
            
            audio_label_input = Variable(audio_label_input)
            label_audio= audio_label_input.to(device)

            audio_data_input = Variable(audio_data_input)
            inputs_audio= audio_data_input.to(device)
    
            visual_data_input = Variable(visual_data_input)
            inputs_visual= visual_data_input.to(device)
    
            final_feature_v,final_feature_a,final_classifier_v,final_classifier_a = net(inputs_visual,inputs_audio)

            final_feature_v = final_feature_v.detach().cpu().numpy()
            final_feature_a = final_feature_a.detach().cpu().numpy()
            label = label_audio.detach().cpu().numpy()

            print('...Evaluation...')
            img_to_txt = fx_calc_map_label(final_feature_v,final_feature_a, label)
            print('...Image to Audio MAP = {:.4f}'.format(img_to_txt))
            txt_to_img = fx_calc_map_label(final_feature_a, final_feature_v, label)
            print('...Audio to Image MAP = {:.4f}'.format(txt_to_img))
            MAP = (img_to_txt + txt_to_img) / 2.
            print('...Average MAP = {:.4f}'.format(MAP))

            # save_type1 = "audio"
            # save_type2 = "visual"
            # save_type3 = "visual_audio"
            # plot_tsne_castter_visual(out_class_size,final_feature_v,label, epoch,"mscnn-cross-{}-{}-{}".format(save_type2,dataset,local_time),"results_image/show_image_new/")
            # plot_tsne_castter_audio(out_class_size,final_feature_a,label, epoch,"mscnn-cross-{}-{}-{}".format(save_type1,dataset,local_time),"results_image/show_image_new/")
            # plot_tsne_castter_v_a(out_class_size,final_feature_v,final_feature_a,label, epoch,"mscnn-cross-{}-{}-{}".format(save_type3,dataset,local_time),"results_image/show_image_new/")

    return round(img_to_txt,4),round(txt_to_img,4),round(MAP,4)

def plot_cross_modal(net,load_path):
    net.load_state_dict(torch.load('best_model/audio_visual_0.001_0.05_128_0.9115_{}_best.pth'.format(dataset)))
    # net.load_state_dict(torch.load('best_model/audio_visual_0.001_0.5_128_0.8718_{}_best.pth'.format(dataset)))
    visual_test,audio_test = load_dataset_test(load_path,test_dataset_size)
    print('...Evaluation on testing data...')
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(zip(visual_test,audio_test)):
            visual_data_input = data[0][0]
            labels_visual = data[0][1]
            audio_data_input = data[1][0]
            audio_label_input = data[1][1]
            
            audio_label_input = Variable(audio_label_input)
            label_audio= audio_label_input.to(device)

            audio_data_input = Variable(audio_data_input)
            inputs_audio= audio_data_input.to(device)
    
            visual_data_input = Variable(visual_data_input)
            inputs_visual= visual_data_input.to(device)
    
            final_feature_v,final_feature_a,final_classifier_v,final_classifier_a = net(inputs_visual,inputs_audio)

            final_feature_v = final_feature_v.detach().cpu().numpy()
            final_feature_a = final_feature_a.detach().cpu().numpy()
            view_v_predict = torch.argmax(final_classifier_v, dim=1).float()
            view_a_predict = torch.argmax(final_classifier_a, dim=1).float()
            view_v_predict = view_v_predict.detach().cpu().numpy()
            view_a_predict = view_a_predict.detach().cpu().numpy()
            label = label_audio.detach().cpu().numpy()

            classes=['0','1','2','3','4','5','6','7','8','9']
            view_a = "Audio"
            view_v = "Visual"
            plot_confusion_matrix(label,view_v_predict,classes,view_v,"True")   #y_true, y_pred, labels
            print('audio modality classification_report\n',classification_report(view_a_predict, label))
            print('visual modality classification_report\n',classification_report(view_v_predict, label))
            save_name = "Visual2Audio"
            plot_confusion_matrix(label,view_a_predict,classes,view_a,"True")

            print('...Evaluation...')
            img_to_txt = fx_calc_map_label(final_feature_v,final_feature_a, label)
            print('...Image to Audio MAP = {:.4f}'.format(img_to_txt))
            txt_to_img = fx_calc_map_label(final_feature_a, final_feature_v, label)
            print('...Audio to Image MAP = {:.4f}'.format(txt_to_img))
            MAP = (img_to_txt + txt_to_img) / 2.
            print('...Average MAP = {:.4f}'.format(MAP))

    return round(img_to_txt,4),round(txt_to_img,4),round(MAP,4)

def find_best_parameter(local_time):
    save_acc_name = "results/Train_Accuracy_{}_{}.log".format(dataset,local_time)
    lamada =[1]
    alpha = [0.001,0.01,0.1,1]
    beta =  [0.001,0.01,0.1,1]
    time_n = 0
    print("...Find best parameter...")
    test_visual_2_audio = []
    test_audio_2_visual = []
    test_map = []
    test_lamada = []
    test_alpha = []
    test_beta = []
    for g in range(len(lamada)):
        for i in range(len(alpha)):
            for j in range(len(beta)):      
                print("Train Epoch:{} Lamada:{} Alpha:{} Beta:{}".format(time_n,lamada[g],alpha[i],beta[j]))
                img_to_txt,txt_to_img,MAP = train_cross_modal(lamada[g],alpha[i],beta[j],data_path)
                save_acc(save_acc_name,img_to_txt,txt_to_img,MAP,lamada[g],alpha[i],beta[j])
                test_visual_2_audio.append(img_to_txt)
                test_audio_2_visual.append(txt_to_img)
                # test_map.append(MAP)
                # test_lamada.append(lamada[g])
                # test_alpha.append(alpha[i])
                # test_beta.append(beta[j])
                time_n +=1
if __name__ == '__main__':
    start = time.time()
    find_best_parameter(local_time)
    print('Training Time:\t {:.3f}'.format(time.time() - start))