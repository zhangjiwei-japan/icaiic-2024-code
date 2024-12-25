import torch
import torch.nn as nn
from torch.nn import init
from models.msnet import MSNet
# from av_attention import Cross_Attention_Layer,Attention_Layer,Co_Attention_Layer,Multi_Stage_Cross_Attention_Layer
from models.multi_scale_net import MultiScale_Modal_Net
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

classifier_criterion = nn.CrossEntropyLoss().cuda()

class Multi_modal_Fusion_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self,input_dim,dim_k):
        super(Multi_modal_Fusion_Attention,self).__init__()
        self.q_v = nn.Linear(input_dim,dim_k)
        self.q_a = nn.Linear(input_dim,dim_k)
        self.k_v = nn.Linear(input_dim,dim_k)
        self.k_a = nn.Linear(input_dim,dim_k)

        self.w_1 = nn.Linear(input_dim,dim_k)
        self.w_2 = nn.Linear(input_dim,dim_k)
        self.w_3 = nn.Linear(input_dim,dim_k)
        self.w_4 = nn.Linear(input_dim,dim_k)

        self.joint_attention_map = nn.Bilinear(256, 256, 256, bias = False)
        self._norm_fact = 1 / sqrt(dim_k) 
        self.t_c = nn.Parameter(torch.ones([])) 
    
    def forward(self, img, audio):
        W_q_v = self.q_v(img)  # Q: batch_size * seq_len * dim_k
        W_q_a = self.q_a(audio)  # Q: batch_size * seq_len * dim_k
        w_k_a = self.k_a(audio)
        W_k_v = self.k_v(img) # Q: batch_size * seq_len * dim_k
         
        atten_va = nn.Softmax(dim=-1)(torch.bmm(W_q_v, w_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_vv = nn.Softmax(dim=-1)(torch.bmm(W_q_v,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        atten_av = nn.Softmax(dim=-1)(torch.bmm(W_q_a,W_k_v.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_aa = nn.Softmax(dim=-1)(torch.bmm(W_q_a,w_k_a.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len

        alpha_va = torch.bmm(atten_va,audio)
        alpha_v = torch.bmm(atten_vv,img) 

        alpha_av = torch.bmm(atten_av,img)
        alpha_a = torch.bmm(atten_aa, audio) 

        c_k_va = F.sigmoid(torch.add(self.w_1(alpha_va),self.w_2(alpha_v))*alpha_v)
        c_k_av = F.sigmoid(torch.add(self.w_3(alpha_av),self.w_4(alpha_a))*alpha_a)
        atten_c = self.joint_attention_map(c_k_va,c_k_av)
        j_c = F.sigmoid(atten_c)
        Z_k = (self.t_c*j_c)*img + (1-j_c)*audio

        return Z_k 

class Classifier(nn.Module):
    def __init__(self,latent_dim=64,out_label=10,kaiming_init=False):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(latent_dim, 64),
            # nn.ReLU(),
            nn.Linear(latent_dim, 32, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(32,out_label, bias=False))
        if kaiming_init:
            self._init_weights_classifier()

    def _init_weights_classifier(self):
        for  m in self._modules:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.classifier(x)
        return x #nn.CrossEntropyLoss()
        # return F.softmax(x, dim=1)
        # return F.log_softmax(x, dim = 1) #F.nll_loss

class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=2048):
        super(ImgNN, self).__init__()
        self.visual_encoder = nn.Linear(input_dim, output_dim)
        # self.visual_encoder = nn.Sequential(
        #     nn.Linear(input_dim, output_dim),   # 1024 512
        #     # nn.LayerNorm(mid_dim),
        #     # nn.BatchNorm1d(mid_dim, affine=False,track_running_stats=False), 
        #     # nn.BatchNorm1d(output_dim, affine=False),
        #     # nn.Dropout(0.5),
        #     nn.Linear(output_dim, output_dim)# 512 128
        # )

    def forward(self, x):
        out = F.relu(self.visual_encoder(x))

        return out

class AudioNN(nn.Module):
    """Network to learn audio representations"""
    def __init__(self, input_dim=1024, output_dim=2048):
        super(AudioNN, self).__init__()
        self.audio_encoder =  nn.Linear(input_dim, output_dim) 
        # self.audio_encoder = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),   # 128 128
        #     # nn.LayerNorm(output_dim),
        #     # nn.BatchNorm1d(output_dim, affine=False,track_running_stats=False), 
        #     # nn.BatchNorm1d(input_dim, affine=False), 
        #     # nn.Dropout(0.5),
        #     nn.Linear(input_dim, output_dim)    # 128 128
        # )

    def forward(self, x):
        out = F.relu(self.audio_encoder(x))
        return out

class cross_modal_net(nn.Module):
    def __init__(self, input_dim=1024, mid_dim=512, out_dim=128, class_num = 10, kaiming_init=True):
        super(cross_modal_net, self).__init__()
        self.visual_fine_grained_feature =  MultiScale_Modal_Net(num_features=1)  #  torch.Size([32, 3, 256])
        self.audio_fine_grained_feature =  MultiScale_Modal_Net(num_features=1) #  torch.Size([32, 3, 256])
        self.kaiming_init = kaiming_init # 模型初始化
        self.visual_layer = ImgNN(input_dim=input_dim, mid_dim=mid_dim, output_dim=out_dim,kaiming_init=self.kaiming_init)
        self.audio_layer = AudioNN(input_dim=out_dim, output_dim=out_dim,kaiming_init=self.kaiming_init)

        # self.visual_layer_1 = nn.Linear(input_dim, mid_dim, bias=False)
        # self.visual_layer_2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.MmFA = Multi_modal_Fusion_Attention(128,128)
        self.shared_layer_1 = nn.Linear(out_dim, out_dim, bias=False)
        self.shared_layer_2 = nn.Linear(out_dim*3,out_dim, bias=False)
        self.shared_layer_3 = nn.Linear(out_dim,64, bias=False)
        # self.classifier = nn.Linear(64, class_num, bias=False)
        self.classifier_a = Classifier(latent_dim=64,out_label=class_num,kaiming_init=self.kaiming_init)
        self.classifier_v = Classifier(latent_dim=64,out_label=class_num,kaiming_init=self.kaiming_init)
        # self.classifier.apply(weights_init_classifier)

    def forward(self, visual,audio):
        batch_size = img.size(0)
        dimanal = img.size(1)
        # visual-audio encoder 
        visual_feature = self.visual_layer(img)
        audio_feature = self.audio_layer(audio)

        feature_visual = self.visual_layer(visual) 
        # feature_visual = self.visual_layer_2(feature_visual)
        feature_visual = self.shared_layer_1(feature_visual)
        feature_visual = feature_visual.view(feature_visual.size(0),1,feature_visual.size(1)) # torch.Size([32,1, 128])
        # print(feature_visual.shape)

        feature_audio = self.audio_layer(audio) 
        feature_audio = self.shared_layer_1(feature_audio)
        feature_audio = feature_audio.view(feature_audio.size(0),1,feature_audio.size(1)) # torch.Size([32,1, 128])
        # print(feature_audio.shape)

        audio_out_x1,audio_out_x2,audio_out_x3= self.msnet_a(feature_audio) # torch.Size([32, 128])
        visual_out_x1,visual_out_x2,visual_out_x3= self.msnet_v(feature_visual) # torch.Size([32, 128])
        
        # out_feature_a = torch.cat((feature_map_a_x1,feature_map_a_x2,feature_map_a_x3),1)
        # out_feature_v = torch.cat((feature_map_v_x1,feature_map_v_x2,feature_map_v_x3),1)
        out_feature_a_0 = torch.cat((audio_out_x1,audio_out_x2,audio_out_x3),1)
        out_feature_v_0 = torch.cat((visual_out_x1,visual_out_x2,visual_out_x3),1)
        out_feature_a_1 = self.shared_layer_2(out_feature_a_0) # 128
        out_feature_v_1 = self.shared_layer_2(out_feature_v_0) # 128
        M_1_k = self.MmFA(out_feature_v_1,out_feature_a_1)
        I_k_v = M_1_k*out_feature_v_1
        I_k_a = M_1_k*out_feature_a_1
        M_2_k = self.MmFA(I_k_v,I_k_a)
    
        final_feature_a = self.shared_layer_3(M_2_k*out_feature_a_1) # 64
        final_feature_v = self.shared_layer_3(M_2_k*out_feature_v_1)
    
        final_classifier_a = self.classifier_a(final_feature_a)
        final_classifier_v = self.classifier_v(final_feature_v)
   
        # out_classifier_a =torch.cat((a_classifier_x1,a_classifier_x2,a_classifier_x3,a_classifier_x4),1)
        # out_classifier_v =torch.cat((v_classifier_x1,v_classifier_x2,v_classifier_x3,v_classifier_x4),1)
        # return final_feature_v,final_feature_a,torch.log_softmax(final_classifier_v,dim=1),torch.log_softmax(final_classifier_a,dim=1)
        return final_feature_v,final_feature_a,final_classifier_v,final_classifier_a

class CrossModal_NN(nn.Module):
    """Network to learn audio representations"""
    
    def __init__(self, img_input_dim=1024, img_output_dim=1024,
                 audio_input_dim=128, audio_output_dim=1024, minus_one_dim=128, output_dim=10):
        super(CrossModal_NN, self).__init__()
        self.visual_layer = ImgNN(input_dim= img_input_dim, output_dim= img_output_dim)
        self.audio_layer = AudioNN(input_dim= audio_input_dim, output_dim= audio_output_dim)
        # self.visual_global_feature = nn.AdaptiveMaxPool1d(output_size = img_output_dim, return_indices=False)
        # self.audio_global_feature = nn.AdaptiveMaxPool1d(output_size = audio_output_dim, return_indices=False)
        self.MmFA = Multi_modal_Fusion_Attention(256,256)

        self.visual_fine_grained_feature =  MultiScale_Modal_Net(num_features=1)  #  torch.Size([32, 3, 256])
        self.audio_fine_grained_feature =  MultiScale_Modal_Net(num_features=1) #  torch.Size([32, 3, 256])

        self.out_layer = nn.Sequential(
            nn.Linear(in_features=768, out_features = 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features = minus_one_dim)
            )
        self.classifier_audio = Classifier(latent_dim= minus_one_dim,out_label= output_dim)
        self.classifier_visual = Classifier(latent_dim= minus_one_dim,out_label= output_dim)

    def forward(self, img, audio):
        batch_size = img.size(0)
        dimanal = img.size(1)
        # visual-audio encoder 
        visual_feature = self.visual_layer(img)
        audio_feature = self.audio_layer(audio)
        # FGR
        visual_multi_scale_feature = self.visual_fine_grained_feature(visual_feature)
        audio_multi_scale_feature = self.audio_fine_grained_feature(audio_feature)
        # two-stage fusion, multi-modal fusion attention
        M_1_k = self.MmFA(visual_multi_scale_feature,audio_multi_scale_feature)
        I_k_v = M_1_k*visual_multi_scale_feature
        I_k_a = M_1_k*audio_multi_scale_feature
        M_2_k = self.MmFA(I_k_v,I_k_a)
        out_visual_feat = M_2_k*visual_multi_scale_feature
        out_audio_feat = M_2_k*audio_multi_scale_feature
        
        # output feature
        out_visual_feature = self.out_layer(out_visual_feat.view(batch_size, -1))
        out_audio_feature = self.out_layer(out_audio_feat.view(batch_size, -1))
        # out_visual_feature = F.normalize(out_visual_feature, dim=-1)
        # out_audio_feature = F.normalize(out_audio_feature, dim=-1)

        visual_predict = self.classifier_visual(out_visual_feature)
        audio_predict = self.classifier_audio(out_audio_feature)

        return out_visual_feature, out_audio_feature, visual_predict, audio_predict


# if __name__ == '__main__':
#     x_A = torch.rand(32, 1024) 
#     x_B = torch.rand(32, 128) 
#     net = CrossModal_NN()
#     # net = cross_modal_base(input_dim=1024, mid_dim=512, out_dim=128, class_num = 10)
#     out_A,out_B,label_A,label_B = net(x_A,x_B)
#     print(out_A.shape)
#     print(out_B.shape)
#     print(label_A.shape)
#     print(label_B.shape)









