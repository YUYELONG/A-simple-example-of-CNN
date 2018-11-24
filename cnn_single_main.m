%%%  matlab实现LeNet-5
%%%  作者：xd.wp
%%%  时间：2016.10.22  14:29
%% 程序说明
%          1、池化（pooling）采用平均2*2
%          2、网络结点数说明：
%                           输入层：28*28
%                           第一层：24*24（卷积）*20
%                           tanh
%                           第二层：12*12（pooling）*20
%                           第三层：100(全连接)
%                           第四层：10(softmax)
%          3、网络训练部分采用800个样本，检验部分采用100个样本
clear all;clc;

%% 网络初始化
layer_c1_num=20;
layer_s1_num=20;
layer_f1_num=100;
layer_output_num=10;
%权值调整步进
yita=0.01;
%bias初始化
%bias_c1 belongs to [-1,1]/sqrt(20)
bias_c1=(2*rand(1,20)-ones(1,20))/sqrt(20);
%bias_f1 belongs to [-1,1]/sqrt(20)
bias_f1=(2*rand(1,100)-ones(1,100))/sqrt(20);

%卷积核初始化
[kernel_c1,kernel_f1]=init_kernel(layer_c1_num,layer_f1_num);
%pooling核初始化
pooling_a=ones(2,2)/4;
%全连接层的权值
weight_f1=(2*rand(20,100)-ones(20,100))/sqrt(20);
weight_output=(2*rand(100,10)-ones(100,10))/sqrt(100);
disp('网络初始化完成......');

%% prepare the train data
pre_train = pre_train_data();


%% 开始网络训练
disp('开始网络训练......');
for iter=1:30
    
for n=1:5000
    5000*30-(iter-1)*5000-n
    for m=0:9
        %读取样本
        full_data = pre_train{m+1};
        train_data= full_data(:,:,n);
        % 去均值
%       train_data=wipe_off_average(train_data);
        %前向传递,进入卷积层1
        for k=1:layer_c1_num
            state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            %进入激励函数
            state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
            %进入pooling1
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
        end
        %进入f1层
        [state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
        %进入激励函数
        for nn=1:layer_f1_num
            state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        %进入softmax层
        for nn=1:layer_output_num
            output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
        end
       %% 误差计算部分
        Error_cost=-output(1,m+1);
%         if (Error_cost<-0.98)
%             break;
%         end
        %% 参数调整部分
        [kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1]=CNN_upweight(yita,Error_cost,m,train_data,...
                                                                                                state_c1,state_s1,...
                                                                                                state_f1,state_f1_temp,...
                                                                                                output,...
                                                                                                kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1);

    end    
end
end
disp('网络训练完成，开始检验......');

%% 开始网络检验
% prepare the train data
pre_test = pre_test_data();

count=0;
for n=1:20
    for m=0:9
        %读取样本
        
        full_test_data = pre_test{m+1};
        train_data= full_test_data(:,:,n);
        
        % 去均值
%       train_data=wipe_off_average(train_data);
        %前向传递,进入卷积层1
        for k=1:layer_c1_num
            state_c1(:,:,k)=convolution(train_data,kernel_c1(:,:,k));
            %进入激励函数
            state_c1(:,:,k)=tanh(state_c1(:,:,k)+bias_c1(1,k));
            %进入pooling1
            state_s1(:,:,k)=pooling(state_c1(:,:,k),pooling_a);
        end
        %进入f1层
        [state_f1_pre,state_f1_temp]=convolution_f1(state_s1,kernel_f1,weight_f1);
        %进入激励函数
        for nn=1:layer_f1_num
            state_f1(1,nn)=tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        %进入softmax层
        for nn=1:layer_output_num
            output(1,nn)=exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
        end
        [p,classify]=max(output);
        if (classify==m+1)
            count=count+1;
        end
        fprintf('真实数字为%d  网络标记为%d  概率值为%d \n',m,classify-1,p);
    end
end
