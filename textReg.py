# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:16:17 2018

@author: whao
"""
import cv2 as cv
import random
import os
import pickle
import sys
import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D, Dense
from keras.layers import Flatten,BatchNormalization,Permute,TimeDistributed,Dense,Bidirectional,GRU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback


from keras import backend as K

from keras.layers import Lambda
from keras.optimizers import SGD

#用于每轮结束后及训练完成后保存权值，每轮结束后，轮流用modelWeights0.h5和modelWeights1.h5保存，结束后覆盖modelWeights.h5,只保存loss变小的轮次
class modelHistory(Callback):
    def __init__(self,filepath=''):
        self.filepath=filepath
        self.lastfile=None
        self.lastloss=-1
        self.lastName=0
        
    def on_epoch_end(self,epoch=0,logs=None):
        if epoch==0 or logs['loss'] <= self.lastloss:
            self.model.save_weights(self.filepath+str(self.lastName)+'.h5')
            self.lastfile=self.filepath+str(self.lastName)+'.h5'
            self.lastloss=logs['loss']
            self.lastName=(self.lastName+1)%2
        else:
            print('not in')
        '''
        print(self.params)
        print(logs)
        print(epoch,epoch%2)
        print('save model '+self.lastfile)
        '''
    def on_train_end(self,logs=None):
        fr=open(self.lastfile,'rb')
        fw=open(self.filepath+'.h5','wb')
        fw.write(fr.read())
        fr.close()
        fw.close()
        print('end')

class textReg:
    def __init__(self):
        
        #原始数据
        self.data=[]#每个元数据是(pic,text)
        self.num=0
        self.img=None
        

        #字典
        self.characters=set()
        self.dic={}
        self.nclass=0
        self.ch=[]

        #每张训练图片的大小
        self.height=128
        self.width=500

        #模型的训练及预测信息
        self.rateTrain=0.7
        self.batch_size=16
        self.epochs=1000
        self.weightFile='modelWeights'
        self.saveNow='0'
        self.model=None
    #生成字典,flag=True表示该目录是最后一个用于生成字典的目录
    def getDic(self,textDir,flag):
        for parent,dirname,filenames in os.walk(textDir):
            for filename in filenames:
                if filename[-3:]=='txt':
                    tx1=pd.read_csv(textDir+R'\\'+filename,header=None,quoting=3,sep='.[0-9]{1,2},',
                                        encoding='utf-8',engine='python')
                    tx1.apply(lambda x:self.characters.update(str(x[8])) if x[8] != '###' and x[8] !='' else 0,axis=1)
        print(self.characters,len(self.characters))
        if flag:
            self.dic=dict(zip(self.characters,range(len(self.characters))))
            with open('dic1.pkl','wb') as f:
                pickle.dump(self.dic,f)
            self.dic={}

    def save(self):
        random.shuffle(self.data)
        with open('data.pkl','wb') as f:
            pickle.dump(self.data,f)

        
        #原始数据
        self.text=[]
        self.pic=[]
        self.filfname=None
        self.num=0
        self.img=None
        
        
    def cut(self,x):
        if x[8]=='###' or x[8]=='':
            return
        pts = np.array([[x[0]-1,x[1]-1],[x[6]+1,x[7]-1],[x[4]+1,x[5]+1],[x[2]-1,x[3]+1]], np.int32)
        l0=int(np.sqrt((pts[1][0]-pts[0][0])**2+(pts[1][1]-pts[0][1])**2))
        l1=int(np.sqrt((pts[2][0]-pts[1][0])**2+(pts[2][1]-pts[1][1])**2))
        l2=int(np.sqrt((pts[3][0]-pts[2][0])**2+(pts[3][1]-pts[2][1])**2))
        l3=int(np.sqrt((pts[0][0]-pts[3][0])**2+(pts[0][1]-pts[3][1])**2))
        lw=max(l0,l2)
        lh=max(l1,l3)
        if lw*lh<=50 or lw <=6 or lh<=6:
            return
        pts1=np.float32(pts)
        pts2=np.float32([[0,0],[lw,0],[lw,lh],[0,lh]])    
        M = cv.getPerspectiveTransform(pts1,pts2)
        perspective = cv.warpPerspective(self.img,M,(lw,lh))
        if ' ' in set(str(x[8])):
            print(str(x[8]))
        if self.num%3000 == 0:
            cv.imwrite('data\\'+str(self.num)+'.jpg',perspective)
            print(perspective.shape)
            print(x[8])
        if x[8]=='小童话?大智慧?小百科?大启发':
            print(l0,l1,l2,l3)
            print(pts)
            cv.imshow('cut',perspective)
            cv.waitKey(0)
        self.data.append((perspective,str(x[8])))
        self.num+=1
    def cutPictures(self,imageDir,textDir):
        for parent,dirname,filenames in os.walk(imageDir):
            for filename in filenames:
                if filename[-3:]=='jpg':
                    tx1=pd.DataFrame()
                    self.img=cv.imread(imageDir+'\\'+filename)
                    if self.img is None:
                        print('error------------',filename)
                        continue
                    if os.path.isfile(textDir+'\\'+filename[:-8]+'.txt'):
                        tx1=pd.read_csv(textDir+'\\'+filename[:-8]+'.txt',header=None,quoting=3,sep='.[0-9]{1,2},',
                                        encoding='utf-8',engine='python')
                    elif os.path.isfile(textDir+'\\'+filename[:-4]+'.txt'):
                        tx1=pd.read_csv(textDir+'\\'+filename[:-4]+'.txt',header=None,quoting=3,sep='.[0-9]{1,2},',
                                        encoding='utf-8',engine='python')
                    else:
                        print('error',filename)
                        continue
                    self.filfname=filename
                    tx1.apply(self.cut,axis=1)
        #print(self.dic)
    def one_hot(self,maxLabelLength,text):
        #print('test:',text)
        label = np.zeros(maxLabelLength,dtype=int)
        #label[0]=charactersNum
        for i, char in enumerate(text):
            if char not in self.dic.keys():
                label[i]=0
                continue
            index = self.dic[char]
            label[i] = int(index)
            #label[i*2+2]=charactersNum
            #print(label.shape)
        return label
    def loadData(self):
        with open('data.pkl','rb') as f:
            self.data=pickle.load(f)
        self.data=self.data[:1000]
        #random.shuffle(self.data)
        self.num=len(self.data)
        """
        for i in range(self.num):
            print(self.data[i][1])
            cv.imshow(str(i),self.data[i][0])
            cv.waitKey(0)
        """
        
    def loadModel(self,flag):
        #加载字典,字典大小与模型相关
        with open('dic.pkl','rb') as f:
            self.dic=pickle.load(f)
        with open('dic1.pkl','rb') as f:
            dic1=pickle.load(f)
        print(set(self.dic.keys())-set(dic1.keys()),set(dic1.keys())-set(self.dic.keys()))
        self.nclass=len(self.dic)
        self.characters=set(self.dic)
        self.ch=[0]*self.nclass
        print('nclass: ',self.nclass)
        for u in self.dic.items():
            self.ch[u[1]]=u[0]

        print('check the dic')
        for i in range(self.nclass):
            if self.dic[self.ch[i]]!=i:
                print('dic error -------------')
        print('end check dic')

        #加载模型,flag=True表示用于训练,反之用于预测
        self.model=self.getmodel(self.height,flag)
        if os.path.exists(self.weightFile+'.h5'):
            self.model.load_weights(self.weightFile+'.h5',by_name=True)
        else :
            print('not exist modelWeights.h5')
    def gen(self,data,flag):
        
        while True:
            i = 0
            n = len(data)
            X=[]
            Y=[]
            cnt=0
            for i in range(n):
                '''
                数据增强，加水平翻转，垂直翻转，对角线旋转
                '''
                img=data[i][0]
                Y.append(data[i][1])
                if flag=='train':
                    flip_h=cv.flip(img,1)
                    flip_v=cv.flip(img,0)
                    flip_hv=cv.flip(img,-1)
                    X.append(cv.resize(img,(self.width,self.height)))
                    X.append(cv.resize(flip_h,(self.width,self.height)))
                    X.append(cv.resize(flip_v,(self.width,self.height)))
                    X.append(cv.resize(flip_hv,(self.width,self.height)))
                    Y.append(data[i][1])
                    Y.append(data[i][1])
                    Y.append(data[i][1])
                    cnt+=4
                else:
                    cnt+=1
                if cnt==self.batch_size or i==n-1:
                    input_length=np.array(list(map(lambda x:int(x.shape[1]/4)+1,X)))
                    label_length=np.array(list(map(lambda x:len(x),Y)))
                    maxLabelLength=max(label_length)
                    nowY=np.array(list(map(lambda x:self.one_hot(maxLabelLength,x),Y)),
                               dtype=object)
                    nowX=np.array(X)
                    yield ([nowX,nowY,input_length,label_length],np.ones(cnt))
                    X=[]
                    Y=[]
                    cnt=0
                    
    def train(self):

        ##input_length=np.array(list(map(lambda x:int(x.shape[1]/4)+1,self.pic[])))
        checkpoint=modelHistory(self.weightFile)
        callbacks_list = [checkpoint]
        trainNum=int(self.num*self.rateTrain)
        valiNum=self.num-trainNum
        print('steps_per_epoch=',(trainNum*4-1)//self.batch_size+1)
        self.model.fit_generator(self.gen(self.data[:trainNum]),steps_per_epoch=(trainNum*4-1)//self.batch_size+1,epochs=self.epochs,
                            callbacks=callbacks_list,validation_data=self.gen(self.data[-valiNum:]),
                            validation_steps=(valiNum*4-1)//self.batch_size+1)

    def predict(self):
        input_length=np.array(list(map(lambda x:int(x.shape[1]/4)+1,self.pic[:int(self.num*self.rateTrain)])))
        label_length=np.array(list(map(lambda x:len(x),self.text[:int(self.num*self.rateTrain)])))
        maxLabelLength=max(label_length)
        Y=np.array(list(map(lambda x:self.one_hot(maxLabelLength,x),self.text[:int(self.num*self.rateTrain)])))
        X=np.array(self.pic[:int(self.num*self.rateTrain)])
        for _ in range(10):
            l=50
            input_length=input_length[:l]
            label_length=label_length[:l]
            Y=Y[:l]
            X=X[:l]
            print(X.shape,Y.shape,input_length.shape)
            pre=self.model.predict(X)
            print([max(pre[u])] for u in range(pre.shape[0]))
            print(pre)
            pre=K.ctc_decode(pre,input_length,greedy=False,beam_width=6,top_paths=3)
            pre=pre[0][0].eval(session=tf.Session())
            #print(type(pre))
            print(pre.shape)
        
            for i in range(l):
                
                res1=''
                res2=''
                print(pre[i])
                print(Y[i])
                for j in range(pre.shape[1]):
                    if pre[i][j] == -1:
                        break
                    res1=res1+self.ch[pre[i][j]]
                print(label_length[i])
                for j in range(label_length[i]):
                    #print(Y[j])
                    res2=res2+self.ch[Y[i][j]]
                print(res1,res2)
                cv.imshow(str(i),X[i])
                cv.waitKey(0)

        '''
        for i in range(10):
            print(pre[0][i].eval(session=tf.Session()),Y[i])
        '''
    def ctc_lambda_func(self,args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    def getmodel(self,height,flag):#if flag=False,return basemodel,if flag =True return model
        rnnunit  = 256
        input = Input(shape=(height,None,3),name='the_input')
        m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(input)
        m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)#/2
        m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)
        m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)#/2
        m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)
        m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)
    
        m = ZeroPadding2D(padding=(0,1))(m)#width+2
        m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)#height/2,width-1
    
        m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)
        m = BatchNormalization(axis=1)(m)
        m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)
        m = BatchNormalization(axis=1)(m)
        m = ZeroPadding2D(padding=(0,1))(m)#width+2
        m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)#height/2,width-1
        m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)#height-1,width-1
        
        m1 = TimeDistributed(Flatten(),name='timedistrib')(m)
        m1 = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstmv1')(m1)
        m1 = Dense(rnnunit,name='blstm1_out',activation='linear')(m1)
        m1 = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstmv2')(m1)
        
        m = Permute((2,1,3),name='permute')(m)
        
        m = TimeDistributed(Flatten(),name='timedistrib')(m)
    
        m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
        m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
        m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)
        y_pred = Dense(self.nclass+1,name='blstm2_out_1',activation='softmax')(m)
        if not flag:
            print('not in ')
            basemodel = Model(inputs=input,outputs=y_pred)
            basemodel.summary()
            return basemodel
    
        labels = Input(name='the_labels', shape=[None,], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        model.summary()
        
        return model
    def test(self):
        reg.loadData()
        
        reg.loadModel(True)
        for i in range(3):
            img=self.data[i][0]
            '''
            flipud=tf.image.flip_up_down(img)
            fliplr=tf.image.flip_left_right(img)
            flipt=tf.image.transpose_image(img)
            '''
            flip_h=cv.flip(img,1)
            flip_v=cv.flip(img,0)
            flip_hv=cv.flip(img,-1)
            cv.imshow('raw image',img)
            cv.imshow('flipud',flip_h)
            cv.imshow('fliplr',flip_v)
            cv.imshow('flipt',flip_hv)
            cv.waitKey(0)
            cv.destroyAllWindows()
    

if __name__=='__main__':
    args=sys.argv
    print(args)
    reg=textReg()
    type=args[1]
    print(type)
    #用于生成字典,
    if type=='getDic':
        reg.getDic(R'data\txt',True)
    #用于切割文本行或列
    if type=='cutPictures':
        reg.cutPictures(R'data\image',R'data\txt')
        reg.save()
    #用于训练
    if type=='train':
        reg.loadModel(True)
        reg.loadData()
        reg.train()
    if type=='predic':
        reg.loadModel(False)
        reg.loadData()
        #reg.predict()
    if type=='test':
        reg.test()
