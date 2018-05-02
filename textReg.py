# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:16:17 2018

@author: whao
"""
import cv2 as cv
import os
import pickle
import numpy as np
import pandas as pd
import keras.backend as K
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
            self.model.save_weights(self.filepath+str(lastName)+'.h5')
            self.lastfile=self.filepath+str(lastName)+'.h5'
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
        self.text=[]
        self.pic=[]
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
        with open('rawY1.pkl','wb') as f:
            pickle.dump(self.text,f)
        with open('rawX1.pkl','wb') as f:
            pickle.dump(self.pic,f)
            
        #原始数据
        self.text=[]
        self.pic=[]
        self.num=0
        self.img=None
        
        
    def cut(self,x):
        if x[8]=='###' or x[8]=='':
            return
        pts = np.array([[x[0],x[1]],[x[6]+2,x[7]],[x[4]+2,x[5]+2],[x[2],x[3]+2]], np.int32)
        l0=int(np.sqrt((pts[1][0]-pts[0][0])**2+(pts[1][1]-pts[0][1])**2))
        l1=int(np.sqrt((pts[2][0]-pts[1][0])**2+(pts[2][1]-pts[1][1])**2))
        l2=int(np.sqrt((pts[3][0]-pts[2][0])**2+(pts[3][1]-pts[2][1])**2))
        l3=int(np.sqrt((pts[0][0]-pts[3][0])**2+(pts[0][1]-pts[3][1])**2))
        pts1=np.float32(pts)
        pts2=np.float32([[0,0],[max(l0,l2),0],[max(l0,l2),max(l1,l3)],[0,max(l1,l3)]])    
        M = cv.getPerspectiveTransform(pts1,pts2)
        perspective = cv.warpPerspective(self.img,M,(max(l0,l2),max(l1,l3)))
        #resize the image
        #perspective=cv.resize(perspective,(self.width,self.height))
        
        if ' ' in set(str(x[8])):
            print(str(x[8]))
            
        if self.num%3000 == 0:
            cv.imwrite('data\\'+str(self.num)+'.jpg',perspective)
            print(perspective.shape)
            print(x[8])
        
        self.text.append(str(x[8]))
        self.pic.append(perspective)
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
                    tx1.apply(self.cut,axis=1)
        #print(self.dic)
    def one_hot(self,maxLabelLength,text):
        #print('test:',text)
        label = np.zeros(maxLabelLength,dtype=int)
        #label[0]=charactersNum
        for i, char in enumerate(text):
            index = self.dic[char]
            label[i] = int(index)
            #label[i*2+2]=charactersNum
            #print(label.shape)
        return label
    def loadData(self):
        with open('rawY.pkl','rb') as f:
            self.text=pickle.load(f)
        with open('rawX.pkl','rb') as f:
            self.pic=pickle.load(f)
        self.num=len(self.pic)
        
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

    def train(self):
        
        input_length=np.array(list(map(lambda x:int(x.shape[1]/4)+1,self.pic[:int(self.num*self.rateTrain)])))
        label_length=np.array(list(map(lambda x:len(x),self.text[:int(self.num*self.rateTrain)])))
        maxLabelLength=max(label_length)
        Y=np.array(list(map(lambda x:self.one_hot(maxLabelLength,x),self.text[:int(self.num*self.rateTrain)])),
                   dtype=object)
        X=np.array(self.pic[:int(self.num*self.rateTrain)],dtype=object)

        checkpoint=modelHistory(self.weightFile)
        callbacks_list = [checkpoint]
        self.model.fit([X,Y,input_length,label_length],np.ones(X.shape[0]),batch_size=self.batch_size,epochs=self.epochs,
                  callbacks=callbacks_list)

    def predict(self):
        for _ in range(10):
            l=5
            Y=np.array(list(map(lambda x:self.one_hot(50,x),self.text[:l])),dtype=object)
            X=np.array(self.pic[:l],dtype=object)
            input_length=np.array(list(map(lambda x:int(x.shape[1]/4)+1,self.pic[:l])))
            label_length=np.array(list(map(lambda x:len(x),self.text[:l])))
            print(X.shape,Y.shape,input_length.shape)
            pre=self.model.predict(X)
            print([max(pre[u])] for u in range(pre.shape[0]))
            print(pre)
            pre=K.ctc_decode(pre,input_length,greedy=False,beam_width=6,top_paths=3)
            pre=pre[0][0].eval(session=tf.Session())
            print(type(pre))
            print(pre.shape)
            ch=list(self.characters)
        
            for i in range(l):
                
                res1=''
                res2=''
                print(pre[i])
                print(Y[i])
                for j in range(pre.shape[1]):
                    if pre[i][j] == -1:
                        break
                    res1=res1+ch[pre[i][j]]
                print(label_length[i])
                for j in range(label_length[i]):
                    #print(Y[j])
                    res2=res2+ch[Y[i][j]]
                print(res1,res2)
                cv.imshow(str(i),self.pic[i])
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
    
        m = Permute((2,1,3),name='permute')(m)
        m = TimeDistributed(Flatten(),name='timedistrib')(m)
    
        m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
        m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
        m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)
        y_pred = Dense(self.nclass+1,name='blstm2_out_1',activation='softmax')(m)
        if not flag:
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
    

if __name__=='__main__':
    reg=textReg()
    type=1
    #用于生成字典,
    if type==0:
        reg.getDic(R'data\txt_1000',True)
    #用于切割文本行或列
    if type==1:
        reg.cutPictures(R'data\image_1000',R'data\txt_1000')
        reg.save()
    #用于训练
    if type==2:
        reg.loadModel(True)
        reg.loadData()
        reg.train()
    if type==3:
        reg.loadModel(False)
        reg.loadData()
        reg.predict()
