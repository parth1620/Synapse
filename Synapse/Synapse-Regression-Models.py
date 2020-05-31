#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:49:03 2020

@author: Parth Dhameliya Project(Synapse Regression Model)
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import preprocessing 
import SessionState

st.title("Synapse Regression Models")

en = preprocessing.Encoder()

    
uploaded_file = st.file_uploader("Choose a csv file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    label_encoder = st.checkbox('Label Encoder')
    if label_encoder:
        labelledcolumns = st.multiselect(
        'Select Column for Label Encoding',
        data.columns.to_list()
    )
        if labelledcolumns:
            target = np.array(data[labelledcolumns])
            data_label = pd.DataFrame({label_encoder:en.get_label_encode(target)})
            data[labelledcolumns] = data_label
            st.write(data[labelledcolumns])

if uploaded_file is not None:
    st.sidebar.header("Options")
    droppedcolumns = st.sidebar.multiselect(
        'X : Features (selected will be dropped)',
        data.columns.to_list()
    )
    st.header('X : Features')
    X = data.drop(droppedcolumns,axis = 1)
    st.write(X)
    
    addtarget = st.sidebar.selectbox(
        'y : select target',
        data.columns.to_list()
    )
    st.header('y : Target')
    y = data[addtarget]
    st.write(y)
    one_hot_encoder = st.checkbox('One Hot Encoding')
    
    if one_hot_encoder:
        y= en.get_one_hot_encode(y)
        y = pd.DataFrame(y)
        st.write(y)
        
    Shuffle = st.sidebar.checkbox('Shuffle Data')
    if Shuffle:
        state = st.sidebar.number_input('random state',value = 1)
        X,y = preprocessing.shuffle(X,y,state)
        st.success("Successfully Shuffled")
        

class utils:
        
    def initialize_variables(self,X,y):
        X = np.array(X)
        m,n = X.shape
        X0 = np.ones((m,1))
        X = np.hstack((X0,X))
        y = np.array(y)
        y = np.reshape(y,(y.shape[0],1))
        w = np.random.randn(n+1,1)*0.01
        return X,y,w
    
    def initialization_Val(self,X,y):
        X = np.array(X)
        m,n = X.shape
        X0 = np.ones((m,1))
        X = np.hstack((X0,X))
        y = np.array(y)
        y = np.reshape(y,(y.shape[0],1))
        return X,y
    
    def initialize_variables_S(self,X,y):
        X = np.array(X)
        m,n = X.shape
        X0 = np.ones((m,1))
        X = np.hstack((X0,X))
        y = np.array(y)
        m1,n1 = y.shape
        y = np.reshape(y,(m1,n1))
        w = np.random.randn(n+1,n1)*0.01
        return X,y,w
    
    def initialization_Val_S(self,X,y):
        X = np.array(X)
        m,n = X.shape
        X0 = np.ones((m,1))
        X = np.hstack((X0,X))
        y = np.array(y)
        m1,n1 = y.shape
        y = np.reshape(y,(m1,n1))
        return X,y
    
    def grad(self,X,y,y_hat,lambd,w):
        m,n = X.shape
        dZ = y_hat-y
        dw = (1/m)*np.dot(X.T,dZ)
        dw_reg = dw + (lambd/m)*w
        dw_reg[0] = dw[0]
        return dw_reg
    
    def validation_setup(self,validation_split,X,y):
        split = int(validation_split*len(X))
        X_train = X[:-split]
        Y_train = y[:-split]
        X_Val   = X[-split:]
        Y_Val   = y[-split:]
        X_Val,Y_Val = self.initialization_Val(X_Val,Y_Val)
        X_train,Y_train,w = self.initialize_variables(X_train,Y_train)
        
        return X_train,Y_train,X_Val,Y_Val,w
    
    def validation_setup_S(self,validation_split,X,y):
        split = int(validation_split*len(X))
        X_train = X[:-split]
        Y_train = y[:-split]
        X_Val   = X[-split:]
        Y_Val   = y[-split:]
        X_Val,Y_Val = self.initialization_Val_S(X_Val,Y_Val)
        X_train,Y_train,w = self.initialize_variables_S(X_train,Y_train)
        
        return X_train,Y_train,X_Val,Y_Val,w
    
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def predictlinear(self,X,w):
        Z = np.dot(X,w)
        return Z
    
    def softmax(self,Z):
        return np.exp(Z) / np.sum(np.exp(Z),axis = 1,keepdims = True)
    
    def normalequation(self,X,y):
        w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
        return w 
        
    def rmse(self,y_hat,y):
        return np.sqrt(np.mean(np.square(y - y_hat)))
    
    def mse(self,y_hat,y):
        return np.mean(np.square(y - y_hat))
    
    def create_mini_batches(self,X, y, batch_size): 
        mini_batches = [] 
        data = np.hstack((X, y)) 
        np.random.shuffle(data) 
        n_minibatches = data.shape[0] // batch_size 
        i = 0
      
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        return mini_batches 
    
    def create_mini_batches_multi(self,X, y, batch_size): 
        m,n = X.shape
        mini_batches = [] 
        data = np.hstack((X, y)) 
        np.random.shuffle(data) 
        n_minibatches = data.shape[0] // batch_size 
        i = 0
      
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, 0:n] 
            Y_mini = mini_batch[:,n:]
            mini_batches.append((X_mini, Y_mini)) 
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, 0:n] 
            Y_mini = mini_batch[:,n:]
            mini_batches.append((X_mini, Y_mini)) 
        return mini_batches 
    
    def gradientdescentreg(self,X,y,y_hat,lambd,alpha,w,batch_size,V_dw,beta):
        if batch_size > 0 :
            mini_batches = self.create_mini_batches(X, y, batch_size)
            for mini_batch in mini_batches: 
                X_mini, y_mini = mini_batch 
                try:
                    y_hat = self.predictlinear(X_mini, w)
                    dw = self.grad(X_mini,y_mini,y_hat,lambd,w)
                    V_dw = beta*V_dw + (1-beta)*dw
                    w = w - alpha*V_dw
                except : pass
        else : 
            dw = self.grad(X,y,y_hat,lambd,w)
            V_dw = beta*V_dw + (1-beta)*dw
            w = w - alpha*V_dw
        return w,V_dw
    
    def gradientdescentclass(self,X,y,y_hat,lambd,alpha,w,batch_size,V_dw,beta):
        if batch_size > 0 :
            mini_batches = self.create_mini_batches(X, y, batch_size)
            for mini_batch in mini_batches: 
                X_mini, y_mini = mini_batch 
                try:
                    y_hat = self.sigmoid(self.predictlinear(X_mini, w))
                    dw = self.grad(X_mini,y_mini,y_hat,lambd,w)
                    V_dw = beta*V_dw + (1-beta)*dw
                    w = w - alpha*V_dw
                except : pass
        else : 
            dw = self.grad(X,y,y_hat,lambd,w)
            V_dw = beta*V_dw + (1-beta)*dw
            w = w - alpha*V_dw
        return w,V_dw
    
    def gradientdescentsoftmax(self,X,y,y_hat,lambd,alpha,w,batch_size,V_dw,beta):
        if batch_size > 0 :
            mini_batches = self.create_mini_batches_multi(X, y, batch_size)
            for mini_batch in mini_batches: 
                X_mini, y_mini = mini_batch 
                try:
                    y_hat = self.softmax(self.predictlinear(X_mini, w))
                    dw = self.grad(X_mini,y_mini,y_hat,lambd,w)
                    V_dw = beta*V_dw + (1-beta)*dw
                    w = w - alpha*V_dw
                except : pass
        else : 
            dw = self.grad(X,y,y_hat,lambd,w)
            V_dw = beta*V_dw + (1-beta)*dw
            w = w - alpha*V_dw
        return w,V_dw

    
    def showgraph_for_regression(self,epochs,j_history_train,j_history_Val,validation_split):
        
        if validation_split>0.0:
                st.subheader('Training-Validation loss per epoch')
                x = np.linspace(0,epochs,epochs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=j_history_train,
                    mode='lines',
                    name='Train Loss'))
                fig.add_trace(go.Scatter(x=x, y=j_history_Val,mode = 'lines',name = 'Val Loss'))
                st.plotly_chart(fig, use_container_width=True)
                
            
        else :
                st.subheader('Training loss per epoch')
                x = np.linspace(0,epochs,epochs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=j_history_train,
                    mode='lines',
                    name='Train Loss'))
                st.plotly_chart(fig, use_container_width=True)
                
    def showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split):
        
        if validation_split>0.0:
                st.subheader('Training-Validation loss per epoch')
                x = np.linspace(0,epochs,epochs)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x, y=j_history_train,
                    mode='lines',
                    name='Train Loss'))
                fig1.add_trace(go.Scatter(x=x, y=j_history_Val,mode = 'lines',name = 'Val Loss'))
                st.plotly_chart(fig1, use_container_width=True)
                
                st.subheader('Training-Validation Accuracy per epoch')
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x, y=accuracy_train,
                    mode='lines',
                    name='Train acc'))
                fig2.add_trace(go.Scatter(x=x, y=accuracy_Val,mode = 'lines',name = 'Val acc'))
                st.plotly_chart(fig2, use_container_width=True)
            
        else :
            
                st.subheader('Training loss per epoch')
                x = np.linspace(0,epochs,epochs)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x, y=j_history_train,mode='lines',name='Train Loss'))
                st.plotly_chart(fig1, use_container_width=True)
                st.subheader('Training Accuracy per epoch')
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x, y=accuracy_train,mode = 'lines',name = 'Train acc'))
                st.plotly_chart(fig2, use_container_width = True)
                
                
    def find_accuracy(self,h,y):
        
        for i in range(h.shape[0]):
            if h[i]>=0.5:
                h[i] = 1
            else:
                h[i] = 0
                
        k1 = np.double(h == y)
        acc_train = np.mean(k1)
        return acc_train
    
    def find_accuracy_multiclass(self,h,y):
        y = y.argmax(axis =1)
        h = h.argmax(axis =1)
        return np.double(h == y).mean()
        
class LinearRegression(utils):
    
    def computecost(self,X,y,y_hat,lambd,w):
        m,n = X.shape
        L2 = (lambd/(2*m))*(np.sum(np.square(w)))
        J = (1/(2*m))*(np.dot((y_hat-y).T,(y_hat-y))) + L2
        return J
   
    def train(self,X,y,epochs = 500 ,alpha = 0.01,lambd = 0,validation_split=0.0,method = 'GradientDescent',batch_size = 0,beta = 0,normalization = False):
        
        if validation_split>0.0:
            X_train,Y_train,X_Val,Y_Val,self.w= utils.validation_setup(self,validation_split,X,y)
            j_history_train = []
            j_history_Val = []
            if normalization:
                self.train_mean = np.mean(X_train[:,1:],axis = 0)
                self.train_std = np.std(X_train[:,1:],axis = 0)
                X_train[:,1:] = (X_train[:,1:] - self.train_mean)/self.train_std
                X_Val[:,1:] = (X_Val[:,1:] - self.train_mean)/self.train_std
                
                st.text("X training mean : {}".format(self.train_mean))
                st.text("X training standard deviation : {}".format(self.train_std))
            else : pass
                
        else :
            X_train,Y_train,self.w = utils.initialize_variables(self,X,y)
            j_history_train = []
            if normalization:
                self.train_mean = np.mean(X_train[:,1:],axis = 0)
                self.train_std = np.std(X_train[:,1:],axis = 0)
                X_train[:,1:] = (X_train[:,1:] - self.train_mean)/self.train_std
                
                st.text("X training mean : {}".format(self.train_mean))
                st.text("X training standard deviation : {}".format(self.train_std))
            else : pass
        
        V_dw = 0

        if method == 'GradientDescent':
            for i in range(0,epochs):
                
                y_hat = self.predictlinear(X_train,self.w)
                self.J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                j_history_train = np.append(j_history_train,self.J_train)
                
                if validation_split>0.0:
                        y_hat_Val= self.predictlinear(X_Val,self.w)
                        self.J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                        j_history_Val = np.append(j_history_Val,self.J_Val)
                        print("Epoch : ",i,"train_loss : ",self.J_train,"Val_loss : ",self.J_Val)
                        
                else:
                        print("Epoch : ",i,"loss : ",self.J_train)
                
                self.w,V_dw = utils.gradientdescentreg(self,X_train,Y_train,y_hat,lambd,alpha,self.w,batch_size,V_dw,beta)
                
            if validation_split>0.0:
                    utils.showgraph_for_regression(self,epochs,j_history_train,j_history_Val,validation_split)
                    self.mse_score_train = utils.mse(self,y_hat,Y_train)
                    self.rmse_score_train = utils.rmse(self,y_hat,Y_train)
                    self.mse_score_Val = utils.mse(self,y_hat_Val,Y_Val)
                    self.rmse_score_Val= utils.rmse(self,y_hat_Val,Y_Val)
                    
            else:
                    j_history_Val = 0
                    utils.showgraph_for_regression(self,epochs,j_history_train,j_history_Val,validation_split)
                    self.mse_score_train = utils.mse(self,y_hat,Y_train)
                    self.rmse_score_train = utils.rmse(self,y_hat,Y_train)
            
            
        
        elif method == 'Stochastic Gradient Descent':
            
            for i in range(0,epochs):
                y_hat = utils.predictlinear(self,X_train,self.w)
                self.J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                j_history_train = np.append(j_history_train,self.J_train)
                
                if validation_split>0.0:
                        y_hat_Val= self.predictlinear(X_Val,self.w)
                        self.J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                        j_history_Val = np.append(j_history_Val,self.J_Val)
                        print("Epoch : ",i,"train_loss : ",self.J_train,"Val_loss : ",self.J_Val)
                        
                else:
                        print("Epoch : ",i,"loss : ",self.J_train)
                    
                batch_size = 1    
                self.w,V_dw = utils.gradientdescentreg(self,X_train,Y_train,y_hat,lambd,alpha,self.w,batch_size,V_dw,beta)
                
            if validation_split>0.0:
                    utils.showgraph_for_regression(self,epochs,j_history_train,j_history_Val,validation_split)
            else:
                    j_history_Val = 0
                    utils.showgraph_for_regression(self,epochs,j_history_train,j_history_Val,validation_split)
            
            print("Training completed")
            
        
            
        elif method == 'Analytical Solution':
            
            self.w = utils.normalequation(self,X_train,Y_train)
            y_hat = utils.predictlinear(self, X_train,self.w)
            J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
            
            if validation_split > 0.0:
                y_hat_val = utils.predictlinear(self, X_Val,self.w)
                J_Val = self.computecost(X_Val,Y_Val,y_hat_val,lambd,self.w)
                
            else: pass
            st.success("Done!")
            if validation_split > 0.0:
                st.markdown("loss for train : **{}**".format(float(J_train)))
                st.markdown("loss for Validation : **{}**".format(float(J_Val)))
                st.markdown("RMSE for train : **{}**".format(utils.rmse(self,y_hat,Y_train)))
                st.markdown("RMSE for Validation: **{}**".format(utils.rmse(self,y_hat_val,Y_Val)))
                
            else :
                st.markdown("loss for train : **{}**".format(float(J_train)))
                st.markdown("RMSE for train : **{}**".format(utils.rmse(self,y_hat,Y_train)))
        
        if method == 'GradientDescent' or method == 'Stochastic Gradient Descent':
            st.success("Done!")
            st.subheader('Training-Report') 
            
            if validation_split > 0.0 :
                st.markdown('Training-Validation Loss **{}-{}**'.format(float(self.J_train),float(self.J_Val)))
                st.markdown('MSE Score of Training-Validation Set **{}-{}**'.format(self.mse_score_train,self.mse_score_Val))
                st.markdown('RMSE Score of Training-Validation Set **{}-{}**'.format(self.rmse_score_train,self.rmse_score_Val))
    
            else :   
                 st.markdown('Training Loss: **{}**'.format(float(self.J_train)))
                 st.markdown('MSE Score of Training Set **{}**'.format(self.mse_score_train))
                 st.markdown('RMSE Score of Training-Validation Set **{}**'.format(self.rmse_score_train))
                
        
                  
    
    def prediction(self,X):
        X = np.array(X)
        X0 = np.ones((X.shape[0],1))
        X = np.hstack((X0,X))
        Z = X.dot(self.w)
        Z = np.reshape(Z, (Z.shape[0],))
        return Z
    
    def rmse(self,y,y_hat):
        error = utils.rmse(self, y_hat, y)
        return error
        
    def weights(self):
        return self.w
                
            

class LogisticRegression(utils):
    
    def computecost(self,X,y,y_hat,lambd,w):
        m,n = X.shape
        L2 = (lambd/(2*m))*(np.sum(np.square(w)))
        J = -(1/m)*(np.dot(y.T,np.log(y_hat+0.0000001))+np.dot((1-y).T,np.log(1-y_hat+0.0000001))) + L2
        return J
    
    def compute_precision_recall_f1score(self,y,y_hat):
        tp = np.sum((y == 1) & (y_hat == 1))
        tn = np.sum((y == 0) & (y_hat == 0))
        fn = np.sum((y == 1) & (y_hat == 0))
        fp = np.sum((y == 0) & (y_hat == 1))
        
        precision = tp/float(tp + fp)
        recall = tp/float(tp + fn)
        f1score = (2*precision*recall)/(precision+recall)
        
        return precision,recall,f1score
    
    def train(self,X,y,epochs = 500,alpha = 0.001,lambd = 0,validation_split=0.0,method = 'GradientDescent',batch_size = 0,beta = 0,normalization = True):
        
        if validation_split>0.0:
            X_train,Y_train,X_Val,Y_Val,self.w= utils.validation_setup(self,validation_split,X,y)
            j_history_train = []
            accuracy_train = []
            j_history_Val = []
            accuracy_Val = []
            if normalization:
                self.train_mean = np.mean(X_train[:,1:],axis = 0)
                self.train_std = np.std(X_train[:,1:],axis = 0)
                X_train[:,1:] = (X_train[:,1:] - self.train_mean)/self.train_std
                X_Val[:,1:] = (X_Val[:,1:] - self.train_mean)/self.train_std
                st.text("X training mean : {}".format(self.train_mean))
                st.text("X training standard deviation : {}".format(self.train_std))
            else : pass
                
        else :
            X_train,Y_train,self.w = utils.initialize_variables(self,X,y)
            j_history_train = []
            accuracy_train = []
            if normalization:
                self.train_mean = np.mean(X_train[:,1:],axis = 0)
                self.train_std = np.std(X_train[:,1:],axis = 0)
                X_train[:,1:] = (X_train[:,1:] - self.train_mean)/self.train_std
                st.text("X training mean : {}".format(self.train_mean))
                st.text("X training standard deviation : {}".format(self.train_std))
            else : pass
            
            
        
        V_dw = 0
        if method == 'GradientDescent':
                
            for i in range(0,epochs):
                
                y_hat = utils.sigmoid(self,utils.predictlinear(self,X_train,self.w))
                self.J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                
                h = np.copy(y_hat)
                self.acc_train = utils.find_accuracy(self,h,Y_train)
                j_history_train = np.append(j_history_train,self.J_train)
                accuracy_train = np.append(accuracy_train,self.acc_train)
                
                
                if validation_split>0.0:    
                        y_hat_Val= utils.sigmoid(self,utils.predictlinear(self,X_Val,self.w))
                        self.J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                        self.acc_Val = utils.find_accuracy(self,y_hat_Val,Y_Val)
                        j_history_Val = np.append(j_history_Val,self.J_Val)
                        accuracy_Val = np.append(accuracy_Val,self.acc_Val)
                    
                        print("Epoch : ",i,"train loss : ",self.J_train,"acc_train : ",self.acc_train,"val loss : ",self.J_Val,"acc_val : ",self.acc_Val)
                else :
                        print("Epoch : ",i,"train loss : ",self.J_train,"acc_train : ",self.acc_train)
                
                self.w,V_dw = utils.gradientdescentclass(self,X_train,Y_train,y_hat,lambd,alpha,self.w,batch_size,V_dw,beta)
        
            if validation_split>0.0:
                    utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            else:
                    j_history_Val = 0
                    accuracy_Val = 0
                    utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            
            self.pr,self.r,self.f = self.compute_precision_recall_f1score(Y_train,h)
            
        elif method == 'Stochastic Gradient Descent':
            
            for i in range(0,epochs):
                
                y_hat = utils.sigmoid(self,utils.predictlinear(self,X_train,self.w))
                self.J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                
                h = np.copy(y_hat)
                self.acc_train = utils.find_accuracy(self,h,Y_train)
                j_history_train = np.append(j_history_train,self.J_train)
                accuracy_train = np.append(accuracy_train,self.acc_train)
                
                
                if validation_split>0.0:    
                        y_hat_Val= utils.sigmoid(self,utils.predictlinear(self,X_Val,self.w))
                        self.J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                        self.acc_Val = utils.find_accuracy(self,y_hat_Val,Y_Val)
                        j_history_Val = np.append(j_history_Val,self.J_Val)
                        accuracy_Val = np.append(accuracy_Val,self.acc_Val)
                    
                        print("Epoch : ",i,"train loss : ",self.J_train,"acc_train : ",self.acc_train,"val loss : ",self.J_Val,"acc_val : ",self.acc_Val)
                else :
                        print("Epoch : ",i,"train loss : ",self.J_train,"acc_train : ",self.acc_train)
                
                batch_size = 1
                self.w,V_dw = utils.gradientdescentclass(self,X_train,Y_train,y_hat,lambd,alpha,self.w,batch_size,V_dw,beta)
        
            if validation_split>0.0:
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            else:
                j_history_Val = 0
                accuracy_Val = 0
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            
            self.pr,self.r,self.f = self.compute_precision_recall_f1score(Y_train,h)
                
            
        elif method == 'Newtons Method':
        
            m,n = X_train.shape
            H = np.zeros((n,n))
            
            for i in range(epochs):
                
                y_hat = utils.sigmoid(self,utils.predictlinear(self,X_train,self.w))
                self.J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                dw = self.grad(X_train,Y_train,y_hat,lambd,self.w)
                for v in range(m):
                    x = np.reshape(X_train[v],(n,1))
                    H = H + np.multiply(np.multiply(y_hat[v],(1-y_hat[v])),np.dot(x,x.T))
                H = H/m
            
                h = np.copy(y_hat)
                self.acc_train = utils.find_accuracy(self,h,Y_train)
                j_history_train = np.append(j_history_train,self.J_train)
                accuracy_train = np.append(accuracy_train,self.acc_train)
                
                if validation_split>0.0:    
                        y_hat_Val= utils.sigmoid(self,utils.predictlinear(self,X_Val,self.w))
                        self.J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                        self.acc_Val = utils.find_accuracy(self,y_hat_Val,Y_Val)
                        j_history_Val = np.append(j_history_Val,self.J_Val)
                        accuracy_Val = np.append(accuracy_Val,self.acc_Val)
                    
                        print("Epoch : ",i,"train loss : ",self.J_train,"acc_train : ",self.acc_train,"val loss : ",self.J_Val,"acc_val : ",self.acc_Val)
                else :
                        print("Epoch : ",i,"train loss : ",self.J_train,"acc_train : ",self.acc_train)
                
                c,c_=H.shape
                self.w = self.w - np.dot(np.linalg.inv(H+np.random.randn(c,c)*0.000000001),dw) 
            if validation_split>0.0:
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            else:
                j_history_Val = 0
                accuracy_Val = 0
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            self.pr,self.r,self.f = self.compute_precision_recall_f1score(Y_train,h)
            
        if method == 'GradientDescent' or method == 'Stochastic Gradient Descent' or method == 'Newtons Method':       
            
            st.subheader("Training Report")
            if validation_split > 0.0:
                st.markdown("Precision : **{}**".format(self.pr))
                st.markdown("Recall : **{}**".format(self.r))
                st.markdown("F1 score : **{}**".format(self.f))
                st.markdown("Training Accuracy : **{}**".format(self.acc_train))
                st.markdown("Validation Accuracy : **{}**".format(self.acc_Val))
                st.markdown("Training Loss : **{}**".format(float(self.J_train)))
                st.markdown("Validation Loss: **{}**".format(float(self.J_Val)))
            else:
                st.markdown("Precision : **{}**".format(self.pr))
                st.markdown("Recall : **{}**".format(self.r))
                st.markdown("F1 score : **{}**".format(self.f))
                st.markdown("Training Accuracy : **{}**".format(self.acc_train))
                st.markdown("Training Loss : **{}**".format(float(self.J_train)))
            
                
    def prediction(self,X):
        X = np.array(X)
        X0 = np.ones((X.shape[0],1))
        X = np.hstack((X0,X))
        Z = X.dot(self.w)
        A = 1/(1+np.exp(-Z))
        for i in range(A.shape[0]):
            if A[i]>=0.5:
                A[i] = 1
            else:
                A[i] = 0
                
        return A


    def weights(self):
        return self.w
        
        
class SoftmaxRegression(utils):
    
    def computecost(self,X,y,y_hat,lambd,w):
        m,n = X.shape
        L2 = (lambd/(2*m))*(np.sum(np.square(w)))
        cost = -np.sum(y * np.log(y_hat + 0.0000001)) + L2
        J = cost/m
        return J
    
    def onehotencode(self,y):
        y = utils.encode(self, y)
        return y 
    
    def train(self,X,y,epochs = 500, alpha = 0.001,lambd = 0,validation_split=0.0,method = 'GradientDescent',batch_size = 0,beta = 0,normalization = True):
        
        if validation_split>0.0:
            X_train,Y_train,X_Val,Y_Val,self.w= utils.validation_setup_S(self,validation_split,X,y)
            j_history_train = []
            accuracy_train = []
            j_history_Val = []
            accuracy_Val = []
            if normalization:
                self.train_mean = np.mean(X_train[:,1:],axis = 0)
                self.train_std = np.std(X_train[:,1:],axis = 0)
                X_train[:,1:] = (X_train[:,1:] - self.train_mean)/self.train_std
                X_Val[:,1:] = (X_Val[:,1:] - self.train_mean)/self.train_std
                st.text("X training mean : {}".format(self.train_mean))
                st.text("X training standard deviation : {}".format(self.train_std))
            else : pass
                
        else :
            X_train,Y_train,self.w = utils.initialize_variables_S(self,X,y)
            j_history_train = []
            accuracy_train = []
            if normalization:
                self.train_mean = np.mean(X_train[:,1:],axis = 0)
                self.train_std = np.std(X_train[:,1:],axis = 0)
                X_train[:,1:] = (X_train[:,1:] - self.train_mean)/self.train_std
                st.text("X training mean : {}".format(self.train_mean))
                st.text("X training standard deviation : {}".format(self.train_std))
            else : pass
        V_dw = 0     
        if method == 'GradientDescent':
                
            for i in range(0,epochs):
                
                y_hat = utils.softmax(self,utils.predictlinear(self,X_train,self.w))
                J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                
                h = np.copy(y_hat)
                acc_train = utils.find_accuracy_multiclass(self,h,Y_train)
                j_history_train = np.append(j_history_train,J_train)
                accuracy_train = np.append(accuracy_train,acc_train)
                

            
                if validation_split>0.0:    
                    y_hat_Val= utils.softmax(self,utils.predictlinear(self,X_Val,self.w))
                    J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                    acc_Val = utils.find_accuracy_multiclass(self,y_hat_Val,Y_Val)
                    j_history_Val = np.append(j_history_Val,J_Val)
                    accuracy_Val = np.append(accuracy_Val,acc_Val)
                
                    print("Epoch : ",i,"train loss : ",J_train,"acc_train : ",acc_train,"val loss : ",J_Val,"acc_val : ",acc_Val)
                else :
                    print("Epoch : ",i,"train loss : ",J_train,"acc_train : ",acc_train)
                
                
                self.w ,V_dw = utils.gradientdescentsoftmax(self,X_train,Y_train,y_hat,lambd,alpha,self.w,batch_size,V_dw,beta)
        
            
            if validation_split>0.0:
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            else:
                j_history_Val = 0
                accuracy_Val = 0
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            
                
        elif method == 'Stochastic Gradient Descent':
            
            for i in range(0,epochs):
                
                y_hat = utils.sigmoid(self,utils.predictlinear(self,X_train,self.w))
                J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                
                h = np.copy(y_hat)
                acc_train = utils.find_accuracy_multiclass(self,h,Y_train)
                j_history_train = np.append(j_history_train,J_train)
                accuracy_train = np.append(accuracy_train,acc_train)
                
                
                if validation_split>0.0:    
                    y_hat_Val= utils.softmax(self,utils.predictlinear(self,X_Val,self.w))
                    J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                    acc_Val = utils.find_accuracy_multiclass(self,y_hat_Val,Y_Val)
                    j_history_Val = np.append(j_history_Val,J_Val)
                    accuracy_Val = np.append(accuracy_Val,acc_Val)
                
                    print("Epoch : ",i,"train loss : ",J_train,"acc_train : ",acc_train,"val loss : ",J_Val,"acc_val : ",acc_Val)
                else :
                    print("Epoch : ",i,"train loss : ",J_train,"acc_train : ",acc_train)
                
                batch_size = 1
                self.w,V_dw = utils.gradientdescentsoftmax(self,X_train,Y_train,y_hat,lambd,alpha,self.w,batch_size,V_dw,beta)
        
            if validation_split>0.0:
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            else:
                j_history_Val = 0
                accuracy_Val = 0
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
                
            
        elif method == 'Newtons Method':
        
            m,n = X_train.shape
            H = np.zeros((n,n))
            
            for i in range(epochs):
                
                y_hat = utils.sigmoid(self,utils.predictlinear(self,X_train,self.w))
                J_train = self.computecost(X_train,Y_train,y_hat,lambd,self.w)
                dw = self.grad(X_train,Y_train,y_hat,lambd,self.w)
                for v in range(m):
                    x = np.reshape(X_train[v],(n,1))
                    H = H + np.multiply(np.sum(np.multiply(y_hat[v],(1-y_hat[v]))),np.dot(x,x.T))
                H = H/m
                
                h = np.copy(y_hat)
                acc_train = utils.find_accuracy_multiclass(self,h,Y_train)
                j_history_train = np.append(j_history_train,J_train)
                accuracy_train = np.append(accuracy_train,acc_train)
                
                
                if validation_split>0.0:    
                    y_hat_Val= utils.softmax(self,utils.predictlinear(self,X_Val,self.w))
                    J_Val = self.computecost(X_Val,Y_Val,y_hat_Val,lambd,self.w)
                    acc_Val = utils.find_accuracy_multiclass(self,y_hat_Val,Y_Val)
                    j_history_Val = np.append(j_history_Val,J_Val)
                    accuracy_Val = np.append(accuracy_Val,acc_Val)
                
                    print("Epoch : ",i,"train loss : ",J_train,"acc_train : ",acc_train,"val loss : ",J_Val,"acc_val : ",acc_Val)
                else :
                    print("Epoch : ",i,"train loss : ",J_train,"acc_train : ",acc_train)
                
                self.w = self.w - np.dot(np.linalg.inv(H),dw)
                
            if validation_split>0.0:
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            else:
                j_history_Val = 0
                accuracy_Val = 0
                utils.showgraph_for_classification(self,epochs,j_history_train,j_history_Val,accuracy_Val,accuracy_train,validation_split)
            
    def prediction(self,X):
        X = np.array(X)
        X0 = np.ones((X.shape[0],1))
        X = np.hstack((X0,X))
        Z = X.dot(self.w)
        A = utils.softmax(self, Z)
        A = A.argmax(axis = 1)
        return A

    def weights(self):
        return self.w
                

class LocallyWeightedRegression(utils):
    
    def kernel(self,weight_matrix,X,xquery,tau):

        for i in range(X.shape[0]):
            diff = X[i].reshape(-1,1)-xquery
            weight_matrix[i,i]=np.exp(np.dot(diff.T,diff)/(-2*tau**2))
        return weight_matrix
    
    def find_w(self,X,weight_matrix,y):
        try :
            w = np.dot(np.linalg.inv(np.dot(np.dot(X.T,weight_matrix),X)),np.dot(np.dot(X.T,weight_matrix),y))
        except : raise ValueError('increase value of tau')
        return w
    
    def cost(self,weight_matrix,y,y_hat):
        m,n = y.shape
        J = (1/(2*m))*np.dot(np.dot(weight_matrix,(y_hat-y)).T,(y_hat-y))
        return J
   
    def test(self,xquery,X,y,tau=0.5):
        
        X = np.array(X)
        y = np.array(y).reshape(-1,1)
        xquery = np.array(xquery).reshape(-1,1)
        xquery0 = np.ones((xquery.shape[1],1))
        xquery = np.vstack((xquery0,xquery))
        X0 = np.ones((X.shape[0],1))
        X = np.hstack((X0,X))
        weight_matrix = np.array(np.eye(X.shape[0]))
        
        weight_matrix = self.kernel(weight_matrix,X,xquery,tau)
        w = self.find_w(X,weight_matrix,y)
        y_hat = utils.predictlinear(self, X, w)
        yquery = utils.predictlinear(self,xquery.T,w)
        loss = self.cost(weight_matrix,y,y_hat)
        
        error = utils.rmse(self, y_hat, y)
        
        st.subheader('Report')
        st.markdown("overall cost using value of tau **{}** is : **{}**".format(tau,loss))
        st.markdown("rmse :**{}**".format(error))
        return yquery
    
    def rmse(self,y,y_hat):
        error = utils.rmse(self, y_hat, y)
        return error
 
if uploaded_file is not None:               
        Algorithms = st.sidebar.selectbox(
            'Algorithm',
            ('None','Linear Regression','Logistic Regression','Softmax Regression','Locally Weighted Regression')
            )
        
if uploaded_file is not None:
    if Algorithms == 'Linear Regression':
       methods = st.sidebar.radio('Select any method below',("GradientDescent","Stochastic Gradient Descent","Analytical Solution"))
       
       if methods == 'GradientDescent':
           norm = st.sidebar.checkbox('normalization')
           alpha_value = st.sidebar.number_input('alpha',value = 0.03,format = '%.5f')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           batch_size = int(st.sidebar.number_input('Batch Size',value = 0))
           momentum = st.sidebar.number_input('Momentum',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
           
       elif methods == 'Stochastic Gradient Descent':
           norm = st.sidebar.checkbox('normalization')
           alpha_value = st.sidebar.number_input('alpha',value = 0.03,format = '%.5f')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           momentum = st.sidebar.number_input('Momentum',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
           batch_size = 1
           
       elif methods == 'Analytical Solution':
           norm = st.sidebar.checkbox('normalization')
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
        
        
       
       if st.sidebar.checkbox('Click Here to start training'):
           
           with st.spinner('Your is data getting train...'):
               if methods == 'GradientDescent' or methods =='Stochastic Gradient Descent':
                   d = LinearRegression()
                   d.train(X,y,epochs = epochs_value,alpha = alpha_value,validation_split = val_split,method = methods,beta = momentum,lambd = reg_lambd,normalization = norm)
                   w = d.weights()
                   st.subheader("Weights for prediction")
                   st.text(w)
                   st.latex(r'''x_{0} \in 1''')
                   st.latex(r'''\hat{y} = x_{0}*w_{0} + x_{1}*w_{1} + x_{2}*w_{2} + . . . + x_{n}*w_{n}''')
                   
               elif methods == 'Analytical Solution':
                   d = LinearRegression()
                   d.train(X,y,validation_split = val_split,method = methods,normalization = norm)
                   w = d.weights()
                   st.markdown('Weights for prediction : **{}**'.format(d.weights()))
                   st.latex(r'''x_{0} \in 1''')
                   st.latex(r'''\hat{y} = x_{0}*w_{0} + x_{1}*w_{1} + x_{2}*w_{2} + . . . + x_{n}*w_{n}''')
                   
          
    elif Algorithms == 'Logistic Regression':
        methods = st.sidebar.radio('Select any method below',("GradientDescent","Stochastic Gradient Descent","Newtons Method"))
        
        if methods == 'GradientDescent':
           norm = st.sidebar.checkbox('normalization')
           alpha_value = st.sidebar.number_input('alpha',value = 0.03,format = '%.5f')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           batch_size = int(st.sidebar.number_input('Batch Size',value = 0))
           momentum = st.sidebar.number_input('Momentum',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
           
        elif methods == 'Stochastic Gradient Descent':
           norm = st.sidebar.checkbox('normalization')
           alpha_value = st.sidebar.number_input('alpha',value = 0.03,format = '%.5f')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           momentum = st.sidebar.number_input('Momentum',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
           batch_size = 1
           
        elif methods == 'Newtons Method':
           norm = st.sidebar.checkbox('normalization')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)

        if st.sidebar.checkbox('Click Here to start training'):
           with st.spinner('Your data is getting train...'):
               
               if methods == 'GradientDescent' or methods =='Stochastic Gradient Descent':
                   d = LogisticRegression()
                   d.train(X,y,epochs = epochs_value,alpha = alpha_value,validation_split = val_split,method = methods,beta = momentum,lambd = reg_lambd,batch_size = batch_size,normalization = norm)
                   w = d.weights()
                   st.subheader("Weights for prediction")
                   st.text(w)
                   st.latex(r'''x_{0} \in 1''')
                   st.latex(r'''z = x_{0}*w_{0} + x_{1}*w_{1} + x_{2}*w_{2} + . . . + x_{n}*w_{n}''')
                   st.latex(r'''\hat{y} = \frac{1}{1+e^{-z}}''')
                   
               elif methods == 'Newtons Method':
                   d = LogisticRegression()
                   d.train(X,y,validation_split = val_split,epochs = epochs_value,method = methods,lambd = reg_lambd,normalization = norm)
                   w = d.weights()
                   st.subheader("Weights for prediction")
                   st.text(w)
                   st.latex(r'''x_{0} \in 1''')
                   st.latex(r'''z = x_{0}*w_{0} + x_{1}*w_{1} + x_{2}*w_{2} + . . . + x_{n}*w_{n}''')
                   st.latex(r'''\hat{y} = \frac{1}{1+e^{-z}}''')
               
                    
                   
                   
    elif Algorithms == 'Softmax Regression':
        methods = st.sidebar.radio('Select any method below',("GradientDescent","Stochastic Gradient Descent","Newtons Method"))
        
        if methods == 'GradientDescent':
           norm = st.sidebar.checkbox('normalization') 
           alpha_value = st.sidebar.number_input('alpha',value = 0.03,format = '%.5f')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           batch_size = int(st.sidebar.number_input('Batch Size',value = 0))
           momentum = st.sidebar.number_input('Momentum',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
           
        elif methods == 'Stochastic Gradient Descent':
           norm = st.sidebar.checkbox('normalization') 
           alpha_value = st.sidebar.number_input('alpha',value = 0.03,format = '%.5f')
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           momentum = st.sidebar.number_input('Momentum',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
           batch_size = 1
           
        elif methods == 'Newtons Method':
           norm = st.sidebar.checkbox('normalization') 
           epochs_value = int(st.sidebar.number_input('epochs',value = 100))
           val_split = st.sidebar.number_input('Validation Split',value = 0.0)
           reg_lambd = st.sidebar.number_input('Lambda(L2)',value = 0.0)
        
        if st.sidebar.checkbox('Click Here to start training'):
           
           with st.spinner('Your data is getting train...'):
               if methods == 'GradientDescent' or methods =='Stochastic Gradient Descent':
                   d = SoftmaxRegression()
                   d.train(X,y,epochs = epochs_value,alpha = alpha_value,validation_split = val_split,method = methods,beta = momentum,lambd = reg_lambd,normalization = norm)
                   st.subheader("Weights for prediction")
                   w = d.weights()
                   st.text(w)
                   st.latex(r'''x_{0} \in 1''')
                   st.latex(r'''z = x_{0}*w_{0} + x_{1}*w_{1} + x_{2}*w_{2} + . . . + x_{n}*w_{n}''')
                   st.latex(r'''\hat{y} = \frac {e^{Z}}{sum(e^{Z})}''')
                   
               elif methods == 'Newtons Method':
                   d = SoftmaxRegression()
                   d.train(X,y,validation_split = val_split,epochs = epochs_value,method = methods,lambd = reg_lambd,normalization = norm)
                   st.subheader("Weights for prediction")
                   w = d.weights()
                   st.text(w)
                   st.latex(r'''x_{0} \in 1''')
                   st.latex(r'''z = x_{0}*w_{0} + x_{1}*w_{1} + x_{2}*w_{2} + . . . + x_{n}*w_{n}''')
                   st.latex(r'''\hat{y} = \frac {e^{Z}}{sum(e^{Z})}''')
                   
    elif Algorithms == 'Locally Weighted Regression':
        
        st.sidebar.subheader("X-query")
        st.sidebar.info("Do not Shuffle Data when using Locally Weighted Regression")
        fnames = X.columns.to_list()
        feature = {}
        for i in fnames:
            feature[i] = [st.sidebar.number_input(i,format = '%.5f')]
            
        xquery = pd.DataFrame(feature)
        st.subheader("X Query")
        st.write(xquery)
        T = st.sidebar.number_input('Tau',value = 0.3,format = '%.5f')
        
        if st.sidebar.checkbox('Click Here for prediction'):
            
            with st.spinner('Predicting.....'):
                d = LocallyWeightedRegression()
                yquery = d.test(xquery, X, y,tau = T)
                st.subheader('Prediction')
                st.text(yquery)
        
        
            

                
   
                   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
           
