import numpy as np
import pandas as pd
import scipy.linalg as sl
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import FastICA,KernelPCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import minepy
from sklearn.decomposition import PCA

#时间窗
def Timewindow(data,time_steps): #时间窗函数
    tw=[]
    rows=len(data)
    for i in range(0,time_steps):
        df=data.copy()
        df.index=df.index+i
        tw.append(df)
    data_tw=pd.concat(tw,axis=1)
    data_tw=data_tw.iloc[:rows]
    data_tw.fillna(0,inplace=True)
    return data_tw.values

def correlation(data): #画相关系数热力图 
    l=len(data.columns)
    cor=np.zeros((l,l))
    for i in range(0,l):
        for j in range(0,l):
            mine=minepy.MINE(alpha=0.6,c=15)
            mine.compute_score(data.iloc[:,i],data.iloc[:,j])
            cor[i,j]=mine.mic()
            print(i)
    df=pd.DataFrame(data=cor,index=data.columns[0:],columns=data.columns[0:])
    plt.subplots(figsize=(6.67,5),dpi=300,facecolor="w")
    fig=sns.heatmap(df,annot=True,vmax=1,square=True,cmap="Blues",fmt='.2g')
    plt.show()

class SSA():
    def __init__(self,LengthOfEpoch,window_size = 5, threshold = 1, NumberOfSS = -1):
        self.LengthOfEpoch = LengthOfEpoch
        self.window_size = window_size
        self.threshold = threshold
        self.NumberOfSS = NumberOfSS
        
    #突变点检测函数
    def sliding_window_detection(self,data): 
        window_means = []
        for i in range(len(data) - self.window_size + 1):
            window = data[i:i+self.window_size]
            window_mean = np.mean(window)
            window_means.append(window_mean)
        change_points = []
        for i in range(1, len(window_means)):
            diff = abs(window_means[i] - window_means[i-1])
            if diff > self.threshold:
                change_points.append(i)
        return change_points

    #平稳子空间分析主算法
    def SSAMain(self,data):
        mean_all=[] #记录所有片段的均值
        cov_all=[] #记录所有片段的样本方差
        mean_avarage=[0 for _ in range(np.shape(data)[1])]
        cov_avarage=np.zeros((np.shape(data)[1],np.shape(data)[1]))
        N=0 #记录总的片段数
        for i in range(0,len(data),self.LengthOfEpoch):
            if i+self.LengthOfEpoch<=len(data):
                mean_epoch=np.mean(data.iloc[i:i+self.LengthOfEpoch])
                cov_epoch=np.cov(data.iloc[i:i+self.LengthOfEpoch],ddof=1,rowvar=False)
                mean_all.append(mean_epoch)
                cov_all.append(cov_epoch)
                mean_avarage=mean_avarage+mean_epoch
                cov_avarage=cov_avarage+cov_epoch
                N=N+1
        mean_avarage=mean_avarage/N
        cov_avarage=cov_avarage/N
        cov_avarage_inv=np.linalg.inv(cov_avarage)
        S=0
        for i in range(0,N):
            S=S+np.dot(mean_all[i],mean_all[i].T)+2*np.dot(np.dot(cov_all[i],cov_avarage_inv),cov_all[i])
        S=S/N-np.dot(mean_avarage,mean_avarage.T)-2*cov_avarage
        eigenvalues, eigenvectors = sl.eig(S,cov_avarage)
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        if self.NumberOfSS == -1:
            ChangePoints = self.sliding_window_detection(eigenvalues_sorted)
        else:
            ChangePoints = [self.NumberOfSS]
        eigenvectors_sorted=np.array(eigenvectors_sorted)
        Bs=eigenvectors_sorted[:,0:ChangePoints[0]]
        Bn=eigenvectors_sorted[:,ChangePoints[0]:]
        print("changepoint[0]=",ChangePoints[0])
        return Bs.real,Bn.real

class SVMFaultDetect():
    def __init__(self, svmkernel='linear', svmC=1.0):
        self.svmkernel = svmkernel
        self.svmC = svmC
    
    def SVMPred(self, X_train_ori, X_test_ori, y_train_ori):
        sc=StandardScaler()
        X_train_std = sc.fit_transform(X_train_ori)
        X_test_std = sc.transform(X_test_ori)
        svm = SVC(kernel=self.svmkernel, C=self.svmC, random_state=1)
        svm.fit(X_train_std, y_train_ori)
        return svm.predict(X_test_std)    

class SSASVM():
    def __init__(self, LengthOfEpoch=10, window_size = -1, threshold = -1, NumberOfCP=12, svmkernel='linear', svmC=1.0):
        self.LengthOfEpoch = LengthOfEpoch
        self.window_size = window_size
        self.threshold = threshold
        self.NumberOfCP = NumberOfCP
        self.svmkernel = svmkernel
        self.svmC = svmC
        
    def SSASVMPred(self, data_normal_train, data_normal_test, data_fault_train, data_fault_test, variable, y_train_ori):
        ssa = SSA(LengthOfEpoch=self.LengthOfEpoch, window_size=self.window_size, threshold=self.threshold, NumberOfSS=self.NumberOfCP)
        Bs_normal,Bn_normal = ssa.SSAMain(data_normal_train[variable])
        Ss_normal_train = np.dot(data_normal_train[variable], Bs_normal) #训练数据平稳信号 d*时间序列长度
        Ss_normal_test = np.dot(data_normal_test[variable], Bs_normal) #测试数据平稳信号
        Ss_Fault_train = np.dot(data_fault_train[variable], Bs_normal)
        Ss_Fault_test = np.dot(data_fault_test[variable], Bs_normal)
        X_train_ssa = np.concatenate((Ss_normal_train, Ss_Fault_train), axis=0)
        X_test_ssa = np.concatenate((Ss_normal_test, Ss_Fault_test), axis=0)

        svm = SVC(kernel=self.svmkernel, C=self.svmC, random_state=1)
        svm.fit(X_train_ssa, y_train_ori)
        return svm.predict(X_test_ssa)

class LDASVM():
    def __init__(self,LDAComponents=2,svmkernel='linear',svmC=1.0):
        self.LDAComponents=LDAComponents
        self.svmkernel=svmkernel
        self.svmC=svmC
        
    def LDASVMPred(self,X_train_ori,X_test_ori,y_train_ori):
        sc=StandardScaler()
        X_train_std=sc.fit_transform(X_train_ori)
        X_test_std=sc.transform(X_test_ori)
        lda = LinearDiscriminantAnalysis(n_components=self.LDAComponents)
        X_train_lda=lda.fit_transform(X_train_std,y_train_ori)
        X_test_lda=lda.transform(X_test_std)
        svm=SVC(kernel=self.svmkernel, C=self.svmC, random_state=1)
        svm.fit(X_train_lda, y_train_ori)
        y_pred=svm.predict(X_test_lda)
        return y_pred

class DDFSVM():
    def __init__(self, StepOfWindow=3, LDAComponents=2, svmkernel='linear', svmC=1.0):
        self.StepOfWindow = StepOfWindow
        self.LDAComponents = LDAComponents
        self.svmkernel=svmkernel
        self.svmC=svmC
    
    def DDFSVMPred(self, X_train_ori,X_test_ori,y_train_ori, data_normal_train, data_normal_test, data_fault_train, data_fault_test, variable):
        #特征提取
        print("StepOfWindow=", self.StepOfWindow)
        X_train_window = Timewindow(pd.DataFrame(X_train_ori), self.StepOfWindow)
        X_test_window = Timewindow(pd.DataFrame(X_test_ori), self.StepOfWindow)
        lda = LinearDiscriminantAnalysis(n_components=self.LDAComponents)
        lda.fit(X_train_window, y_train_ori)
        X_train_lda = lda.transform(X_train_window)
        X_test_lda = lda.transform(X_test_window)

        #特征拼接
        index_ntrain = [i for i in range(len(data_normal_train))]
        index_ntest = [i for i in range(len(data_normal_test))]
        index_ftrain = [i for i in range(len(data_normal_train), len(data_normal_train) + len(data_fault_train))]
        index_ftest = [i for i in range(len(data_normal_test), len(data_normal_test) + len(data_fault_test))]
        Xtr_normal_concat = np.concatenate((data_normal_train[variable], X_train_lda[index_ntrain]), axis=1)
        Xte_normal_concat = np.concatenate((data_normal_test[variable], X_test_lda[index_ntest]), axis=1)
        Xtr_fault_concat = np.concatenate((data_fault_train[variable], X_train_lda[index_ftrain]), axis=1)
        Xte_fault_concat = np.concatenate((data_fault_test[variable], X_test_lda[index_ftest]), axis=1)

        X_train_ddf = np.concatenate((Xtr_normal_concat, Xtr_fault_concat))
        X_test_ddf = np.concatenate((Xte_normal_concat, Xte_fault_concat))
        #SVM分类算法
        svm = SVC(kernel='linear', C=1.0, random_state=1)
        svm.fit(X_train_ddf, y_train_ori)
        y_pred = svm.predict(X_test_ddf)
        return y_pred

class PCASVM():
    def __init__(self,PCAComponents=2,svmkernel='linear',svmC=1.0):
        self.PCAComponents=PCAComponents
        self.svmkernel=svmkernel
        self.svmC=svmC
        
    def PCASVMPred(self,X_train_ori,X_test_ori,y_train_ori):
        sc=StandardScaler()
        X_train_std=sc.fit_transform(X_train_ori)
        X_test_std=sc.transform(X_test_ori)
        pca = PCA(n_components=self.PCAComponents)
        X_train_lda=pca.fit_transform(X_train_std)
        X_test_lda=pca.transform(X_test_std)
        svm=SVC(kernel=self.svmkernel, C=self.svmC, random_state=1)
        svm.fit(X_train_lda, y_train_ori)
        y_pred=svm.predict(X_test_lda)
        return y_pred

class MLPFaultDetect():
    def __init__(self,hidden_layer_sizes=(15,),max_iter=1000):
        self.hidden_layer_sizes=hidden_layer_sizes
        self.max_iter=max_iter
        
    def MLPPred(self,X_train_ori,X_test_ori,y_train_ori):
        #多层感知机预测算法
        scaler=StandardScaler()
        X_train_std=scaler.fit_transform(X_train_ori)
        X_test_std=scaler.transform(X_test_ori)
        mlp=MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,random_state=1)
        mlp.fit(X_train_std, y_train_ori)
        y_pred=mlp.predict(X_test_std)
        return y_pred
        
class ICASVM():
    def __init__(self,ICAComponents=2,svmkernel='linear',svmC=1.0):
        self.ICAComponents=ICAComponents
        self.svmkernel=svmkernel
        self.svmC=svmC
    
    def ICASVMPred(self,X_train_ori,X_test_ori,y_train_ori):
        sc=StandardScaler()
        X_train_std=sc.fit_transform(X_train_ori)
        X_test_std=sc.transform(X_test_ori)
        ica = FastICA(n_components=self.ICAComponents)
        X_train_ica=ica.fit_transform(X_train_std)
        X_test_ica=ica.transform(X_test_std)
        svm=SVC(kernel=self.svmkernel, C=self.svmC, random_state=1)
        svm.fit(X_train_ica, y_train_ori)
        y_pred=svm.predict(X_test_ica)
        return y_pred

class KPCASVM():
    def __init__(self,KPCAComponents=2,svmkernel='linear',svmC=1.0):
        self.KPCAComponents=KPCAComponents
        self.svmkernel=svmkernel
        self.svmC=svmC
    
    def KPCASVMPred(self,X_train_ori,X_test_ori,y_train_ori):
        sc=StandardScaler()
        X_train_std=sc.fit_transform(X_train_ori)
        X_test_std=sc.transform(X_test_ori)
        kpca = KernelPCA(n_components=self.KPCAComponents)
        X_train_kpca=kpca.fit_transform(X_train_std)
        X_test_kpca=kpca.transform(X_test_std)
        svm=SVC(kernel=self.svmkernel, C=self.svmC, random_state=1)
        svm.fit(X_train_kpca, y_train_ori)
        y_pred=svm.predict(X_test_kpca)
        return y_pred

class PCAFaultDetect():
    def __init__(self,alpha=0.01,PCAComponents=1):
        self.alpha = alpha
        self.PCAComponents=PCAComponents
        print("PCAComponent=",self.PCAComponents) 

    def centralize_data(self,data):
        mean=np.mean(data,axis=0)
        centralized_data=data-mean
        return np.array(centralized_data)

    def whiten_data(self,data):
        centralized_data=self.centralize_data(data)
        cov_matrix=np.cov(centralized_data, rowvar=False)
        eigenvalues, eigenvectors=np.linalg.eig(cov_matrix)
        if all(eigenvalues): #没有0特征值
            whitening_matrix=np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues)))
        else:
            whitening_matrix=np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues+1e-5)))
        whitened_data = np.dot(centralized_data, whitening_matrix)
        return whitened_data
    
    #PCA故障诊断算法
    def PCAFaultDetectPred(self,data_normal_train,data_fault_train,X_test_ori,variable):
        Xtr_normal_cw=self.centralize_data(data_normal_train[variable])
        Xtr_fault_cw=self.centralize_data(data_fault_train[variable])
        Xte_cw=self.centralize_data(X_test_ori)
        CovMatrix=(Xtr_normal_cw.T@Xtr_normal_cw)/(len(Xtr_normal_cw)-1)
        eigenvalues, eigenvectors = sl.eig(CovMatrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        P=eigenvectors_sorted[:,:self.PCAComponents].real
        S=np.diag(eigenvalues_sorted[0:self.PCAComponents]).real
        threshold_t2 = self.PCAComponents*((len(Xtr_normal_cw)*len(Xtr_normal_cw))-1)*stats.f.ppf(1-self.alpha,self.PCAComponents,len(Xtr_normal_cw)-self.PCAComponents)/len(Xtr_normal_cw)/(len(Xtr_normal_cw)-self.PCAComponents)  # 设置显著性水平为0.05，计算阈值
        threshold_spe=5*np.sqrt(eigenvalues_sorted[self.PCAComponents])
        t2_Xte=[]
        spe_Xte=[]
        I=np.eye(len(variable))
        index_t2_fault=[]
        index_spe_fault=[]
        for i in range(len(Xte_cw)):
            t2_Xte.append(Xte_cw[i,:]@P@np.linalg.inv(S)@P.T@Xte_cw[i,:].T)
            spe_Xte.append(Xte_cw[i,:]@(I-P@P.T).T@(I-P@P.T)@Xte_cw[i,:].T)
        for i in range(len(Xte_cw)):
            if (t2_Xte[i]>threshold_t2):
                index_t2_fault.append(i)
            if (spe_Xte[i]>threshold_spe):
                index_spe_fault.append(i)
        svm = SVC(kernel='linear', C=1.0, random_state=1)
        svm.fit(Xtr_fault_cw, data_fault_train["Label"])
        y_pred_fault = svm.predict(Xte_cw[index_t2_fault]) #对错误样本的分类
        y_pred=np.zeros(len(X_test_ori))
        y_pred[index_t2_fault]=y_pred_fault #对所有样本的分类
        return y_pred

class KDA():
    def __init__(self, X, y, gamma, n_components):
        self.X = X
        self.y = y
        self.gamma = gamma
        self.n_components = n_components
        self.nSample, self.nDim = self.X.shape
        self.total_index = np.arange(self.nSample)
        self.labels = np.unique(self.y)
        self.nClass = len(self.labels)
        self.total_mean = np.mean(self.X, axis=0)
        self.class_size = []
        self.class_mean = self.get_Mean()
        #self.KM = pairwise_kernels(X=X,Y=X, metric='rbf')
        self.KM = rbf_kernel(X=X, gamma=self.gamma)
        self.N = self.get_N()
        self.H = self.get_H()
        epsilon = 1e-08
        eig_val, eig_vec = sl.eigh(sl.inv(self.N + epsilon * np.eye(self.nSample))@self.H)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        if self.n_components is not None:
            Theta = eig_vec[:,:self.n_components]
        else:
            Theta = eig_vec[:,:self.nClass-1]
        self.Theta = Theta
        print(self.Theta.shape)

    def transform(self, X):
        # Kernel_Tran_input = pairwise_kernels(X=self.X, Y=X, metric='rbf')
        Kernel_Tran_input = rbf_kernel(X=self.X, Y=X, gamma=self.gamma)
        X_transform = self.Theta.T @ Kernel_Tran_input
        return X_transform.T

    def get_Mean(self):
        Mean = np.zeros((self.nClass, self.nDim))
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y == lab)[0]
            Mean[i] = np.mean(self.X[idx_list], axis=0)
            self.class_size.append(len(idx_list))
        return Mean

    def get_H(self):
        H = np.zeros((self.nSample, self.nSample))
        M_star = self.KM.sum(axis=1)
        M_star = M_star.reshape((-1,1))
        M_star = (1 / self.nSample) * M_star
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y==lab)[0]
            Ki = self.KM[np.ix_(self.total_index, idx_list)]
            M_c = Ki.sum(axis=1)
            M_c = M_c.reshape((-1, 1))
            M_c = (1 / self.class_size[i]) * M_c
            H += self.class_size[i] * (M_c - M_star) @ (M_c - M_star).T
        return H

    def get_N(self):
        N = np.zeros((self.nSample, self.nSample))
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y == lab)[0]
            # Kq = rbf_kernel(X=self.X, Y=self.X[idx_list],gamma=self.gamma)
            Ki = self.KM[np.ix_(self.total_index, idx_list)]
            N += Ki @ (np.eye(self.class_size[i]) - np.ones(self.class_size[i])*(1/self.class_size[i])) @ Ki.T
        N += np.eye(self.nSample) * 1e-8
        return N