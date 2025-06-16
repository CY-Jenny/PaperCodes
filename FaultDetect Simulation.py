import PySimpleGUI as sg
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import DetectMethod as dm

#设置全局变量
def set_global_var(): 
    global data_is_scrolling, ScrollDataIndex, NewScrollData, data_is_trained, heading, variable, StepOfWindow, LengthOfEpoch, NumOfSS
    data_is_scrolling = [False]
    ScrollDataIndex = [0]
    NewScrollData = []
    data_is_trained = [False]
    heading = ["Time","OER","Perm","CO","H2","CO2","SWS","OEF","CAF","WP","BGC","BGI","TCT","TP1","TP2","TP3","TP4","OEP","CWP1","CWP2","TPD","HWP1","HWP2","AWS","CWT","HWT","TTEN","TTWS","TTWN","TTES","TTDT","DC","BT","Set_CI","Actual_CI","Last_CI"]
    variable=["OER","Perm","CO","H2","CO2","SWS","OEF","CAF","WP","BGI","TCT","TP1","OEP","CWP1","TPD","HWP1","AWS","TTEN","TTWS","TTWN","TTES","TTDT","DC","Last_CI"] #选取不相关的变量
    StepOfWindow = [3] #默认的滑动时间窗步长
    LengthOfEpoch = [10] #默认的ssa分段长度
    NumOfSS = [12] #默认的平稳分量个数

def data_test_online_prepare():
    global data_test_online, TimeForScroll
    data_test_online=pd.read_csv("data_test.csv",
                    skiprows=1,encoding='gb18030',header=None,
                    names=["Time","OER","Perm","CO","H2","CO2","SWS","OEF","CAF","WP","BGC","BGI","TCT","TP1","TP2","TP3",
                    "TP4","OEP","CWP1","CWP2","TPD","HWP1","HWP2","AWS","CWT","HWT","TTEN","TTWS","TTWN","TTES","TTDT","DC",
                    "BT","Set_CI","Actual_CI","Last_CI","Label"])
    data_test_online['Time'] =pd.to_datetime(data_test_online['Time'],errors='coerce') #时间列转换与处理
    data_test_online["Dif_s"]=(data_test_online["Time"]-data_test_online["Time"][0])/pd.Timedelta("1s") #计算时间差，以分钟为单位
    data_test_online["Dif_s"]=data_test_online["Dif_s"].diff()
    data_test_online["Dif_s"][0]=0
    TimeForScroll=data_test_online["Dif_s"]

#设置模型参数
def param_setting(window, event, value):
    if (event == '-InputStepOfWindow-') | (event == '-InputStepOfWindow2-'):
        try:
            StepOfWindow[0] = int(value['-InputStepOfWindow-']) if event == '-InputStepOfWindow-' else int(value['-InputStepOfWindow2-'])
        except ValueError:
            if value['-InputStepOfWindow-'] and value['-InputStepOfWindow2-']:
                sg.popup("Window Size Should Be Integer!")
    if (event =='-InputLenOfEpoch-') | (event =='-InputLenOfEpoch2-'):
        try:
            LengthOfEpoch[0] = int(value['-InputLenOfEpoch-']) if event == '-InputLenOfEpoch-' else int(value['-InputLenOfEpoch2-'])
        except ValueError:
            if value['-InputLenOfEpoch-'] and value['-InputLenOfEpoch2-']:
                sg.popup("Length Of SSA Epoch Should Be Integer!")
    if (event == '-InputNumOfSS-') | (event == '-InputNumOfSS2-'):
        try:
            NumOfSS[0] = int(value['-InputNumOfSS-']) if event == '-InputNumOfSS-' else int(value['-InputNumOfSS2-'])
        except ValueError:
            if value['-InputNumOfSS-'] and  value['-InputNumOfSS2-']:
                sg.popup("Number Of Stationary Sources Should Be Integer!")    
    if (event == '-DefaultValue-') | (event == '-DefaultValue2-'):
        StepOfWindow[0] = 3 #默认的滑动时间窗步长
        LengthOfEpoch[0] = 10 #默认的ssa分段长度
        NumOfSS[0] = 6 #默认的平稳分量个数
        if event == '-DefaultValue-':
            window['-InputStepOfWindow-'].update('3')
            window['-InputLenOfEpoch-'].update('10')
            window['-InputNumOfSS-'].update('6')
        elif event == '-DefaultValue2-':
            window['-InputStepOfWindow2-'].update('3')
            window['-InputLenOfEpoch2-'].update('10')
            window['-InputNumOfSS2-'].update('6')
    if event == '-AutoParam-':
        try:
            ParamSet=[]
            score=[]
            l = len(data_train)
            df_train = pd.concat([data_train[:int(15*l/50)], data_train[int(2*l/5):int(3*l/5)], data_train[int(35*l/50):int(45*l/50)]])
            df_test = pd.concat([data_train[int(15*l/50):int(2*l/5)],data_train[int(3*l/5):int(35*l/50)],data_train[int(45*l/50):]])
            num0 = len(df_test[df_test["Label"]==0])
            num1 = len(df_test[df_test["Label"]==1])
            num2 = len(df_test[df_test["Label"]==2])
            for i in range(2,6):
                for j  in range(8,12):
                    for k in range(5,8):
                        print('111')
                        model_train_once(df_train, i, j, k)
                        print('222')
                        y_param = model_pred(df_test[variable], i)    
                        print('333')
                        df_test["pred"] = y_param
                        df_error = df_test[df_test["pred"] != df_test["Label"]]  
                        norm_to_fault = len(df_error[df_error["Label"] == 0])
                        fault1_to_norm = len(df_error[(df_error["Label"] == 1) & (df_error["pred"] == 0)])
                        fault2_to_norm = len(df_error[(df_error["Label"] == 2) & (df_error["pred"] == 0)]) 
                        FAR = norm_to_fault/num0*100
                        MAR1 = fault1_to_norm/num1*100
                        MAR2 = fault2_to_norm/num2*100 
                        ParamSet.append([i,j,k])  
                        score.append(FAR*0.2+MAR1*0.4+MAR2*0.4)   
                        print(score)   
        except:
            1
    #参数更改后必须重新训练模型
    print(StepOfWindow[0], LengthOfEpoch[0], NumOfSS[0])
    data_is_trained[0] = False

#开始数据的滚动
def data_scroll_control(window, event, value):
    if data_is_trained[0]:    
        if event == '-StartDetect-':
            data_is_scrolling[0] = True
            window.perform_long_operation(lambda :data_scroll_funcion(window), None)
        elif event == '-StopDetect-':
            data_is_scrolling[0] = False
    else:
        sg.popup('Please Train Fault Detect Model First.')

#数据的滚动函数    
def data_scroll_funcion(window):
    DataTable=window['-DataTable-']
    last_y_pred = -1
    while data_is_scrolling[0]:
        NewScrollData.append(data_test_online[heading].iloc[ScrollDataIndex[0]].tolist())
        DataTable.update(values=NewScrollData)
        DataTable.set_vscroll_position(1.0)
        ScrollDataIndex[0]=ScrollDataIndex[0]+1
        y_pred = int(model_pred(data_test_online[variable].iloc[max(ScrollDataIndex[0]-StepOfWindow[0], 0):ScrollDataIndex[0]], StepOfWindow[0]))
        if y_pred != last_y_pred:    
            state_update(window, y_pred)
        last_y_pred = y_pred
        time.sleep(TimeForScroll[ScrollDataIndex[0]]-1)    

def state_update(window, y_pred):
    if y_pred == 0:
        window.Element('-StateText-').update('Normal')
    elif y_pred == 1:
        window.Element('-StateText-').update('Fault1', text_color = 'red')
    elif y_pred == 2:
        window.Element('-StateText-').update('Fault2', text_color = 'red')        

def model_pred(data, win):
    X_test_window = dm.Timewindow(data, win)
    X_test_lda = lda.transform(X_test_window)
    X_test_concate = np.concatenate([data.iloc[-1], X_test_lda[0]])
    X_test_std = sc.transform(X_test_concate.reshape(1,-1))
    X_test_ssa = np.dot(X_test_std, Bs_normal)
    return svm.predict(X_test_ssa)

def train_data_load(window, event, value):
    if event == 'model_training_done':
        data_is_trained[0] = True 
        sg.popup('The Fault Detect Model Has been Fully Trained. You Can Test Now!', title='Notice')
    if (event == '-CSVLoad1-') | (event == '-CSVLoad2-'):
        global data_train
        csv_file = window['-TrainFileIn1-'].get() if event == '-CSVLoad1-' else window['-TrainFileIn2-'].get()
        if csv_file:
            try: 
                data_train = pd.read_csv(csv_file,
                        skiprows=1,encoding='gb18030',header=None,
                        names=["Time","OER","Perm","CO","H2","CO2","SWS","OEF","CAF","WP","BGC","BGI","TCT","TP1","TP2","TP3",
                        "TP4","OEP","CWP1","CWP2","TPD","HWP1","HWP2","AWS","CWT","HWT","TTEN","TTWS","TTWN","TTES","TTDT","DC",
                        "BT","Set_CI","Actual_CI","Last_CI","Label"])
                if data_train.empty | data_train.isnull().all().all():
                    sg.popup("The File is Empty or Incorrectly Formatted")
                else:
                    sg.popup('File Load Successfully')
            except Exception as e:
                sg.popup(f'File Import Failed: {e}')
    if (event == '-TrainModelOnline-') | (event == '-TrainModelOffline-'):
        #弹出模型正在训练的提示窗
        #sg.popup('Start Model Training. Please Wait for Completion Notification', title='Notice')
        window.perform_long_operation(lambda :model_train_once(data_train, StepOfWindow[0], LengthOfEpoch[0], NumOfSS[0]), 'model_training_done')
        
def model_train_once(data_train, win, len, ss):
    global lda, sc, Bs_normal, svm
    #数据集划分
    X_train_ori = data_train[variable]
    y_train_ori = data_train["Label"]
    index_ntrain = data_train[data_train["Label"] == 0].index
    index_ftrain = data_train[data_train["Label"] != 0].index
    #滑动时间窗+LDA特征提取
    X_train_window = dm.Timewindow(pd.DataFrame(X_train_ori), win)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda.fit_transform(X_train_window, y_train_ori)
    #特征拼接
    Xtr_normal_concat = np.concatenate([data_train[variable].iloc[index_ntrain], X_train_lda[index_ntrain]], axis = 1)
    Xtr_fault_concat = np.concatenate([data_train[variable].iloc[index_ftrain], X_train_lda[index_ftrain]], axis = 1)
    #标准化
    sc = StandardScaler()
    sc.fit(np.concatenate((Xtr_normal_concat, Xtr_fault_concat)))
    Xtr_normal_std = sc.transform(Xtr_normal_concat)
    Xtr_fault_std = sc.transform(Xtr_fault_concat)
    #平稳子空间分析ASSA
    ssa = dm.SSA(LengthOfEpoch=len, NumberOfSS=ss)
    Bs_normal, Bn_normal = ssa.SSAMain(pd.DataFrame(Xtr_normal_std))
    Ss_normal_train=np.dot(Xtr_normal_std, Bs_normal) #训练数据平稳信号 d*时间序列长度
    Ss_Fault_train = np.dot(Xtr_fault_std, Bs_normal)
    X_train_ssa = np.concatenate((Ss_normal_train, Ss_Fault_train), axis=0)
    #SVM分类算法
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_ssa, y_train_ori)

def offline_detect(window, event, value):
    if event == '-CSVLoad3-':
        csv_file = window['-TrainFileIn3-'].get()
        if csv_file:
            try: 
                global data_test_offline
                data_test_offline = pd.read_csv(csv_file,
                        skiprows=1,encoding='gb18030',header=None,
                        names=["Time","OER","Perm","CO","H2","CO2","SWS","OEF","CAF","WP","BGC","BGI","TCT","TP1","TP2","TP3",
                        "TP4","OEP","CWP1","CWP2","TPD","HWP1","HWP2","AWS","CWT","HWT","TTEN","TTWS","TTWN","TTES","TTDT","DC",
                        "BT","Set_CI","Actual_CI","Last_CI","Label"])
                window['-DataTableOffline-'].update(values=np.array(data_test_offline[heading]).tolist())
                if data_test_offline.empty | data_test_offline.isnull().all().all():
                    sg.popup("The File is Empty or Incorrectly Formatted")
                else:
                    sg.popup('File Load Successfully')
            except Exception as e:
                sg.popup(f'File Import Failed: {e}')
    if event == '-OfflinePred-':
        if data_is_trained[0]:  
            global X_test_lda_offline, X_test_ssa_offline, y_pred_offline
            X_test_window = dm.Timewindow(data_test_offline[variable], StepOfWindow[0])
            X_test_lda_offline = lda.transform(X_test_window)
            X_test_concate = np.concatenate([data_test_offline[variable], X_test_lda_offline], axis=1)
            X_test_std = sc.transform(X_test_concate)
            X_test_ssa_offline = np.dot(X_test_std, Bs_normal)  
            y_pred_offline = svm.predict(X_test_ssa_offline)
            data_test_offline['Label'] = y_pred_offline
            window['-DataTableOffline-'].update(values=np.array(data_test_offline[heading+['Label']]).tolist())
            sg.popup('Detect Completed')
        else:
            sg.popup('Please Train Fault Detect Model First.')
    if event == '-OfflineSave-':
        file_path = value['-OfflineSavePath-']
        data_test_offline.to_csv(file_path, index=False)
        sg.popup(f"The Detect Result Has Been Saved to: {file_path}")

def offline_pic(window, event, value):
    if event == '-LDAPicDraw-':
        plt.figure(figsize=(2.8, 1.7))
        for i in range(len(y_pred_offline)):
            if y_pred_offline[i] == 0:
                s0 = plt.scatter(x=X_test_lda_offline[i,0], y=X_test_lda_offline[i,1],color='r', s=5)
            if y_pred_offline[i] == 1:
                s1 = plt.scatter(x=X_test_lda_offline[i,0], y=X_test_lda_offline[i,1],color='g', s=5)  
            if y_pred_offline[i] == 2:
                s2 = plt.scatter(x=X_test_lda_offline[i,0], y=X_test_lda_offline[i,1],color='b', s=5) 
        plt.gca().set_xlabel("LDA Component1", fontsize=8, labelpad=1)
        plt.gca().set_ylabel("LDA Component2", fontsize=8, labelpad=-1)
        plt.legend((s0,s1,s2),('Normal', 'Fault1', 'Fault2'), prop={'size': 6})
        plt.xticks([-100,-75,-50,-25,0,25,50,75,100], fontsize=6)
        plt.yticks([-100,-75,-50,-25,0,25,50,75,100], fontsize=6)
        ax = plt.gca()
        ax.xaxis.set_tick_params(pad=-5)
        ax.yaxis.set_tick_params(pad=-5)
        plt.savefig('LDA_plot.png', bbox_inches='tight', pad_inches=0.02, transparent=True)
        window['-LDAImg-'].Update(filename='LDA_plot.png')
    if event =='-SSAPicDraw-':
        plt.figure(figsize=(3, 1.7))
        sns.set(style='whitegrid')
        sns.lineplot(X_test_ssa_offline, linewidth=1, linestyle='solid')
        plt.gca().set_xlabel("Time", fontsize=8, labelpad=1)
        plt.gca().set_ylabel("Normalized Value", fontsize=8, labelpad=-1)
        plt.xticks([0,25,50,75,100,125,150,175,200], fontsize=6)
        plt.yticks([-20,-15,-10,-5,0,5], fontsize=6)
        ax = plt.gca()
        ax.xaxis.set_tick_params(pad=-5)
        ax.yaxis.set_tick_params(pad=-5)
        h=ax.legend_.legendHandles
        ax.legend(h, ["ss" + str(i) for i in range(1, NumOfSS[0]+1)], prop={'size': 6}, ncol = 2, loc="lower right")
        plt.savefig('SSA_plot.png', bbox_inches='tight', pad_inches=0.02, transparent=True)
        window['-SSAImg-'].Update(filename='SSA_plot.png')
    if event == '-SavePicLDA-':
        save_path = value['-OfflinePicSavePos-']
        shutil.copy('LDA_plot.png', save_path)
        sg.popup(f"LDA Image Has Been Saved to: {save_path}")
    if event == '-SavePicSSA-':
        save_path = value['-OfflinePicSavePos-']
        shutil.copy('SSA_plot.png', save_path)
        sg.popup(f"SSA Image Has Been Saved to: {save_path}")

#设置窗口
def create_window():
    #在线监测模块的页面排布设置
    sg.theme('SystemDefaultForReal')   # 设置当前主题
    ChooseCSVText1 = sg.Text('Choose CSV File to Train Fault Detect Model')
    CSVInput1 = sg.Input(key='-TrainFileIn1-', size=50)
    CSVBrowser1 = sg.FileBrowse('Browse', size=6)
    CSVLoad1 = sg.Button('Load', key='-CSVLoad1-', size=6)
    DataLine = sg.Table(values=[], headings=heading, size=[5,10], auto_size_columns=False, col_widths=[15]+[7]*35, display_row_numbers=False, num_rows=15, key='-DataTable-', vertical_scroll_only=False, justification='center')
    StateText = sg.Text('State', key='-StateText-', justification='center', size=(40,1), font=('Arial', 50))
    AutoParam = sg.Button("Auto Parameters Choose", key='-AutoParam-', size=18)
    layout_Param = [
        [sg.Text("Param1: Window Size", size=25, justification='left'), sg.Input(key="-InputStepOfWindow-", size=10, justification='center', default_text=3, change_submits=True)],
        [sg.Text("Param2: Length Of SSA Epoch", size=25, justification='left'), sg.Input(key="-InputLenOfEpoch-", size=10, justification='center', default_text=10, change_submits=True)],
        [sg.Text("Param3: Stationary Sources", size=25, justification='left'), sg.Input(key="-InputNumOfSS-", size=10, justification='center', default_text=12, change_submits=True)],
        [sg.Button('Default Value', key='-DefaultValue-', size=18), sg.Button('Train Model', key='-TrainModelOnline-', size=15)]
    ]
    StartStop = sg.Column([
        [sg.Button('Start Detect', key='-StartDetect-', size=[15, 3], font=('Arial', 11))],
        [sg.Button('Stop Detect', key='-StopDetect-', size=[15, 3], font=('Arial', 11))]
    ])
    layout_Online = [
        [DataLine],
        [ChooseCSVText1, CSVInput1, CSVBrowser1, CSVLoad1],
        [sg.Frame('Train Model', layout_Param, size=[350, 140], element_justification='center'), StartStop, StateText]
    ]
    
    #离线检测模块的页面排布设置
    layout_offline_train = [
        [sg.Text('Choose CSV File to Train Fault Detect Model'), sg.Input(key='-TrainFileIn2-', size=50), \
            sg.FileBrowse('Browse', size=6), sg.Button('Load', key='-CSVLoad2-', size=6)],
        [sg.Text('Window Size'), sg.Input(key='-InputStepOfWindow2-', size=4, justification='center', default_text=3, change_submits=True),\
            sg.Text('Length Of SSA Epoch'), sg.Input(key='-InputLenOfEpoch2-', size=4, justification='center', default_text=10, change_submits=True),\
            sg.Text('Stationary Sources'), sg.Input(key='-InputNumOfSS2-', size=4, justification='center', default_text=12, change_submits=True),\
            sg.Button('Default Value', key='-DefaultValue2-', size=15), sg.Button('Train Model', key='-TrainModelOffline-', size=15)]
    ]
    layout_offline_test = [
        [sg.Text('Choose CSV File That Needs To Be Detected'), sg.Input(key='-TrainFileIn3-', size=45), \
            sg.FileBrowse('Browse', size=8), sg.Button('Load', key='-CSVLoad3-', size=8)],
        [sg.Table(values=[], headings=heading+["Label"], key='-DataTableOffline-', size=[5,10], auto_size_columns=False, col_widths=[15]+[7]*36, display_row_numbers=False, num_rows=3, vertical_scroll_only=False, justification='center')],
        [sg.Button('Detect', key='-OfflinePred-', size=20), sg.Text("Save Result:"), sg.Input(key='-OfflineSavePath-'), sg.FileSaveAs('Browse', size=8), sg.Button('Save', key='-OfflineSave-', size=8)]
    ]
    layout_offline_pic = [
        [sg.Column([[sg.Button('LDA Picture', key='-LDAPicDraw-', size=23)], [sg.Button('SSA Picture', key='-SSAPicDraw-', size=23)],\
                [sg.Input(key='-OfflinePicSavePos-', size=19), sg.FileSaveAs('Browse', size=5, pad=(4,0))], [sg.Button('Save LDAPic', key='-SavePicLDA-', size=10), sg.Button('Save SSAPic', key='-SavePicSSA-', size=10, pad=(13,0))]]), \
            sg.Image(filename='LDA_plot_init.png', key='-LDAImg-', size=(300,160), expand_x=True), sg.Image(filename='SSA_plot_init.png', key='-SSAImg-', size=(300,160), expand_x=True)]
    ]
    layout_Offline = [
        [sg.Frame('Train Model', layout_offline_train, element_justification='center')],
        [sg.Frame('Detect Fault', layout_offline_test, element_justification='center')],
        [sg.Frame('Show Picture', layout_offline_pic, element_justification='center')]
    ]
    
    # 创建TabGroup并添加子页面
    tab_group = sg.TabGroup([[
        sg.Tab("Online Detect", layout_Online, expand_x=True, expand_y=True, element_justification='center'),
        sg.Tab("Offline Detect", layout_Offline)
    ]])
    CloseWindowButton = sg.Button('Close')
    layout = [
        [tab_group],
    ]
    return sg.Window('Fault Detect', layout, finalize=True, size=[800,500])
    
event_callbacks = {
    '-StartDetect-': data_scroll_control,
    '-StopDetect-': data_scroll_control,
    '-CSVLoad1-': train_data_load,
    '-CSVLoad2-': train_data_load,
    '-TrainModelOnline-': train_data_load,
    '-TrainModelOffline-': train_data_load,
    'model_training_done': train_data_load,
    '-InputStepOfWindow-': param_setting,
    '-InputStepOfWindow2-': param_setting,
    '-InputLenOfEpoch-': param_setting,
    '-InputLenOfEpoch2-': param_setting,
    '-InputNumOfSS-': param_setting,
    '-InputNumOfSS2-': param_setting,
    '-DefaultValue-': param_setting,
    '-DefaultValue2-': param_setting,
    '-AutoParam-': param_setting,
    '-CSVLoad3-': offline_detect,
    '-OfflinePred-': offline_detect,
    '-OfflineSave-': offline_detect,
    '-LDAPicDraw-': offline_pic,
    '-SSAPicDraw-': offline_pic,
    '-SavePicLDA-': offline_pic,
    '-SavePicSSA-': offline_pic,
}

def main():
    # 框内大循环
    set_global_var()
    data_test_online_prepare()
    window=create_window()
    while True:
        event, value = window.read()
        if (event == sg.WIN_CLOSED) | (event == 'Close'): break
        if event in event_callbacks:
            event_callbacks[event](window, event, value) 
    window.close()

if __name__ == '__main__':
    main()