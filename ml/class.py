# -*- coding: utf-8 -*-
import jieba
import os
import pickle  # 持久化
from numpy import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import time


catelist = os.listdir("E:/Project/文本分类/ml/data/")
class_len = len(catelist) #类别数
num=0
#建立 类别-编号的字典
class_num={}
for label in catelist:
    class_num[label]=num
    num+=1

#分词
def segText(inputPath, resultPath,stopWordList):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            each_resultPathFile=each_resultPath+eachFile
            #  print(eachFile)
            with open(eachPathFile, encoding='utf-8',mode='r') as file:
                content=file.read()
            # content = str(content)
            # result = content.replace("\r\n","").strip()
            #cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开 
            result = (str(content)).replace("\r\n", "").strip()  
            cut_list=[i for i in jieba.cut(result)]
            cutResult=[]
            for cut in cut_list:
                if cut in stopWordList:
                    pass
                else:
                    cutResult.append(cut)
            with open(each_resultPathFile, encoding='utf-8',mode='w') as file:
                file.write( " ".join(cutResult))

def DirSave(inputFile, outputFile):
    catelist = os.listdir(inputFile)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签
            bunch.filenames.append(fullName)  # 保存当前文件的路径
            with open(fullName,'rb') as file:
                bunch.contents.append(file.read().strip())  # 保存文件词向量
    with open(outputFile, 'wb') as file_obj:  # 持久化
        pickle.dump(bunch, file_obj)

def getStopWord(inputFile):
    with open(inputFile, encoding='utf-8',mode='r') as file:
        stopWordList = file.read().splitlines()
    return stopWordList


def getTFIDFMat(inputPath, stopWordList, outputPath,vocabulary_path):  # 求得TF-IDF向量
    with open(inputPath, 'rb') as file:
        bunch = pickle.load(file) #反序列化
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    # 初始化向量空间
    vectorizer = TfidfVectorizer(max_features=10000,stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵
    tfidfspace.tdm= vectorizer.fit_transform(bunch.contents)

    '''

    #使用SVD对稀疏矩阵进行降维
    svd = TruncatedSVD(1000)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    Xnew = lsa.fit_transform(tfidfspace.tdm)
    tfidfspace.tdm=Xnew

    '''
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇

    with open(vocabulary_path, encoding='utf-8', mode='w') as file:
        file.write(str(tfidfspace.vocabulary ))
    with open(outputPath, 'wb') as file:
        pickle.dump(tfidfspace, file)


def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath,vocabulary_path):
    with open(testSetPath, 'rb') as file:
        bunch = pickle.load(file)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                      vocabulary={})
    # 导入训练集的词袋
    with open(trainSpacePath, 'rb') as file:
        trainbunch = pickle.load(file)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    #vectorizer = TfidfVectorizer(max_features=10000,stop_words=stopWordList, sublinear_tf=True, max_df=0.5,vocabulary=trainbunch.vocabulary)
    vectorizer = TfidfVectorizer(max_features=10000,stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    #transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    with open(vocabulary_path, encoding='utf-8', mode='w') as file:
        file.write(str(testSpace))
    '''
    #对稀疏矩阵进行降维
    svd = TruncatedSVD(1000)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    Xnew = lsa.fit_transform(testSpace.tdm)
    testSpace.tdm=Xnew
    '''
    
    with open(testSpacePath, 'wb') as file:
        pickle.dump(testSpace, file)

def display(matrix):
    #可视化输出
    plt.rcParams['font.sans-serif']=['SimHei']   #显示中文标签
    plt.rcParams['axes.unicode_minus']=False

    f,ax1 = plt.subplots(figsize=(8,4), nrows=1) #设置图像大小
    cmap = sns.cubehelix_palette(start=1.5, rot=5, gamma=0.8, as_cmap=100)  # 从数字到色彩空间的映射
    dataframe=pd.DataFrame(matrix,index=catelist,columns=catelist)
    sns.heatmap(dataframe,annot=True,ax=ax1,cmap=cmap,center=None,fmt='g')
    ax1.set_xlabel('预测类别')
    ax1.set_ylabel('真实类别')
    ax1.set_title('混淆矩阵')
    plt.show()


def trainNB(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix) #训练集数量
    numWords=len(trainMatrix[0]) #维数
    pAbusive=[0]*class_len
    pVect=[0]*class_len
    count_class=[0]*class_len
    for i in range(len(trainCategory)):
        count_class[trainCategory[i]]+=1 #统计每个类别的数量
    #计算先验概率
    for i in range(class_len):
        pAbusive[i]=count_class[i]/float(numTrainDocs)
        #print(count_class[i])
    pNum=[[1.0]*(numWords)]*class_len #平滑处理，避免运算中出现0概率
    pDenom=[2.0]*class_len
    #计算条件概率
    for i in range(numTrainDocs): #每个样本
        for j in range(class_len): #每一类
            if trainCategory[i]==j:
                pNum[j]+=trainMatrix[i] #j类别的列表相加
                pDenom[j]+=sum(trainMatrix[i]) #j类别求和
    for i in range(class_len):
        pVect[i]=pNum[i]/pDenom[i]
        #pVect[i] = log(pNum[i])-log(pDenom[i])
    return pVect,pAbusive



def classfyNB(testvec,pVect,pAbusive):  #使用训练好的模型进行预测
    p=[0]*class_len
    for i in range(class_len):
        p[i] = sum(testvec * pVect[i]) *(pAbusive[i])
        #p[i]=sum(testvec*pVect[i])+log(pAbusive[i])
    return p.index((max(p)))


'''
    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
    #X=(trainSet.tdm).toarray()
    #np.savetxt('E:/Project/文本分类/ml/matrix.csv', X, delimiter=',')
    predicted = clf.predict(testSet.tdm)
'''

def bayesAlgorithm(trainPath, testPath):
    with open(trainPath, 'rb') as file:
        trainSet = pickle.load(file)
    with open(testPath, 'rb') as file:
        testSet = pickle.load(file)

    classVec = []  # 训练类别编号
    for each in trainSet.label:
        classVec.append(class_num[each])
    start_time = time.time()
    X=(trainSet.tdm).toarray()
    pVect, pAbusive = trainNB(X, classVec)
    end_time = time.time()
    print("训练时间： ",end_time-start_time,"s") #输出训练时间
    start_time = time.time()
    predicted=[]
    Y=(testSet.tdm).toarray()
    for i in range(len(Y)):
        str=catelist[classfyNB(Y[i],pVect, pAbusive)]
        predicted.append(str)
    end_time = time.time()
    print("测试时间： ",end_time-start_time,"s") #输出测试时间
    total = len(predicted)
    count_error = 0
    real_class={} #真实类别
    predict_class={} #预测类别
    for label in catelist:
        real_class[label]=0
        predict_class[label]=0
    matrix=[] #混淆矩阵
    for i in range(class_len):  #初始化
        matrix.append([0]*class_len)
    for Truelabel, fileName, Prelable in zip(testSet.label, testSet.filenames, predicted):
        matrix[class_num[Truelabel]][class_num[Prelable]]+=1
        real_class[Truelabel]+=1
        predict_class[Prelable]+=1
        if Truelabel != Prelable: #错误分类
            count_error += 1
    print("贝叶斯分类 正确率:", 100-float(count_error) * 100 / float(total), "%")

    #计算精准率、召回率、F1
    sum_precision=0
    sum_recall=0
    sum_F1=0
    for label in catelist:
        precision=float(matrix[class_num[label]][class_num[label]])*100/float(predict_class[label])
        recall=float(matrix[class_num[label]][class_num[label]])*100/float(real_class[label])
        F1=2*precision*recall/(precision+recall)/100
        sum_precision+=(precision*float(real_class[label])/float(total))
        sum_recall+=(recall*float(real_class[label])/float(total))
        sum_F1+=(F1*float(real_class[label])/float(total))
        print(label,":   精确率：",'%.14f'%precision, "%","   召回率: ",'%.14f'%recall, "%","   F1: ",'%.14f'%(F1*100), "%")
    print();
    print("加权平均",":   精确率：",'%.14f'%sum_precision, "%","   召回率: ",'%.14f'%sum_recall, "%","   F1: ",'%.14f'%(sum_F1*100), "%")
    print()
    print("混淆矩阵（贝叶斯）：")
    
    for i in range(class_len):
        sum1=0
        for j in range(class_len):
            sum1+=matrix[i][j]
        for j in range(class_len):
            matrix[i][j]=float(matrix[i][j])/float(sum1)
            
    for i in range(10):
        for j in range(10):
            matrix[i][j]=float(format(matrix[i][j],'.4f'))
        print(matrix[i])

    display(matrix)


def svmAlgorithm(trainPath,testPath):
    with open(trainPath, 'rb') as file:
        trainSet = pickle.load(file)
    with open(testPath, 'rb') as file:
        testSet = pickle.load(file)
    model = svm.SVC(kernel='linear', C=1, gamma=1)
    #model = svm.SVC(kernel='rbf', C=1, gamma=1)
    #model = svm.SVC(kernel='poly', C=1, gamma=1)
    print("导入模型成功")
    model.fit(trainSet.tdm, trainSet.label)
    print("训练模型成功")
    #model.score(trainSet.tdm, trainSet.label)
    predicted = model.predict(testSet.tdm)
    print("预测成功")
    total = len(predicted)
    rate=0
    real_class={} #真实类别
    predict_class={} #预测类别
    for label in catelist:
        real_class[label]=0
        predict_class[label]=0
    matrix=[] #混淆矩阵
    for i in range(class_len):
        matrix.append([0]*class_len)
    for Truelabel, fileName, Prelable in zip(testSet.label, testSet.filenames, predicted):
        matrix[class_num[Truelabel]][class_num[Prelable]] += 1
        real_class[Truelabel] += 1
        predict_class[Prelable] += 1
        if Truelabel != Prelable: #错误分类
            rate += 1
    
    print("支持向量机 正确率:", 100-float(rate) * 100 / float(total), "%")

    #计算精准率、召回率、F1
    sum_precision=0
    sum_recall=0
    sum_F1=0
    for label in catelist:
        precision=float(matrix[class_num[label]][class_num[label]])*100/float(predict_class[label])
        recall=float(matrix[class_num[label]][class_num[label]])*100/float(real_class[label])
        F1=2*precision*recall/(precision+recall)/100
        sum_precision+=(precision*float(real_class[label])/float(total))
        sum_recall+=(recall*float(real_class[label])/float(total))
        sum_F1+=(F1*float(real_class[label])/float(total))
        print(label,":   精确率：",'%.14f'%precision, "%","   召回率: ",'%.14f'%recall, "%","   F1: ",'%.14f'%(F1*100), "%")

    print();
    print("加权平均",":   精确率：",'%.14f'%sum_precision, "%","   召回率: ",'%.14f'%sum_recall, "%","   F1: ",'%.14f'%(sum_F1*100), "%")
    print()
    print("混淆矩阵（支持向量机）：")
    for i in range(class_len):
        sum1=0
        for j in range(class_len):
            sum1+=matrix[i][j]
        for j in range(class_len):
            matrix[i][j]=float(matrix[i][j])/float(sum1)
            
    for i in range(10):
        for j in range(10):
            matrix[i][j]=float(format(matrix[i][j],'.4f'))
        print(matrix[i])
        
    display(matrix)

# 获取停用词
stopWordList = getStopWord("E:/Project/文本分类/ml/stop/stopword.txt")


# 对训练集进行分词
segText("E:/Project/文本分类/ml/data/",
        "E:/Project/文本分类/ml/segResult/",stopWordList)



# 输入分词，输出分词向量
DirSave("E:/Project/文本分类/ml/segResult/",
          "E:/Project/文本分类/ml/train_set.dat")


# 输入词向量，输出特征空间
getTFIDFMat("E:/Project/文本分类/ml/train_set.dat",
            stopWordList, "E:/Project/文本分类/ml/tfidfspace.dat","E:/Project/文本分类/ml/train_vocabulary.txt")




# 测试集
segText("E:/Project/文本分类/ml/test/",
        "E:/Project/文本分类/ml/test_segResult/",stopWordList)

DirSave("E:/Project/文本分类/ml/test_segResult/",
          "E:/Project/文本分类/ml/test_set.dat")
         

getTestSpace("E:/Project/文本分类/ml/test_set.dat",
             "E:/Project/文本分类/ml/tfidfspace.dat", stopWordList,
             "E:/Project/文本分类/ml/testspace.dat","E:/Project/文本分类/ml/test_vocabulary.txt")


# 贝叶斯分类
bayesAlgorithm("E:/Project/文本分类/ml/tfidfspace.dat",
               "E:/Project/文本分类/ml/testspace.dat")

# 支持向量机分类
svmAlgorithm("E:/Project/文本分类/ml/tfidfspace.dat",
               "E:/Project/文本分类/ml/testspace.dat")


