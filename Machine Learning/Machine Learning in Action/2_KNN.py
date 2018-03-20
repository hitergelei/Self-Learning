# -*- coding: utf-8 -*-
"""
k-近邻算法
标签分类
group：数据集
lables:标签分类
"""
import numpy as np 
import operator

def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group,labels

# =============================================================================
# if __name__ == '__main__':
# 	 #创建数据集
#     group, labels = createDataSet()
#     print(group)
#     print(labels)
# =============================================================================


"""
k-近邻算法
根据两点距离公式，计算距离，选择距离最小的前k个点，并返回分类结果。
"""

def classify0(inX,dataSet, labels,k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #np.tile()表示：在行方向上重复inX数据共1次，在列方向重复inX共dataSetSize次
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #print(sqDiffMat)
    #sum()表示所有元素相加，sum(0)列向量相加，sum(1)行向量分别相加
    sqDistances = sqDiffMat.sum(axis = 1)
    #print(sqDistances)
    #开方求距离
    distances = sqDistances**0.5
    print(distances)
    #argsort()返回的是distances中元素从小到大排序的索引值
    sortedDistIndicies =  distances.argsort()
    print("sortedDostIndicies=",sortedDistIndicies)
    #定义一个记录类别次数的字典
    classCount = {}
    
    for i in range(k):
        print("sortedDistIndicies[",i,"] = ",sortedDistIndicies[i])
        voteIlabel = labels[sortedDistIndicies[i]] #排名前k个贴标签
        print("voteIlabel=",voteIlabel)
        #dict.get(key,defualt = None),字典的get()方法，返回指定键的值，如果值不在字典中，返回默认值
        #计算类别次数
        #print ("类别 次数：",classCount.get(voteIlabel,0))
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1  #不断累加计数的过程，体现在字典的更新中
        print("classCount[",voteIlabel,"]为 ：",classCount[voteIlabel])
        
        #python3中用items()替换python2中的iteritems()
	
        #key = operator.itemgetter(1)根据字典的值进行排序
        #key = operator.itemgetter(0)根据字典的键进行排列
	#如下面例子：
	# b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
        # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，
	# 形成多级排序。 
        #reverse = True表示降序排列字典，sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
	#sorted()排序方法不懂的可以参考：http://www.cnblogs.com/HongjianChen/p/8612176.html
        print("sortedClassCount： ",sortedClassCount)
         #返回出现次数最多的value的key
        return sortedClassCount[0][0]
     
if __name__ == '__main__':
    #创建数据集
    group,labels = createDataSet()
    test = [101,20]
    #KNN分类
    test_class = classify0(test,group,labels,3)
    #打印分类结果
    print(test_class)
     
     

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    
