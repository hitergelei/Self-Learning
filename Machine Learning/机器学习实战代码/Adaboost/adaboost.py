import numpy as np
import matplotlib.pyplot as plt
def loadSimpData():
	dataMat = np.matrix([[1., 2.1],
		              [2., 1.1],
		              [1.3, 1.],
		              [1., 1.],
		              [2., 1.]])

	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	
	return dataMat, classLabels

def showDataSet(dataMat, label):
	"""
	数据可视化
	Parameters:
		dataMat - 数据矩阵
		label - 数据标签
	Returns:
	    无
	"""
	
	# 方法一 via CHJ
	data = np.array(dataMat)
	#print(data)
	for i in range(len(data)):
		if label[i] == 1.0:
			#用plt.scatter画散点图也行
			plt.plot(data[i][0], data[i][1], marker = 'o', color = 'red')
		else:
			plt.plot(data[i][0], data[i][1],  marker = 's',color = 'blue')

	plt.show()

	# # 方法二	
	# data_plus = [] #正样本
	# data_mins = [] #负样本
	# for i in range(len(dataMat)):
	# 	if label[i] > 0:
	# 		data_plus.append(dataMat[i])
	# 	else:
	# 		data_mins.append(dataMat[i])

	# data_plus_np = np.array(data_plus)  #转换成numpy矩阵  #[[[1.  2.1]], [[2.  1.1]], [[2.  1. ]]]
	# data_mins_np = np.array(data_mins)  #转换成numpy矩阵  #[[[1.3 1. ]], [[1.  1. ]]]
	# #需要转置，否则绘图不正确
	# x = np.transpose(data_plus_np)  # [[[1.  2.  2. ]], [[2.1 1.1 1. ]]]
	# y = np.transpose(data_mins_np)  # [[[1.3 1. ]], [[1.  1. ]]]
	# plt.scatter(x[0], x[1])  # 正样本散点图
	# plt.scatter(y[0], y[1])  # 负样本散点图
	# plt.show()


print("------------7.4节:基于单层决策树构建分类器-----------------------")

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	"""
	(将数据集，按照feature列的value进行 二分法切分比较来赋值分类)
	单层决策树分类函数
	Parameters:
	    dataMatrix - 数据矩阵
	    dimen - 第dimen列，也就是第几列特征
	    threshVal - 阈值（特征列要比较的值）
	    threshIneq - 阈值不等式（这里有两个：lt和gt)
	Returns:
	    retArray - 分类结果（np.array类型）
	"""
	retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 生成m 行1列的单位矩阵
	# thresh_ineq == 'lt'表示阈值不等式取lt(less than)
	if threshIneq == 'lt':
		# data_mat[:, dimen] 表示数据集中第dimen列的所有值
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0   # 如果小于等于阈值,则赋值为-1


	else:  # 表示阈值不等式取gt(great than)
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0	 # 如果大于阈值,则赋值为-1

	return retArray

def buildStump(dataArr, classLabels, D):

	"""
    找到数据集上的最佳单层决策树 -- 单层决策树是指只考虑其中的一个特征，在该特征的基础上进行分类，
    寻找分类错误率最低的阈值即可， 非常简单
    例如本文例子中，如果以第一列特征(dimen = 1)为基础，阈值v选择1.3，并且设置lt：<=1.3的为-1，< 1.3的为+1; gt: <1.3的为+1，<= 1.3的为-1， 
    这样就构造出了一个二分类器
	Parameters:
	    dataArr - 数据矩阵
	    classLabels - 数据标签
	    D - 样本权重
	Returns:
	    bestStump - 最佳单层决策树信息
	    minError - 最小误差
	    bestClassEst - 最佳的分类结果   
	"""

	dataMatrix = np.mat(dataArr)
	labelMat = np.mat(classLabels).T
	m, n = np.shape(dataMatrix)   # m行n列
	
	
     # numSteps用于在特征的所有可能值上进行遍历
	numSteps = 10.0
	bestStump = {}
	bestClassEst = np.mat(np.zeros((m, 1)))  
	minError = float('inf')   # np.inf    # 最小误差初始化为正无穷大
    # 第一层循环：对n列特征进行遍历(如本例中，n =2 )
	for i in range(n):
	    rangeMin = dataMatrix[:, i].min()  # 每次找到该特征中最小的值和最大的值	    
	    rangeMax = dataMatrix[:, i].max()

	    stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长（确定需要多大的步长) -- 阈值查询的步长
        # 第二层循环
	    for j in range(-1, int(numSteps) + 1):
	    	'''
			lt(less than)是指在该阈值下，如果<阈值，则分类为-1
			gt(greater than)是指在该阈值下，如果>阈值，则分类为-1
			就这个题目来说，两者加起来误差肯定为1
			'''
            #第三层循环是在大于和小于之间切换不等式
	    	for inequal in ['lt', 'gt']:  # 遍历小于和大于的情况。
    			threshVal = (rangeMin + float(j) * stepSize)   # 计算阈值
    			predictedVals = stumpClassify(dataMatrix, i , threshVal, inequal) # 计算分类结果
    			errArr = np.mat(np.ones((m,1)))   # 初始化误差矩阵
    			
    			errArr[predictedVals == labelMat] = 0  # 分类正确的，赋值为0
    			# 基于权重向量D而不是其他错误指标来评价分类器的，不同的分类器计算方法不一样
    			weightedError = D.T * errArr # 计算误差--这里没有采用常规方法来评价这个分类器的分类准确率，而是乘上了权重
    			# print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' \
    			# 	%(i, threshVal, inequal, weightedError))
    			if weightedError < minError:
    				minError = weightedError
    				bestClassEst = predictedVals.copy()
    				bestStump['dim'] = i
    				bestStump['thresh'] = threshVal
    				bestStump['ineq'] = inequal


	return bestStump, minError,bestClassEst


print("----------7.4：完整AdaBoost算法的实现----------------------")
def adaBoostTrainDS(dataArr, classLabels, numIt= 40):
	"""
	使用AdaBoost算法提升分类器性能
	Parameters:
	    dataArr - 数据矩阵
	    classLabels - 数据标签
	    numIt - 最大迭代次数
	Returns:
		weakClassArr - 训练好的分类器
		aggClassEst - 类别估计累计值
	"""
	weakClassArr = []
	m = np.shape(dataArr)[0]
	D = np.mat(np.ones((m,1))/m)  # 初始化权重
	aggClassEst = np.mat(np.zeros((m,1)))

	for i in range(numIt):  # 迭代次数
		# 得到决策树的模型
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)  #构建单个单层决策树
		# print("D: ", D.T)
		# 计算弱学习算法的权重alpha，使error不等于0，因为分母不能为0
		# alpha 目的主要是计算每一个分类器实例的权重(加和就是分类结果)
		# 计算每个分类器的 alpha 权重值
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))

		bestStump['alpha'] = alpha  # 存储若学习算法的权重  # store Stump Params in Array
		
		weakClassArr.append(bestStump)  # 存储单层决策树
		#print("classEst = ", classEst.T)
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
		# print("expon = ", expon.T)
		D = np.multiply(D, np.exp(expon)) 
		# print("权重分布D = ", D.T)  
		D = D / D.sum() # 根据样本权重公式，更新样本权重
		# print("更新后权重分布D = ", D.T)  # 使D成为一个概率分布

		#计算AdaBoost的误差，当误差errorRate为0时，退出循环
		# 计算所有类别估计累计值--注意 这里包括了目前已经训练好的每一个弱分类器
		aggClassEst = aggClassEst + alpha * classEst
		#print("aggClassEst: ", aggClassEst.T)

		#计算分类器集成后的错误矩阵，错误设置为1，便于后续计算
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))  #目前的集成分类器，分类错误
		# print("aggErrors = \n", aggErrors)
		errorRate = aggErrors.sum() / m  
		# errorRate:集成分类器分类错误率（平均），如果错误率为0，则整个集成算法停止，训练完成
		#print("total error: ", errorRate)
		print("集成第%s个弱分类器后的错误率ε = %.3f" %(i, errorRate))
		# 如错误率为0，此时分类器以达最佳，循环终止
		if errorRate == 0.0:
			break

	return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
	"""
	Parameters:
		datToClass - 待分类样例
		classifierArr - 训练好的分类器
	Returns:
		分类结果
	"""

	dataMatrix = np.mat(datToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(len(classifierArr)):   # 遍历所有的分类器，进行分类
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		print(aggClassEst)  

	return np.sign(aggClassEst)


##########################7.6 示例： 在一个难数据集上应用AdaBoost##########################
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))  #获得特征（包括类别标签列）的个数
	# print("特征的个数： ", numFeat)  # 输出22个（包括类别标签列）
	dataMat = [] ; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():   # 遍历行数（样本行）
		lineArr = []
		curLine = line.strip().split('\t')  # curLine该行（当前行）的所有特征的数据
		# print("curLine = ", curLine)
		for i in range(numFeat - 1):  # 遍历所有特征（除去类别标签列的）
			lineArr.append(float(curLine[i]))  # 获得该行（当前行）除最后一列的其他所有特征的数据（格式化为浮点型）
		dataMat.append(lineArr)
		# i = 0列时，dataMat =  [[2.0, 1.0, 38.5, 66.0, 28.0, 3.0, 3.0, 0.0, 2.0, 5.0, 4.0, 4.0, \
		#                        0.0, 0.0, 0.0, 3.0, 5.0, 45.0, 8.4, 0.0, 0.0]]
		# print("dataMat = ", dataMat)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat

#################7.7节：非均衡分类问题--TPR和FPR和ROC曲线绘制-###########################


def plotRoc(predStrengths, classLabels):
	"""
	Parameters:
		predStrengths--分类器的预测强度
		classLabels--类别
	Returns:
	无
	"""
	import matplotlib.pyplot as plt
	cur = (1.0, 1.0)  # 绘制光标的位置
	ySum = 0.0         # 用于计算AUC面积
	numPosClas = np.sum(np.array(classLabels) == 1.0) # 统计正类的数量
	yStep = 1 / float(numPosClas)   # y轴步长
	xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长

	sortedIndicies = predStrengths.argsort()  # 预测强度排序，从低到高
	# print("sortedIndicies = \n", sortedIndicies)
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	# print("sortedIndicies.tolist()[0] = \n", sortedIndicies.tolist()[0])
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0 ; delY = yStep
		else:
			delX = xStep; delY = 0
			# 高度累加（xStep是固定的，最后ySum * xStep即可求AUC面积)
			ySum = ySum + cur[1]  # 注意每次cur[1]都可能变化了
		# 绘制ROC(两点一线) 每次点(cur[0],cur[1])和点(cur[0] - delX, cur[1] - delY)连成一条直线
		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b') 
		cur = (cur[0] - delX, cur[1] - delY)    # 更新绘制光标的位置
		#print("cur ====== \n",cur)

	ax.plot([0,1], [0,1], 'b--')  # 画虚线(即随机猜测的结果曲线)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for AdaBoost Horse Colic Detection System')
	ax.axis([0,1,0,1])
	print("AUC面积为：", ySum * xStep)
	plt.show()




if __name__ == '__main__':
	
	dataMat, classLabels = loadSimpData()
	# print(dataMat, classLabels)
	# showDataSet(dataMat, classLabels)
	D = np.mat(np.ones((5,1))/5)
	# print(buildStump(dataMat, classLabels, D))
	weakClassArr, aggClassEst = adaBoostTrainDS(dataMat, classLabels, 9)
	# print("weakClassArr = ", weakClassArr) # 打印出所有弱分类器
	# print("aggClassEst = ", aggClassEst)
	# print('-------分割线-------')
	# print(adaClassify([0,0], weakClassArr))
	# print('=======分割线=======')
	# print(adaClassify([[5,5], [0,0]], weakClassArr))
	print()

	print("-----第7.6节 示例：在一个难数据集上应用AdaBoost-----")
	print()
	dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
	weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
	print("----------用在测试集试试-----------------------")
	testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
	# print("weakClassArr = ", weakClassArr)
	predictions = adaClassify(dataArr,weakClassArr)
	errArr = np.mat(np.ones((len(dataArr), 1)))
	print("训练集上的错误数：%s" %(errArr[predictions != np.mat(LabelArr).T].sum()))
	print('训练集上的错误率： %.3f%%' %float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))

	predictions = adaClassify(testArr, weakClassArr)
	errArr = np.mat(np.ones((len(testArr), 1)))
	print("测试集上的错误数：%s" %(errArr[predictions != np.mat(testLabelArr).T].sum()))
	print('测试集上的错误率： %.3f%%' %float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
	print("-------------7.7节：非均衡分类问题--TPR和FPR和ROC曲线绘制--------------------")
	print()
	plotRoc(aggClassEst.T, LabelArr)


	
