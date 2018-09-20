'''
这里的.txt文件格式如：http://kubicode.me/img/AUC-Calculation-by-Python/evaluate_result.txt
'''
#绘制二分类ROC曲线
import matplotlib.pyplot as plt 
from math import log, exp, sqrt

evaluate_result = './evaluate_result.txt'

db  = [ ]   #[score,nonclk,clk]
pos, neg = 0, 0
with open(evaluate_result, 'r') as fs:
    for line in fs:
        nonclk, clk, score = line.strip().split('\t')  #变成列表list形式
        nonclk = int(nonclk)
        clk = int(clk)
        db.append([score,nonclk,clk])  

        score = float(score)
        neg = neg + nonclk
        #print("neg数量====", neg, end = '')
        pos = pos + clk
        #print(",  pos数量====", pos)
        
#print("db= ",db)

db = sorted(db, key=lambda x:x[0], reverse=True)  #对ad的score进行降序排序
#print("db_reverse= ",db)

#计算ROC坐标点
xy_arr = [ ]
tp, fp = 0., 0.

for i in range(len(db)):
    fp += db[i][1]
    #print("fp ===", fp, end = '')
    tp += db[i][2]   
    #print(",   tp ===",tp)
    xy_arr.append([fp/neg, tp/pos])   #fp除以negative数目， tp除以positive数目。作为坐标点
#print(xy_arr)  #[[0.06666666666666667, 0.0], [0.06666666666666667, 0.029411764705882353],...]

# 计算曲线下面积
auc = 0.
prev_x = 0
for x, y in xy_arr:   #  x= fp/neg， y = tp/pos
    if x != prev_x:
        auc += (x - prev_x) * y  #矩形面积累加
        prev_x = x

print("The auc is %s" %auc)

x = [ v[0] for v in xy_arr]
y = [ v[1] for v in xy_arr]
plt.title("ROC curve of %s (AUC = %.4f)" % ('svm',auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(x, y)
plt.show()
        
    


















        
        
