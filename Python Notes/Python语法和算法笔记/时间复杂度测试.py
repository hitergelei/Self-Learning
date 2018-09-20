'''
裘宗燕《数据结构与算法Python语言描述》
'''
def test1(n):
    lst = [ ]
    for i in range(n*10000):
        lst = lst + [ i ]
    return lst

def test2(n):
    lst = [ ]
    for i in range(n*10000):
        lst.append(i)
    return lst

def test3(n):
    return [ i for i in range(n*10000)]

def test4(n):
    return list(range(n*10000))

if __name__ == '__main__':
    import time
    start1 = time.time()
    #test1(2000)  #时间相当慢，超时
    #end = time.time()
    #print("test1花费时间：%s s" %(time.time()-start1))
    start2 = time.time()
    test2(2000)
    print("test2花费时间：%s s" %(time.time()-start2))
    start3 = time.time()
    test3(2000)
    print("test3花费时间：%s s" %(time.time()-start3))
    start4 = time.time()
    test4(2000)
    print("test4花费时间：%s s" %(time.time()-start4))
    
    
    
    
