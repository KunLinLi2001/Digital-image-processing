import numpy as np

# 将.txt文件中的数据读入矩阵中
def Read(f,m,n):
    A = np.zeros((m, n), dtype=float)  # 先创建一个全零方阵A，并且数据的类型设置为float浮点型
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        list = line.strip('\n').split('\t')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        A[A_row:] = list[0:5]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    return A
# 求解数据集的均值向量
def Mean(A):
    mean = np.average(A, axis=0) # 按列求均值
    mean = mean.transpose() # 将矩阵转置
    return mean
# 计算后验概率
def get_pdf(x,mean,cov):
    # 计算协方差的行列式
    det = np.linalg.det(cov)
    # 计算协方差的逆矩阵
    # numpy中的linalg 模块包含大量线性代数中的函数方法
    cov_inv = np.linalg.inv(cov)
    # 也可以使用**-1幂运算代表逆矩阵
    # cov_inv = cov**-1
    '''用t代表x-μ'''
    t = x-mean
    p = np.exp(-0.5*np.dot( np.dot(t.transpose(),cov_inv),t ))/pow(det,0.5)
    return p
# 找到风险最小的类别号
def find_min_risk(R1,R2,R3):
    if min(R1,R2)==R1:
        if min(R1,R3)==R1:
            Id = 1.0
        else:
            Id = 3.0
    else:
        if min(R2,R3)==R2:
            Id = 2.0
        else:
            Id = 3.0
    return Id
# 计算准确率
def count_accuracy(true,false,id,Id):
    if id==Id:
        true+=1
    else:
        false+=1
    return true,false

'''读取训练集和测试集，拆分训练样本'''
f1 = open('F:\\桌面\\Iris\\train.txt') # 打开训练集
f2 = open('F:\\桌面\\Iris\\test.txt') # 打开测试集
A = Read(f1,75,5)
B = Read(f2,75,5)

A = np.delete(A,0,axis=1) # 删除第一列(类别号)
A1 = A[0:25] # 第一类
A2 = A[25:50] # 第二类
A3 = A[50:75] # 第三类

'''计算均值向量和协方差矩阵'''
mean1 = Mean(A1)
mean2 = Mean(A2)
mean3 = Mean(A3)
# 把每一列看做一组变量求解
cov1 = np.cov(A1,rowvar=False)
cov2 = np.cov(A2,rowvar=False)
cov3 = np.cov(A3,rowvar=False)

'''基于最小错误率的贝叶斯决策'''
true = 0
false = 0
for i in range(0,75):
    B_row = B[[i]] # 获取第i行
    id = B_row[0,0] # 获取文件中的标号
    B_row = np.delete(B_row,0,axis=1) # 删除第一列(类别号)
    B_row = B_row.flatten() # 平铺成列向量
    res1 = get_pdf(B_row,mean1,cov1)
    res2 = get_pdf(B_row,mean2,cov2)
    res3 = get_pdf(B_row,mean3,cov3)
    # print(res1)
    if max(res1,res2)==res1:
        if max(res1,res3)==res1:
            Id = 1.0
        else:
            Id = 3.0
    else:
        if max(res2,res3)==res2:
            Id = 2.0
        else:
            Id = 3.0
    if(id==Id):
        true+=1
    else:
        false+=1
print("基于最小错误率：")
print("正确个数：",true)
print("错误个数：",false)
print("准确率：",true/(true+false))

'''基于最小风险的贝叶斯决策'''
true1 = false1 = 0
true2 = false2 = 0
L1 = np.array([[0,2,1],[3,0,4],[1,2,0]]) # 导入第一组损失参数矩阵
L2 = np.array([[0,1,1],[2,0,8],[1,2,0]]) # 将选错的代价损失减小
for i in range(0,75):
    B_row = B[[i]] # 获取第i行
    id = B_row[0,0] # 获取文件中的标号
    B_row = np.delete(B_row,0,axis=1) # 删除第一列(类别号)
    B_row = B_row.flatten() # 平铺成列向量
    res1 = get_pdf(B_row,mean1,cov1)
    res2 = get_pdf(B_row,mean2,cov2)
    res3 = get_pdf(B_row,mean3,cov3)
    res = [res1,res2,res3]
    # 第一组的相关数据
    R11 = sum(res*L1[0])
    R21 = sum(res*L1[1])
    R31 = sum(res*L1[2])
    Id1 = find_min_risk(R11,R21,R31)
    true1 , false1 = count_accuracy(true1,false1,id,Id1)
    # 第二组的相关数据
    R12 = sum(res*L2[0])
    R22 = sum(res*L2[1])
    R32 = sum(res*L2[2])
    Id2 = find_min_risk(R12,R22,R32)
    true2 , false2 = count_accuracy(true2,false2,id,Id2)

print("基于最小风险率：")
print("第一组损失参数矩阵结果：")
print("正确个数：",true1)
print("错误个数：",false1)
print("准确率：",true1/(true1+false1))
print("第二组损失参数矩阵结果：")
print("正确个数：",true2)
print("错误个数：",false2)
print("准确率：",true2/(true2+false2))


