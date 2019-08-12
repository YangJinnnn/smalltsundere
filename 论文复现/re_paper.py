import numpy as np
from data import get_data
from sklearn.cluster import KMeans
from sklearn import svm

x_train,y_train,x_test,y_test=get_data()
print(x_train.shape,x_test.shape)
# 归一化处理
def norm(x):
    x = x.reshape(x.shape[0], -1)
    x -= np.mean(x, axis=0)       # 减去均值
    x /= (np.std(x, axis=0, ddof=1) + 1e-7) # 除以方差
    x = x.reshape((x.shape[0],28,28))
    return x

# 对x进行切片，对每个x随机取n个6*6的小矩阵，重新组合成x(n*x.shape[0],6,6)大小的数组
def patch_random(x,n):
    x_random=np.zeros((n*x.shape[0],6,6))
    for i in range(x.shape[0]):
        for j in range(n):
            num=np.random.randint(0,23)     # 随机在（0，23）取值当作切片的左上角
            x_random[n*i+j]=x[i,num:num+6,num:num+6]
    return x_random

# 对输入样本进行定步长，定规律，同数量的切片操作，为后面组成新特征做准备
def patch_norm(x,n):
    stride=4               # 步长取4
    x_norm=np.zeros((n*x.shape[0],6,6))
    for i in range(x.shape[0]):
        row=-3             # 从（1，1）开始取
        col=1
        for j in range(n):
            row+=stride
            if row>23:
                row=1
                col+=stride
            x_norm[n*i+j]=x[i,row:row+6,col:col+6]
    return x_norm

# 计算欧式距离
def dists(x,y):
    return np.sqrt(sum(np.power(x - y, 2)))

# 对重组后的x进行聚类得到cluster(x.shape[0],k.shape[0])
def cluster(x,k):
    x=x.reshape(x.shape[0],-1)
    cluster=np.zeros((x.shape[0],k.shape[0]))
    for i in range(x.shape[0]):
        mindist=1111111
        index=0
        for j in range(k.shape[0]):
            dist=dists(x[i],k[j])  # 计算欧式距离
            if dist<mindist:       # 找到最近的k
                mindist=dist
                index=j
        cluster[i][index]=1 # 对每个切片在k.shape[0]维中距离最近的k维上标记为1，其余都为0
    return cluster

# 刚开始想到的弱智版优化特征
def new_feature1(x,n,k):
    x_new2=np.zeros((int(x.shape[0]/n),k.shape[0]*4))
    for i in range(int(x.shape[0]/n)):
        x_new=x[i*n:i*n+36]
        temp = np.zeros((4, k.shape[0]))
        temp[0]=x_new[0]+x_new[1]+x_new[2]+x_new[6]+x_new[7]+x_new[8]+x_new[12]+x_new[13]+x_new[14]
        temp[1]=x_new[3]+x_new[4]+x_new[5]+x_new[9]+x_new[10]+x_new[11]+x_new[15]+x_new[16]+x_new[17]
        temp[2]=x_new[18]+x_new[19]+x_new[20]+x_new[24]+x_new[25]+x_new[26]+x_new[30]+x_new[31]+x_new[32]
        temp[3]=x_new[21]+x_new[22]+x_new[23]+x_new[27]+x_new[28]+x_new[29]+x_new[33]+x_new[34]+x_new[35]
        x_new2[i]=temp.reshape(1,-1)
    return x_new2

# 池化
def new_feature2(x,n,k):
    x_new=np.zeros((int(x.shape[0]/n),k.shape[0]*4))
    s = int(np.sqrt(n))
    m = int(np.sqrt(n)/2)
    for i in range(int(x.shape[0]/n)):   # 遍历每张图片
        temp = np.zeros((4, k.shape[0])) # 池化矩阵
        for j in range(2):               # 遍历temp
            for h in range(m):           # 组成池化块的列
                for l in range(m):       # 组成池化块的行
                    temp[j]+=x[i*n+h+l*s+j*m]
                    temp[j+2]+=x[i*n+h+l*s+j*m+int(n/2)]
        x_new[i]=temp.reshape(1,-1)
    return x_new

# # 利用sklearn得到n个k中心 x(图片个数*切片个数，切片大小) ，n聚类个数
# def get_k(x,n):
#     kmeans = KMeans(n_clusters=n, random_state=0).fit(x)
#     k = kmeans.cluster_centers_
#     return k
#
# # 带重组维度矩阵x(n,28,28),n对每个x切片个数，k(k,x_patch[0].size)聚类中心矩阵
# def rebuild(x,n,k):
#     x=norm(x)
#     x_patch=patch_norm(x,n)
#     x_cluster=cluster(x_patch,k)
#     x_new=new_feature1(x_cluster,n,k)
#     return x_new

x_train=norm(x_train)             #将训练集标准化
x_test=norm(x_test)               #测试机标准化
x=patch_random(x_train,20)        #训练集每张切20块6*6切片
x=x.reshape(x.shape[0],-1)        #转换为n*36聚类
kmeans = KMeans(n_clusters=30, random_state=0).fit(x)
k=kmeans.cluster_centers_         #得到k中心
x_norm=patch_norm(x_train,36)     #标准切片每张图取36块
cluster1=cluster(x_norm,k)        #每个切块用one-hot编码表示（与k距离最近标1）
x_new=new_feature1(cluster1,36,k) #池化得到1维向量表示每张图像
clf=svm.SVC()
clf.fit(x_new,y_train)
x_test=patch_norm(x_test,36)
cluster2=cluster(x_test,k)
x_test=new_feature1(cluster2,36,k)
acc=clf.score(x_test,y_test)
print(acc)
