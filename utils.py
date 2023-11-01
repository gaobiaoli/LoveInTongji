import pandas as pd
import numpy as np
import random
import time
from scipy.spatial.distance import pdist, squareform
import scipy.optimize
import datetime
class SimilarityCalculator(object):
    def __init__(self):
        pass
    def selectMatch(self,dataFrame,iloc,weight):
        ###处理选择筛选问题（专业）
        w = np.array(dataFrame.iloc[:, weight])
        w = w / w.max()
        w_Matrix = np.matmul(w.reshape(-1, 1), w.reshape(1, -1))
        text=dataFrame.iloc[:,iloc]
        sortlist=text.iloc[:,-1]
        num=len(sortlist)
        matrix=np.ones([num,num])
        for i in range(num):
            temp=text.iloc[i,0]
            match=sortlist.apply(lambda x: temp not in x.split(","))
            matrix[i,:]=np.array(match).reshape(1,-1).astype(np.float32)
        return ((matrix+matrix.transpose())/2)*w_Matrix
    def multiChoiceMatch(self,dataFrame,iloc,weight=None,split=","):
        '''
        多选题相似度计算(适用于兴趣爱好、穿衣风格)
        相似度为交并比
        '''
        ##
        text = dataFrame.iloc[:, iloc]
        num = len(text)
        if weight is not None:
            w = np.array(dataFrame.iloc[:, weight])
            w = w / w.max()
            w_Matrix = np.matmul(w.reshape(-1, 1), w.reshape(1, -1))
        else:
            w_Matrix = np.ones([num, num])


        matrix=np.ones([num,num])
        for i in range(num):
            temp=text.iloc[i,0]
            similarity=text.applymap(lambda x: self.calSimilarity(temp,x,split=split))
            matrix[i, :]=np.array(similarity).reshape(1,-1).astype(np.float32)
        return matrix*w_Matrix
    def calSimilarity(self,text1,text2,split=","):

        # 计算交并比IoU
        list1=text1.split(split)
        list2=text2.split(split)
        return 1-len(set(list1) & set(list2))/len(set(list1+list2))
    def classTextMatch(self,dataFrame,iloc,weight):
        ### 处理代表类别的文本标签
        ##权重折减矩阵
        w = np.array(dataFrame.iloc[:, weight])
        w=w/w.max()
        w_Matrix = np.matmul(w.reshape(-1, 1), w.reshape(1, -1))
        ##得分矩阵
        text=dataFrame.iloc[:,iloc]
        textToLabel=pd.get_dummies(text)
        array=np.array(textToLabel)
        Matirx=squareform(pdist(array,"cosine"), force='no', checks=True)

        return Matirx*w_Matrix
    def rangeMatch(self,heightRange1,weight,heightRange2=None):
        '''
        范围匹配问题（适用于身高、年龄匹配）
        双向匹配： 0  单向匹配： 0.5 无匹配：   1
        单变量时为自相似度
        '''

        w = np.array( weight)

        w = w / w.max()
        w_Matrix = np.matmul(w.reshape(-1, 1), w.reshape(1, -1))
        if heightRange2 is not None:
            heightRange1=np.array(heightRange1)
            heightRange2=np.array(heightRange2)
            num1=len(heightRange1)
            num2=len(heightRange2)
            matchMatrix=np.ones([num1,num2])*99
            for i in range(num1):
                for j in range(num2):
                    matchMatrix[i,j]=self.isRangeMatch(heightRange1[i],heightRange2[j])
            return matchMatrix
        else:
            heightRange1 = np.array(heightRange1)
            num1 = len(heightRange1)
            matchMatrix = np.ones([num1,num1]) * 99
            for i in range(num1):
                temp=heightRange1[i]
                for j in range(num1):
                    matchMatrix[i, j] = self.isRangeMatch(temp, heightRange1[j])
            return matchMatrix*w_Matrix

    def isRangeMatch(self,range1,range2):
        flag=1.0
        range1=[int(i) for i in range1]
        range2=[int(i) for i in range2]
        if (range1[0]>=range2[1]) & (range1[0]<=range2[2]):
            flag-=0.5
            pass
        if (range2[0]>=range1[1]) & (range2[0]<=range1[2]):
            flag-=0.5
            pass
        return flag

    def schoolMatch(self,dataFrame,iloc):
        text = dataFrame.iloc[:, iloc]
        accept  = pd.get_dummies(text.iloc[:,-1])
        num = len(text)
        schoolMatch = np.ones([num, num])
        school=np.array(text.iloc[:, 0])

        for i in range(num):
            temp=text.iloc[i,0]
            if accept["可以接受跨校区/跨校"].iloc[i]:
                schoolMatch[i,:]=np.zeros([1,num]).astype(np.float32)
                continue
            elif accept["不能接受跨校区/跨校"].iloc[i]:
                match=school!=temp
                schoolMatch[i, :] = np.array(match).reshape(1, -1).astype(np.float32)
                del match
                continue
            elif accept["仅能接受跨校区"].iloc[i]:
                match = school == temp
                schoolMatch[i, :] = np.array(match).reshape(1, -1).astype(np.float32)
                del match
                continue
        #return schoolMatch
        return (schoolMatch+schoolMatch.transpose())/2

    def ageCal(self,dataFrame,i_brith):
        dataFrame.iloc[:,i_brith].map(lambda x: x.split("-")).map(lambda x: datetime.date.today().year - int(x[0]))


class LoveMatcher():

    def __init__(self,
                 sortedDataFrame,
                 sexCount,
                 mathchSavePath="result/match.xlsx",
                 noMatchSavePath="result/nomatch.xlsx",
                 sexTag=["小姐姐", "小哥哥"], ):
        self.dataFrame = sortedDataFrame
        self.sexCount = sexCount
        self.mathchSavePath = mathchSavePath
        self.noMatchSavePath = noMatchSavePath
        self.sexTag = sexTag
        pass

    def calLoss(self, matrix, match):
        '''
        计算匹配程度
        :param matrix: 邻接矩阵
        :param match: matchPair
        :return: [平均Loss,Loss_list]
        '''
        sum = 0
        Loss = []
        for i in range(len(match[0])):
            sum += matrix[match[0][i], match[1][i]]
            Loss.append(matrix[match[0][i], match[1][i]])
        return [sum / len(Loss), Loss]

    def findNotMatching(self, matchList):
        '''
        寻找未匹配的Id
        :param matchList: matchPair
        :return:
        '''
        girlNotMatch = [i for i in range(self.sexCount[self.sexTag[0]]) if i not in matchList[1]]
        boyNotMatch = [i + self.sexCount[self.sexTag[0]] for i in range(self.sexCount[self.sexTag[1]]) if
                       i + self.sexCount[self.sexTag[0]] not in matchList[0]]
        return girlNotMatch, boyNotMatch

    def match(self, adjMatrix, saveCol=[1, 2, 36,37]):
        '''
        匹配主函数+保存功能
        :param adjMatrix: 邻接矩阵（未防止同性匹配，邻接矩阵取左下角）
        :param saveCol: 需要保存的列序号
        :return:
        '''
        matchPair = scipy.optimize.linear_sum_assignment(adjMatrix, maximize=False)
        # (match[0][i],match[1][i]) 为一个匹配对，（男生ID,女生ID）
        loss = self.calLoss(adjMatrix, matchPair)
        print("平均相似度为{}".format(loss[0]))
        lossFrame = pd.DataFrame(loss[1]+loss[1]).reset_index(drop=True)
        lossFrame.columns=['匹配程度']
        matchPair = list(matchPair)
        matchPair[0] += self.sexCount[self.sexTag[0]]
        boyData = self.dataFrame.iloc[matchPair[0], saveCol].reset_index(drop=True)
        girlData = self.dataFrame.iloc[matchPair[1], saveCol].reset_index(drop=True)
        matchData1 = pd.concat([boyData, girlData], axis=1)
        matchData2 = pd.concat([girlData, boyData], axis=1)
        matchData  = pd.concat([matchData1, matchData2], axis=0)
        matchData  = pd.concat([matchData.reset_index(drop=True), lossFrame], axis=1)
        matchData.to_excel(self.mathchSavePath)
        #寻找未被匹配的id
        girlNotMatch, boyNotMatch = self.findNotMatching(matchPair)
        if girlNotMatch:
            self.dataFrame.iloc[girlNotMatch, saveCol].reset_index(drop=True).to_excel(self.noMatchSavePath)
        if boyNotMatch:
            self.dataFrame.iloc[boyNotMatch, saveCol].reset_index(drop=True).to_excel(self.noMatchSavePath)




