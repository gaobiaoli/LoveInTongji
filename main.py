import pandas as pd
from utils import SimilarityCalculator,LoveMatcher
import datetime

if __name__ == "__main__":
    index = {
        "性别": 3,
        "身高": 4,
        "身高_min": 23,
        "身高_max": 24,
        "颜值": 6,
        "出生日期": 7,
        "年龄_min": 25,
        "年龄_max": 26,
        "校区": 9,
        "MBTI": [11, 12, 13, 14, 18, 19, 20, 21],
        "专业": 22,
        "穿衣风格": 15,
        "兴趣爱好": 16,
        "毕业去向": 17,
        "对方专业": 22,
        "跨校": 27,
    }
    tool = SimilarityCalculator()
    data = pd.read_excel("data/demo.xlsx")
    data = data.drop(0, axis=0)
    sex_count = data.iloc[:, index["性别"]].value_counts()
    print(sex_count)
    data.iloc[:, index["性别"]] = data.iloc[:, index["性别"]].apply(lambda x: int(x == "小哥哥"))  # 男置为1，女置为0
    data.sort_values(by='性别', ascending=True, inplace=True, ignore_index=True)
    matcher = LoveMatcher(sortedDataFrame=data, sexCount=sex_count)


    # MBTi
    w1 = [29]
    M1 = tool.classTextMatch(data, index["MBTI"], w1)
    print(M1)
    # 身高
    height = [index["身高"], index["身高_min"], index["身高_max"]]
    w2 = [31]
    M2 = tool.rangeMatch(data.iloc[:, height], data.iloc[:, w2])
    # 年龄
    data.iloc[:, index["出生日期"]] = data.iloc[:, index["出生日期"]].map(lambda x: x.split("-")).map(lambda x: datetime.date.today().year - int(x[0]))
    age =[index["出生日期"], index["年龄_min"], index["年龄_max"]]
    w3 = [32]
    M3 = tool.rangeMatch(data.iloc[:, age], data.iloc[:, w3])
    # 校区
    school = [index["校区"], index["跨校"]]
    M4 = tool.schoolMatch(data, school)
    # 爱好 34
    hobby = [index["兴趣爱好"]]
    w5 = [34]
    M5 = tool.multiChoiceMatch(data, hobby, w5)
    # 穿衣风格
    style = [index["穿衣风格"]]
    M6 = tool.multiChoiceMatch(data, style, split="，")
    # 毕业后去向33
    place=[index["毕业去向"]]
    w7 = [33]
    M7 = tool.classTextMatch(data, place, w7)
    # 专业 35
    major = [index["专业"], index["对方专业"]]
    w8 = [35]
    M8 = tool.selectMatch(data, major, w8)

    Matirx=M1+M2+M3+M4*99+M5+M6+M7+M8

    # 邻接矩阵取左下角
    adjMatrix = Matirx[sex_count["小姐姐"]:, 0:sex_count["小姐姐"]]
    matcher.match(adjMatrix)
