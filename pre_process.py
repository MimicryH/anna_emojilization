import pandas as pd
import os

basedir = os.path.dirname(os.path.abspath(__file__))
tabledir = os.path.join(basedir, 'xlsx/zhaojing1.xlsx')
df = pd.read_excel(tabledir, "Sheet1")
question_type = df["问题类型"]
question_type_qa = []
df_ = df.isnull()
null_check_question = df_["问题类型"]
null_check_speaker = df_["说话人"]
speaker = df["说话人"]
i = 0
print(null_check_question)
for index, s in enumerate(speaker):
    if null_check_speaker[index] == False:
        print("index = ", index, "i = ", i)
        if null_check_question[index] == True:
            if len(question_type_qa) <= index:
                i = index + 1
                print(i)
                while i < len(null_check_question) and null_check_question[i] == True:
                    i += 1
                for j in range(index, i):
                    if i < len(question_type):
                        question_type_qa.append(question_type[i])
        else:
            question_type_qa.append(question_type[index])

s = pd.Series(question_type_qa)
print(question_type_qa)
s.to_frame('question_type').to_excel("result.xlsx")
