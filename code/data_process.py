import os.path

from textblob import TextBlob
import datetime
import pandas as pd
import category_encoders as ce
import json
class GetUsefulLevel():
    def __init__(self,input_data):
        self.input_data=input_data
    def get_quantile_20_values(self):
        '''按照分位数切分为20等分'''
        grade = pd.DataFrame(columns=['quantile', 'value'])
        for i in range(0, 21):
            grade.loc[i, 'quantile'] = i / 20.0
            grade.loc[i, 'value'] = self.input_data.quantile(i / 20.0)
        cut_point = grade['value'].tolist()
        s_unique = []
        for i in range(len(cut_point)):
            if cut_point[i] not in s_unique:
                s_unique.append(cut_point[i])
        return s_unique
    def get_quantile_interregional(self):
        '''根据去重后的分位数，构造区间'''
        s_unique=self.get_quantile_20_values()
        interregional = []
        for i in range(1, len(s_unique)):
            interregional.append([i, s_unique[i - 1], s_unique[i]])
            if i == len(s_unique) - 1 and len(interregional) < 20:
                interregional.append([i + 1, s_unique[i], s_unique[i]])
        return interregional
    def get_current_level(self,item_data):
        """根据分位数区间获取当前数所对应的的级别"""
        interregional=self.get_quantile_interregional()
        level = 0
        for i in range(len(interregional)):
            if item_data >= interregional[i][1] and item_data < interregional[i][2]:
                level = interregional[i][0]
                break
            elif interregional[i][1] == interregional[i][2]:
                level = interregional[i][0]
                break
        return level

class DataProcess():
    def __init__(self):
        print('dataprocess:')

    def sen_analy(self,text):
        blob = TextBlob(text)
        sentiment = blob.sentiment
        polarity=sentiment.polarity
        if -1 <= polarity < -0.1:
            return -1
        elif -0.1 <= polarity < 0.1:
            return 0
        else:
            return 1


    def text_replace(self,text,word='&#039;'):
        '''替换掉评论数据中的乱码字符'''
        return text.replace(word, "'")

    def time_format(self,date):
        '''将英文文本时间转为datetime格式，并获取年份'''
        format_date = datetime.datetime.strptime(date, '%B %d, %Y')
        year = format_date.year
        return year

    def get_label_dict(self,data):
        '''将字符串数据赋予编号'''
        condition_dict = dict()
        for k, v in enumerate(data):
            condition_dict[v] = k
        return condition_dict
    def target_encoder(self,input_data,target,column):
        '''使用目标编码对drugName、condition进行编码，并将编码结果保存为json文件'''
        base_path='../save_file'
        filename=column+'.json'
        save_path=os.path.join(base_path,filename)
        encoder_dict={}
        target_enc = ce.TargetEncoder()
        target_enc.fit(input_data, target)
        encode_data = target_enc.transform(input_data)
        values=pd.Series(encode_data.iloc[:,0])
        for k,v in zip(input_data,values):
            encoder_dict[k]=v
        item = json.dumps(encoder_dict)
        with open(save_path, "w", encoding='utf-8') as f:
            f.write(item)
            print("^_^ write success")
        return encode_data

    def side_effect_level(self,input_data):
        '''对药品副作用进行顺序编码'''
        effect_dict={'No Side Effects':0,'Mild Side Effects':1,'Moderate Side Effects':2,
                     'Severe Side Effects':3,'Extremely Severe Side Effects':4}
        return input_data.map(effect_dict)

if __name__ == '__main__':
    text="I&#039;ve had this for about 5 months now.. I had NO IDEA others were going through the same things I am... I am tired, moody and have gained about 25 lbs.. I have been working out more and couldn&#039;t figure out what was going on until now. I am dry during sex, not in the mood most of the time. Acne has been pretty bad, didn&#039;t know why that was either (don&#039;t usually have acne). For a few weeks now my body feels like it is pregnant. The moods, tired, etc.... but all tests say not pregnant. It makes me uncomfortable to have so many things going on."
    data_process=DataProcess()
    result=data_process.text_replace(text,word='&#039;')
    print(result)


