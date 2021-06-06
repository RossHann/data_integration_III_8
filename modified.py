# ?????????????????????????????????????????????????
import numpy as np
# Pandas??????????????????????????????????????????
import pandas as pd
# os??????????????????
import os
# ???????
import arrow as ar
# ??
import matplotlib.pyplot as plt
# Seaborn??matplotlib???????????API??,??????????,
import seaborn as sns
# ????matplotlib?????package, Matplotlib ??????
from pyplotz.pyplotz import PyplotZ
pltz=PyplotZ()
# ????????
from palettable.colorbrewer.sequential import Blues_9,BuGn_9,Greys_3,PuRd_5
# re???python???????????,??????????????????????
import re
# ????????
import time
# ????
from tqdm import tqdm
# ????
import pickle
# ????
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# matplotlib????
# plt.rcParams['font.sans-serif'] = ['Songti SC']  # ?????????????????????????Windows??SimHei
plt.rcParams['axes.unicode_minus'] = False  # ????????
# ??????????????????
import warnings
warnings.filterwarnings('ignore')
# matplotlib??????
plt.style.use('fivethirtyeight')
# ???python console??????
# %matplotlib inline
# ??
os.chdir('.')

# round1
# train_path_people_1='./round1_train_cut_by_people.txt'
# train_path_type_1='./round1_train_cut_by_type.txt'
# test_path_a_1='./round1_ijcai_18_test_a_20180301.txt'
# test_path_b_1='./round1_ijcai_18_test_b_20180418.txt'
# # round2
# train_path_type_2='./round2_train_cut_by_type.txt'
# test_path_a_2='./round2_test_a.txt'
# test_path_b_2='./round2_test_b.txt'
# dataset_cut
train_path = './090.txt'
test_path = './test_no_label.txt'
train=pd.read_table(train_path,delimiter=' ')
test=pd.read_table(test_path,delimiter=' ')

print('???? {} ?????? {} ?'.format(train.shape[0],test.shape[0]))
print('???? {} ?????? {} ?'.format(train.shape[1],test.shape[1]))
print('???? {} ????Instance_id'.format(train.shape[0]-len(train.instance_id.unique())))
# intersect1d ?????????????
print('????????? {} ????Instance_id'.format(len(np.intersect1d(train.instance_id.values,test.instance_id.values))))
# ?????????
print('????????') if True not in train.isnull().any().values else print('????????')
# ?????????
print('????????') if True not in test.isnull().any().values else print('????????')

train.head()
test.head()

print('??????'+str(len(train))+'???')

print('???????????'+str(len(train[train.is_trade==0])/len(train[train.is_trade==1])))

print('????'+str(len(train['item_id'].unique()))+'???????,??'+str(len(train['shop_id'].unique()))+'??????')

# ?????????????id
for x in ['instance_id','is_trade','item_id','user_id','context_id','shop_id']:
    print(train[x].value_counts().head())

# ????,????????
f,ax=plt.subplots(1,2,figsize=(14,6))
train['is_trade'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('trade positive-negative ration')
ax[0].set_ylabel('')
sns.countplot('is_trade',data=train,ax=ax[1])
ax[1].set_title('trade positive-negative appear times')
plt.show()

fig, axis1 = plt.subplots(1,1,figsize=(10,6))
item_num=pd.DataFrame({'item_id_num':train['item_id'].value_counts().values})
sns.countplot(x='item_id_num',data=item_num[item_num['item_id_num']<50])
axis1.set_xlabel('item appear times')
axis1.set_ylabel('the number of commodities that appear n times')
axis1.set_title('commodities distribution')


fig, axis1 = plt.subplots(1,1,figsize=(10,6))

item_value=pd.DataFrame(train.item_id.value_counts()).reset_index().head(20)
axis1.set_xlabel('item_id')
axis1.set_ylabel('appear times')
axis1.set_title('commodities that have top20 appear times')
y_pos = np.arange(len(item_value))

plt.bar(y_pos, item_value['item_id'], color=(0.2, 0.4, 0.6, 0.6))
pltz.xticks(y_pos, item_value['item_id'])
pltz.show()

fig, axis1 = plt.subplots(1, 1, figsize=(10, 6))
shop_num = pd.DataFrame({'shop_id_num': train['shop_id'].value_counts().values})
sns.countplot(x='shop_id_num', data=shop_num[shop_num['shop_id_num'] < 50])
axis1.set_xlabel('shop appear times')
axis1.set_ylabel('the number of shops that appear n times')
axis1.set_title('shop distribution')

fig, axis1 = plt.subplots(1, 1, figsize=(10, 6))

shop_value = pd.DataFrame(train.shop_id.value_counts()).reset_index().head(20)
axis1.set_xlabel('shop_id')
axis1.set_ylabel('appear times')
axis1.set_title('shops that have top20 appear times')
y_pos = np.arange(len(shop_value))

plt.bar(y_pos, shop_value['shop_id'], color=(0.2, 0.4, 0.6, 0.6))
pltz.xticks(y_pos, shop_value['shop_id'])
pltz.show()

for x in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
    print(train[x].value_counts().head(5))

plt.figure(figsize=(10,6))
plt.plot(train.groupby('user_id').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('user_id')
plt.ylabel('average is_trade')
plt.legend(loc=0)
print('There are {} users in train and {} in test'.format(len(train.user_id.unique()),len(test.user_id.unique())))
print('There are {} intersect user_id in train and test'.format(len(np.intersect1d(train.item_id.values,test.item_id.values))))
print('There are {} woman, {} man, {} family and {} missing value in train'.format(train.loc[train['user_gender_id']==0,:].shape[0],train.loc[train['user_gender_id']==1,:].shape[0],
                                                    train.loc[train['user_gender_id'] == 2, :].shape[0],train.loc[train['user_gender_id']==-1,:].shape[0]))
print('There are {} woman, {} man, {} family and {} missing value in test'.format(test.loc[test['user_gender_id']==0,:].shape[0],test.loc[test['user_gender_id']==1,:].shape[0],
                                                    test.loc[test['user_gender_id'] == 2,:].shape[0],test.loc[test['user_gender_id'] == -1,:].shape[0]))

plt.figure(figsize=(10,6))
plt.plot(train.groupby('user_gender_id').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('user_gender_id')
plt.ylabel('average is_trade')
plt.legend(loc=0)
plt.figure(figsize=(10,6))
plt.hist(train.user_gender_id.values,bins=100)
plt.xlabel('user_gender_id')
plt.ylabel('number of user')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('user_age_level').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('user_age_level')
plt.ylabel('average is_trade')
plt.xlim((1000,1010))
plt.legend(loc=0)
plt.figure(figsize=(10,6))
plt.hist(train.user_age_level.values,bins=3000)
plt.xlabel('user_age_level')
plt.ylabel('number of user')
plt.xlim((1000,1010))
print('There are {} miss value in user_age_level'.format(len(train.loc[train['user_age_level']==-1,:])))

plt.figure(figsize=(10,6))
plt.plot(train.groupby('user_occupation_id').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('user_occupation_id')
plt.ylabel('average is_trade')
plt.xlim((2000,2010))
plt.legend(loc=0)
plt.figure(figsize=(10,6))
plt.hist(train.user_occupation_id.values,bins=3000)
plt.xlim((2000,2010))
plt.xlabel('user_occupation_id')
plt.ylabel('number of user')
print('There are {} miss value in user_occupation_id'.format(len(train.loc[train['user_occupation_id']==-1,:])))
print('user_occupation_id conclude:{} in train'.format(train.user_occupation_id.unique()))
print('user_occupation_id conclude: {} in test'.format(test.user_occupation_id.unique()))

plt.figure(figsize=(10,6))
plt.plot(train.groupby('user_star_level').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('user_star_level')
plt.ylabel('average is_trade')
plt.xlim((3000,3010))
plt.legend(loc=0)
plt.figure(figsize=(10,6))
plt.hist(train.user_star_level.values,bins=3000)
plt.xlabel('user_star_level')
plt.ylabel('number of user')
plt.xlim((3000,3010))
print('There are {} miss value in user_star_level'.format(len(train.loc[train['user_star_level']==-1,:])))
plt.show()

f,ax=plt.subplots(2,2,figsize=(12,12))
train['user_gender_id'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0][0],shadow=True,colors=Blues_9.hex_colors)
ax[0][0].set_title('user_gender_id')

sns.countplot('user_age_level',data=train,ax=ax[0][1])
ax[0][1].set_title('user_age_level')

sns.countplot('user_occupation_id',data=train,ax=ax[1][0])
ax[1][0].set_title('user_occupation_id')

train['user_star_level'].value_counts().sort_index().plot.pie(autopct='%1.1f%%',ax=ax[1][1],shadow=True,colors=PuRd_5.hex_colors)
ax[1][1].set_title('user_star_level')

plt.show()

for x in ['shop_review_num_level','shop_star_level']:
    print(train[x].value_counts())

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_id').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlabel('shop_id')
plt.ylabel('average is_trade')
print('There are {} shop_id in test'.format(len(test.shop_id.unique())))
print('There are {} shop_id intersection in train and test'.format(len(np.intersect1d(train.shop_id.values,test.shop_id.values))))

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_review_num_level').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlabel('shop_review_num_level')
plt.ylabel('average is_trade')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_review_positive_rate').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlabel('shop_review_positive_rate')
plt.ylabel('average is_trade')
plt.xlim((0.8,1))
print('There are {} miss shop_review_positive_rate'.format(len(train.loc[train['shop_review_positive_rate']==-1,:])))
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_star_level').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlim((4999,5020))
plt.xlabel('shop_star_level')
plt.ylabel('average is_trade')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_score_service').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlim((0.8,1))
plt.xlabel('shop_score_service')
plt.ylabel('average is_trade')
print('There are {} miss shop_score_service'.format(len(train.loc[train['shop_score_service']==-1,:])))
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_score_delivery').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlim((0.8,1))
plt.xlabel('shop_score_delivery')
plt.ylabel('average is_trade')
print('There are {} miss shop_score_delivery'.format(len(train.loc[train['shop_score_delivery']==-1,:])))
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('shop_score_description').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlim((0.8,1))
plt.xlabel('shop_score_description')
plt.ylabel('average is_trade')
print('There are {} miss shop_score_description'.format(len(train.loc[train['shop_score_description']==-1,:])))
plt.show()

f,ax=plt.subplots(2,1,figsize=(12,12))
sns.countplot('shop_review_num_level',data=train,ax=ax[0])
ax[0].set_title('shop review number level distribution')

sns.countplot('shop_star_level',data=train,ax=ax[1])
ax[1].set_title('shop star number level distribution')

plt.style.use('ggplot')
f,ax=plt.subplots(4,2,figsize=(14,18))
plt.tight_layout(5)
sns.boxplot(y=train['shop_review_positive_rate'][train['shop_review_positive_rate']!=-1],ax=ax[0][0])
sns.distplot(train['shop_review_positive_rate'][train['shop_review_positive_rate']>0.98],ax=ax[0][1])
ax[0][1].set_title('shop review positive rate')


sns.boxplot(y=train['shop_score_service'][train['shop_score_service']!=-1],ax=ax[1][0])
sns.distplot(train['shop_score_service'][train['shop_score_service']>0.9],ax=ax[1][1])
ax[1][1].set_title('shop score service score')


sns.boxplot(y=train['shop_score_delivery'][train['shop_score_delivery']!=-1],ax=ax[2][0])
sns.distplot(train['shop_score_delivery'][train['shop_score_delivery']>0.9],ax=ax[2][1])
ax[2][1].set_title('shop score delivery score')


sns.boxplot(y=train['shop_score_description'][train['shop_score_description']!=-1],ax=ax[3][0])
sns.distplot(train['shop_score_description'][train['shop_score_description']>0.9],ax=ax[3][1])
ax[3][1].set_title('shop description score')

for x in ['item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']:
    print(train[x].value_counts())

plt.figure(figsize=(10,6))
plt.hist(train['item_id'].values, bins=100)
plt.xlabel('item_id')
plt.ylabel('number if item')
plt.show()
print('There are {} items in train and {} in test'.format(len(train.item_id.unique()),len(test.item_id.unique())))
print('There are {} intersect in train and test'.format(len(np.intersect1d(train.item_id.values,test.item_id.values))))

train_item_category_list_1=pd.DataFrame([int(i.split(';')[0]) for i in train.item_category_list])
train_item_category_list_2=pd.DataFrame([int(i.split(';')[1]) for i in train.item_category_list])
train_item_category_list_3=pd.DataFrame([int(i.split(';')[2]) for i in train.item_category_list if len(i.split(';'))==3])
test_item_category_list_1=pd.DataFrame([int(i.split(';')[0]) for i in test.item_category_list])
test_item_category_list_2=pd.DataFrame([int(i.split(';')[1]) for i in test.item_category_list])
test_item_category_list_3=pd.DataFrame([int(i.split(';')[2]) for i in test.item_category_list if len(i.split(';'))==3])

print('There are {} item_category_list_1 categories'.format(len(train_item_category_list_1[0].unique())))
print('There are {} item_category_list_2 categories'.format(len(train_item_category_list_2[0].unique())))
print('There are {} item_category_list_3 categories'.format(len(train_item_category_list_3[0].unique())))
print('There are {} item_category_list_3 in train and {} in test'.format(len(train_item_category_list_3[0]),len(test_item_category_list_3[0])))

train_item_property_list=pd.DataFrame([int(len(i.split(';'))) for i in train.item_property_list])
test_item_property_list=pd.DataFrame([int(len(i.split(';'))) for i in test.item_property_list])
plt.figure(figsize=(10,6))
plt.plot(train_item_property_list[0], '.-', label='train')
plt.plot(test_item_property_list[0], '.-', label='test')
plt.title('Number of item property in train and test .')
plt.legend(loc=0)
plt.ylabel('number of property')
plt.show()

# train_item_property_list_2=pd.DataFrame([int(i.split(';')[1]) for i in train.item_property_list])
# train_item_property_list_3=pd.DataFrame([int(i.split(';')[2]) for i in train.item_property_list if len(i.split(';'))==3])
# test_item_property_list_1=pd.DataFrame([int(i.split(';')[0]) for i in test.item_property_list])
# test_item_property_list_2=pd.DataFrame([int(i.split(';')[1]) for i in test.item_property_list])
# test_item_property_list_3=pd.DataFrame([int(i.split(';')[2]) for i in test.item_property_list if len(i.split(';'))==3])
plt.figure(figsize=(10,6))
plt.plot(train.groupby('item_brand_id').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('item_brand_id')
plt.ylabel('average is_trade')
plt.legend(loc=0)
print('There are {} item_brand_id'.format(len(train.item_brand_id.unique())))

plt.figure(figsize=(10,6))
plt.plot(train.groupby('item_city_id').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('item_city_id')
plt.ylabel('average is_trade')
plt.legend(loc=0)
print('There are {} item_city_id'.format(len(train.item_city_id.unique())))
#list(train.item_city_id.values).count(train.groupby('item_city_id').mean()['is_trade'].index[53])

plt.figure(figsize=(10,6))
plt.plot(train.groupby('item_price_level').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('item_price_level')
plt.ylabel('average is_trade')
plt.legend(loc=0)
plt.show()

plt.figure(figsize=(10,6))
plt.hist(train.item_price_level.values,bins=100)
plt.xlabel('item_price_level')
plt.ylabel('number of item_price_level')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('item_sales_level').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('item_sales_level')
plt.ylabel('average is_trade')
plt.legend(loc=0)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('item_collected_level').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('item_collected_level')
plt.ylabel('average is_trade')
plt.legend(loc=0)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(train.groupby('item_pv_level').mean()['is_trade'], 'o-', label='is_trade rate')
plt.xlabel('item_pv_level')
plt.ylabel('average is_trade')
plt.legend(loc=0)
plt.show()

f,ax=plt.subplots(2,2,figsize=(20,20))

item_brand_id_num=pd.DataFrame({'brand_id_num':train['item_brand_id'].value_counts()}).reset_index()
brand_value=pd.DataFrame({'brand_id_num':item_brand_id_num['brand_id_num'][item_brand_id_num['brand_id_num']<4000].sum()},index=[0])
brand_value['index']='below_4000'
brand_value=pd.concat([brand_value,item_brand_id_num[item_brand_id_num['brand_id_num']>=4000]])
pd.Series(data=brand_value.set_index('index')['brand_id_num']).plot.pie(autopct='%1.1f%%',ax=ax[0][0],shadow=True,colors=Blues_9.hex_colors)
ax[0][0].set_title('item_brand_id')
ax[0][0].legend(fontsize=7.5)
#sns.countplot('item_city_id',data=train,ax=ax[0][1])


item_city_id_num=pd.DataFrame({'city_id_num':train['item_brand_id'].value_counts()}).reset_index()
city_value=pd.DataFrame({'city_id_num':item_city_id_num['city_id_num'][item_city_id_num['city_id_num']<3000].sum()},index=[0])
city_value['index']='below_3000'
city_value=pd.concat([city_value,item_city_id_num[item_brand_id_num['brand_id_num']>=3000]])
pd.Series(data=city_value.set_index('index')['city_id_num']).plot.pie(autopct='%1.1f%%',ax=ax[0][1],shadow=True,colors=Blues_9.hex_colors)
ax[0][1].set_title('item_city_id')

sns.countplot('item_sales_level',data=train,ax=ax[1][0])
ax[1][0].set_title('item_sales_level')

train['item_collected_level'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1][1],shadow=True,colors=PuRd_5.hex_colors)
ax[1][1].set_title('item_collected_level')
ax[1][1].legend(fontsize=7.5)
plt.show()

plt.figure(figsize=(10,6)) #
plt.hist(train['is_trade'].values, bins=100)
plt.xlabel('trade information')
plt.ylabel('number of trade')
plt.show()

for x in ['context_id','context_timestamp','context_page_id','predict_category_property']:
    print(train[x].value_counts())

print('There are {} context_id intersection in train and test'.format(len(np.intersect1d(train.context_id.values,test.context_id.values))))
plt.figure(figsize=(10,6))
plt.plot(train.groupby('context_timestamp').mean()['is_trade'], '.', label='is_trade rate')
plt.xlabel('contaxt_timestamp')
plt.ylabel('average is_trade')
plt.figure(figsize=(10,6))
plt.plot(train.groupby('context_page_id').mean()['is_trade'], '.-', label='is_trade rate')
plt.xlabel('context_page_id')
plt.ylabel('average is_trade')
print('context_page_id start is {} and end is {} in train'.format(train.context_page_id.min(),train.context_page_id.max()))
print('context_page_id start is {} and end is {} in test'.format(test.context_page_id.min(),test.context_page_id.max()))

train['num_predict_category_property']=[sum(map(lambda i:0 if i[-2:]=='-1' else 1,re.split(',|;',j))) for j in train.predict_category_property]
plt.figure(figsize=(10,6))
plt.plot(train.groupby('num_predict_category_property').mean()['is_trade'],'.-',label='is_trade rate')
plt.xlabel('num_predict_category_property')
plt.ylabel('average is_trade')


# # round1
# train_path_people_1='./round1_train_cut_by_people.txt'
# train_path_type_1='./round1_train_cut_by_type.txt'
# test_path_a_1='./round1_ijcai_18_test_a_20180301.txt'
# test_path_b_1='./round1_ijcai_18_test_b_20180418.txt'
# # round2
# train_path_type_2='./round2_train_cut_by_type.txt'
# test_path_a_2='./round2_test_a.txt'
# test_path_b_2='./round2_test_b.txt'
# dataset_cut
train_path = './090.txt'
test_path = './test_no_label.txt'
train=pd.read_table(train_path,delimiter=' ')
test=pd.read_table(test_path,delimiter=' ')

# ?????????inplace=True?????
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

# train_ad_data ???
trainLen = len(train)
# ??????????
trainlabel = train['is_trade']
# ??????ID
testInstanceID = test['instance_id']

# ????????
serialize_constant = {}
serialize_constant['trainLen'] = trainLen
serialize_constant['trainlabel'] = trainlabel
serialize_constant['testInstanceID'] = testInstanceID

# ??pickle???
filename = './serialize_constant'

with open(filename, 'wb') as f:
    pickle.dump(serialize_constant, f)

# with open(filename, 'rb') as f:
#     serialize_constant = pickle.load(f)
#     trainLen = serialize_constant['trainLen']
#     trainlabel = serialize_constant['trainlabel']
#     testInstanceID = serialize_constant['testInstanceID']


# ?????????????
key = list(test)  # test ???
mergeData = pd.concat([train, test], keys=key)
# ??????
mergeData = mergeData.reset_index(drop=True)


# ?timestamp???datetime?%Y-%m-%d %H:%M:%S?
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt  # str


# ???datetime64[ns]
mergeData['time'] = pd.to_datetime(mergeData.context_timestamp.apply(timestamp_datetime))

# ????18-24????25?
# ????8/31-9/6???9/7?????9/7?????

mergeData['day'] = mergeData.time.dt.day
mergeData['hour'] = mergeData.time.dt.hour
mergeData['minute'] = mergeData.time.dt.minute

# ID?0????
lbl = preprocessing.LabelEncoder()
for col in ['item_id', 'item_brand_id', 'item_city_id', 'shop_id', 'user_id']:
    mergeData[col] = lbl.fit_transform(mergeData[col])

# ???csv??
mergeData.to_csv('./mergeData.csv', sep=' ')

data = pd.read_csv('./mergeData.csv', sep=' ')
print(data.shape)
print(data['instance_id'].nunique())

# ????????

# item_category_list???2-3?;???id???????????2???
item_category_list_2 = pd.DataFrame([int(i.split(';')[1]) for i in data.item_category_list])

data['item_category_list_2'] = item_category_list_2
data.head()

'''
??????????
'''

# ???????????????????????????
user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
columns={0: 'user_query_day_hour'})
data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

# ??????????????
item_id_frequence = data.groupby([ 'item_id']).size().reset_index().rename(columns={0: 'item_id_frequence'})
item_id_frequence=item_id_frequence/(data.shape[0])
data = pd.merge(data, item_id_frequence, 'left', on=['item_id'])

# ????????????????????
num_user_minute = data.groupby(['user_id','day','minute']).size().reset_index().rename(columns = {0:'num_user_day_minute'})
data = pd.merge(data, num_user_minute,'left',on = ['user_id','day','minute'])

# ???????????????????
day_user_item_id = data.groupby(['day', 'user_id', 'item_id']).size().reset_index().rename(
columns={0: 'day_user_item_id'})
data = pd.merge(data, day_user_item_id, 'left', on=['day', 'user_id', 'item_id'])

# ?????????????????????????
day_hour_minute_user_item_id = data.groupby(
['day', 'hour', 'minute', 'user_id', 'item_id']).size().reset_index().rename(
columns={0: 'day_hour_minute_user_item_id'})
data = pd.merge(data, day_hour_minute_user_item_id, 'left', on=['day', 'hour', 'minute', 'user_id', 'item_id'])

# ??????????????????????
number_day_hour_item_id = data.groupby(['day', 'hour', 'item_id']).size().reset_index().rename(
columns={0: 'number_day_hour_item_id'})
data = pd.merge(data, number_day_hour_item_id, 'left', on=['day', 'hour', 'item_id'])

# ??????????????????
item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(columns={0: 'item_user_id'})
data = pd.merge(data, item_user_id, 'left', on=['item_id', 'user_id'])

data.head()

'''
?????????
'''

# ?????????????????????
item_category_city_id = data.groupby(['item_category_list', 'item_city_id']).size().reset_index().rename(
columns={0: 'item_category_city_id'})
data = pd.merge(data, item_category_city_id, 'left', on=['item_category_list', 'item_city_id'])

# ??????????????????????????????????
item_category_sales_level = data.groupby(
['item_category_list', 'item_sales_level']).size().reset_index().rename(
columns={0: 'item_category_sales_level'})
data = pd.merge(data, item_category_sales_level, 'left', on=['item_category_list', 'item_sales_level'])

# ??????????????????????
item_category_price_level = data.groupby(
['item_category_list', 'item_price_level']).size().reset_index().rename(
columns={0: 'item_category_price_level'})
data = pd.merge(data, item_category_price_level, 'left', on=['item_category_list', 'item_price_level'])

# ????????????????????
item_ID_sales_level = data.groupby(['item_id', 'item_sales_level']).size().reset_index().rename(
columns={0: 'item_ID_sales_level'})
data = pd.merge(data, item_ID_sales_level, 'left', on=['item_id', 'item_sales_level'])

# ????????????????????
item_ID_collected_level = data.groupby(['item_id', 'item_collected_level']).size().reset_index().rename(
columns={0: 'item_ID_collected_level'})
data = pd.merge(data, item_ID_collected_level, 'left', on=['item_id', 'item_collected_level'])

data.head()

'''
??????????
'''

# ??????????
number_user_id = data.groupby(['user_id']).size().reset_index().rename(columns={0: 'number_user_id'})
data = pd.merge(data, number_user_id, 'left', on=['user_id'])

# ??????????
number_shop_id = data.groupby(['shop_id']).size().reset_index().rename(columns={0: 'number_shop_id'})
data = pd.merge(data, number_shop_id, 'left', on=['shop_id'])

lbl = preprocessing.LabelEncoder()

# ???????????????predict_category_property0..4?????????0????????????
for i in range(5):
    data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

# ????????????item_category_list1..2?????????0????????????
for i in range(1, 3):
    data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

# ????????????item_property_list0..9?????????0????????????
for i in range(10):
    data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

# data['context_page0'] = data['context_page_id'].apply(
#   lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

data.head()

'''
?????????????????
'''

# ?????0??
data['gender0'] = data['user_gender_id'].apply(lambda x: x + 1 if x == -1 else x)

# ?????1003??????1000-1007
# print(data['user_age_level'].value_counts())
data['age0'] = data['user_age_level'].apply(lambda x: 1003 if x == -1  else x)

# ?????2005??????2002-2005
# print(data['user_occupation_id'].value_counts())
data['occupation0'] = data['user_occupation_id'].apply(lambda x: 2005 if x == -1  else x)

# ?????3006??????3000-3010
# print(data['user_star_level'].value_counts())
data['star0'] = data['user_star_level'].apply(lambda x: 3006 if x == -1 else x)

'''
?????????
'''

# ??????????????????
number_item_user_id = data.groupby(['item_id', 'user_id']).size().reset_index().rename(
    columns={0: 'number_item_user_id'})
data = pd.merge(data, number_item_user_id, 'left', on=['item_id', 'user_id'])

# ???????????????????????
number_item_brand_positive_rate = data.groupby(
    ['item_brand_id', 'shop_review_positive_rate']).size().reset_index().rename(
    columns={0: 'number_item_brand_positive_rate'})
data = pd.merge(data, number_item_brand_positive_rate, 'left',
                on=['item_brand_id', 'shop_review_positive_rate'])

# ???????????????????????
number_item_brand_shop_star = data.groupby(['item_brand_id', 'shop_star_level']).size().reset_index().rename(
    columns={0: 'number_item_brand_shop_star'})
data = pd.merge(data, number_item_brand_shop_star, 'left', on=['item_brand_id', 'shop_star_level'])

# ???????????????????????????
number_item_city_pv_level = data.groupby(['item_city_id', 'item_pv_level']).size().reset_index().rename(
    columns={0: 'number_item_city_pv_level'})
data = pd.merge(data, number_item_city_pv_level, 'left', on=['item_city_id', 'item_pv_level'])

# ??????????????????????
number_item_city_user_id = data.groupby(['item_city_id', 'user_id']).size().reset_index().rename(
    columns={0: 'number_item_city_user_id'})
data = pd.merge(data, number_item_city_user_id, 'left', on=['item_city_id', 'user_id'])

# ????????????????????????????
number_item_price_sales_level = data.groupby(
    ['item_price_level', 'item_sales_level']).size().reset_index().rename(
    columns={0: 'number_item_price_sales_level'})
data = pd.merge(data, number_item_price_sales_level, 'left', on=['item_price_level', 'item_sales_level'])

# ??????????????????????????????
number_predict_category_sales_level = data.groupby(
    ['predict_category_property', 'item_sales_level']).size().reset_index().rename(
    columns={0: 'number_predict_category_sales_level'})
data = pd.merge(data, number_predict_category_sales_level, 'left',
                on=['predict_category_property', 'item_sales_level'])

# ????????????????????????????
number_collected_shop_id = data.groupby(['item_collected_level', 'shop_id']).size().reset_index().rename(
    columns={0: 'number_collected_shop_id'})
data = pd.merge(data, number_collected_shop_id, 'left', on=['item_collected_level', 'shop_id'])

# ????????????????????????????
for i in range(3):
    data['category_%d' % (i)] = data['item_category_list'].apply(
        lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")

# ????????????????????????????
for i in range(3):
    data['property_%d' % (i)] = data['item_property_list'].apply(
        lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")

# ????????????????????????????????????
for i in range(3):
    data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
        lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " ")

# ?????????????????????????????????????????
# nunique???????
for i in ['item_id', 'shop_id', 'day', 'context_page_id']:
    temp = data.groupby('user_id').nunique()[i].reset_index().rename(columns={i: 'number_' + i + '_query_user'})
    data = pd.merge(data, temp, 'left', on='user_id')

print(data.shape)
print(data['instance_id'].nunique())

data.head()

# ????
basic_data = data[['instance_id']]
ad_information = data[
        ['item_id', 'item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level','item_property_list',
         'item_sales_level', 'item_collected_level', 'item_pv_level']]
user_information = data[
        ['user_id', 'user_age_level', 'user_star_level', 'user_occupation_id','user_gender_id']]
text_information = data[['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']]
shop_information = data[
        ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
         'shop_score_delivery', 'shop_score_description']]
external_information = data[
        ['time', 'day', 'hour', 'minute', 'user_query_day', 'user_query_day_hour', 'day_user_item_id', \
         'day_hour_minute_user_item_id',
         'number_day_hour_item_id', 'number_user_id', 'number_shop_id', \
         'item_category_list_2', 'item_user_id', 'item_category_city_id', 'item_category_sales_level', \
         'item_ID_sales_level', 'item_ID_collected_level', 'item_category_price_level', \
         'predict_category_property0', 'predict_category_property1', 'predict_category_property2', \
         'predict_category_property3', 'predict_category_property4', 'item_category_list1', \
         'item_category_list2', 'item_property_list0', 'item_property_list1', 'item_property_list2', \
         'item_property_list3', 'item_property_list4', 'item_property_list5', 'item_property_list6', \
         'item_property_list7', 'item_property_list8', 'item_property_list9', 'gender0', 'age0', \
         'occupation0', 'star0', 'number_item_brand_positive_rate', 'number_item_brand_shop_star', \
         'number_item_city_pv_level', 'number_item_city_user_id', 'number_item_price_sales_level', \
         'number_predict_category_sales_level', 'number_collected_shop_id'# ,'shop_score_delivery_round','number_item_id_query_user' ,'number_shop_id_query_user','number_day_query_user' ,'number_context_page_id_query_user'
         ]]

# ???????????
result = pd.concat(
    [basic_data, ad_information, user_information, text_information, shop_information, external_information],
    axis=1)

print(result.shape)
print(result['instance_id'].nunique())

result.head()

'''
?????????????????
'''

# ItemCVR = concatDayCVR()
# result = pd.merge(result, ItemCVR, on=['instance_id','item_id','user_id'], how = 'left')

'''
I saved Bayes_smooth result for save time, you can run itemIDBayesSmooth.py, userIDBayesSmooth.py etc to get the result. 
'''
# Item_Bayes = np.load('../../datasets/Item_Bayes.npy')
# # Brand_Bayes=np.load('../../datasets/Brand_id_Bayes.npy')
# Shop_Bayes = np.load('../../datasets/Shop_Bayes.npy')
# # Hour_Bayes = np.load('../../datasets/Item_pv_levelBayesPH.npy')
# User_Bayes = np.load('../../datasets/UserBayesPH.npy')
# result['Item_Bayes'] = Item_Bayes
# result['Brand_Bayes'] = Brand_Bayes
# result['Shop_Bayes'] = Shop_Bayes
# result['User_Bayes'] = concatUserDayCVR()
# result['Hour_Bayes'] = Hour_Bayes
# result['itemCVR'] = concatItemDayCVR()

# result = zuhe(result)
# result = item(result)
# result = user(result)
# result = user_item(result)
# result = user_shop(result)
# result = shop_item(result)
# result = self.zuhe_feature(result)

# ?????????
# 1??????????????????????????
# 2???????????????????

# ?????????????????????????????????????????????
# ????????????????????
for d in range(18, 26):
    # ?????????
    df1 = result[result['day'] == d]

    # df.rank(method='min')??????????dataframe??????????????
    # ????1???
    # ???????groupby?????????????????
    rnColumn_user = df1.groupby('user_id').rank(method='min')
    rnColumn_user_item = df1.groupby(['user_id', 'item_id']).rank(method='min')
    rnColumn_user_shop = df1.groupby(['user_id', 'shop_id']).rank(method='min')

    # ?????????????????????????????
    df1['user_id_order'] = rnColumn_user['context_timestamp']
    df1['user_item_id_order'] = rnColumn_user_item['context_timestamp']
    df1['user_shop_id_order'] = rnColumn_user_shop['context_timestamp']

    # ?????????
    df2 = df1[['user_id', 'instance_id', 'item_id', 'user_id_order', 'user_item_id_order', 'user_shop_id_order']]
    if d == 18:
        Df = df2
    else:
        Df = pd.concat([Df, df2])

Df.drop_duplicates(inplace=True)

result = pd.merge(result, Df, on=['user_id', 'instance_id', 'item_id'], how='left')

print(result.shape)
print(result['instance_id'].nunique())
df1 = result.groupby(["instance_id"]).size()
col = df1[df1 > 1].reset_index()[["instance_id"]]
ttt = pd.merge(col, result, on=["instance_id"])
print(ttt)

# ???????
filename = './serialize_constant'

with open(filename, 'rb') as f:
    serialize_constant = pickle.load(f)
    trainlabel = serialize_constant['trainlabel']
result['is_trade'] = trainlabel

print(result.shape)
print(result['instance_id'].nunique())

result.head()

# ??????????n????????????n-1?????????
for d in range(18, 26):
    df1 = result[result['day'] == d - 1]  # ???
    df2 = result[result['day'] == d]  # ??

    df_cvr = result[(result['day'] == d - 1) & (result['is_trade'] == 1)]  # ?????????

    # ????????????????????{column -> {index -> value}}???
    user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
    # item_trade_cnt = df1.groupby(['item_id','shop_id','is_trade']).count()['instance_id'].to_dict()
    user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
    item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
    shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
    item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
    user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()

    # ????????n?????????n-1??????
    df2['item_cvr_cnt1'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
    df2['user_cvr_cnt1'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))
    df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
    df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
    df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
    # tuple()????axis=1??????
    df2['user_item_cnt1'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)

    # ?????????
    df2 = df2[['user_item_cnt1', 'user_cnt1', \
               # 'item_cnt1', 'shop_cnt1',\
               'item_cvr_cnt1', 'user_cvr_cnt1', \
               'item_id', 'user_id', 'instance_id']]
    if d == 18:
        Df2 = df2
    else:
        Df2 = pd.concat([df2, Df2])

Df2.drop_duplicates(inplace=True)

result = pd.merge(result, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

print(result.shape)
print(result['instance_id'].nunique())

result.head()

# ??????????n????????????0..n-1?????????
for d in range(18, 26):
    df1 = result[result['day'] < d] # ?????
    df2 = result[result['day'] == d] # ??

    df_cvr = result[(result['day'] < d) & (result['is_trade'] == 1)] # ???????????

    # ????????????????????{column -> {index -> value}}???
    user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
    user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
    item_cvr_cnt = df_cvr.groupby(by='item_id').count()['instance_id'].to_dict()
    user_cvr_cnt = df_cvr.groupby(by='user_id').count()['instance_id'].to_dict()

    # ????????n?????????0..n-1??????
    df2['item_cvr_cntx'] = df2['item_id'].apply(lambda x: item_cvr_cnt.get(x, 0))
    df2['user_cvr_cntx'] = df2['user_id'].apply(lambda x: user_cvr_cnt.get(x, 0))
    df2['user_item_cntx'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1) # tuple()????axis=1??????
    df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))

    # ?????????
    df2 = df2[['user_item_cntx', 'user_cntx',
       'item_cvr_cntx', 'user_cvr_cntx', \
       'item_id', 'user_id', 'instance_id']]

    if d == 18:
        Df2 = df2
    else:
        Df2 = pd.concat([df2, Df2])

Df2.drop_duplicates(inplace=True)

result = pd.merge(result, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

print(result.shape)
print(result['instance_id'].nunique())

result.head()

# ????????????
#
# # ??????????n????????????n-2..n-1?????????
# for d in range(18, 26):
#     #print("%d: \n" % d)
#     df1 = result[(result['day'] >= d - 2) & (result['day'] < d)] # n-2?n-1?
#     df2 = result[result['day'] == d] # ??

#     # ????????????????????{column -> {index -> value}}???
#     user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
#     item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
#     shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
#     user_item_cnt = df1.groupby(['item_id', 'user_id']).count()['instance_id'].to_dict()
#     #print("1\n")

#     # ????????n?????????0..n-1??????
#     df2['user_item_cnt2'] = df2[['item_id', 'user_id']].apply(lambda x: user_item_cnt.get(tuple(x), 0), axis=1)
#     df2['user_cnt2'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
#     df2['item_cnt2'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
#     df2['shop_cnt2'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
#     #print("2\n")

#     # ?????????
#     df2 = df2[['user_item_cnt2', 'user_cnt2', 'item_cnt2', 'shop_cnt2', \
#        'item_id', 'user_id', 'instance_id']]

#     if d == 18:
#         Df2 = df2
#     else:
#         Df2 = pd.concat([df2, Df2])
#     #print("3\n")

# # result.to_csv('../../produce/result.csv')
# # Df2.to_csv('../../produce/Df2.csv')
# result = pd.merge(result, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

# ??????????????

# import pandas as pd
# result = pd.read_csv('../../produce/result.csv', sep=',')
# Df2 = pd.read_csv('../../produce/Df2.csv', sep=',')
# def merge_size(left_frame, right_frame, group_by, how='left'):
#     left_groups = left_frame.groupby(group_by).size()
#     right_groups = right_frame.groupby(group_by).size()
#     left_keys = set(left_groups.index)
#     right_keys = set(right_groups.index)
#     intersection = right_keys & left_keys
#     left_diff = left_keys - intersection
#     right_diff = right_keys - intersection

#     left_nan = len(left_frame[left_frame[group_by] != left_frame[group_by]])
#     right_nan = len(right_frame[right_frame[group_by] != right_frame[group_by]])
#     left_nan = 1 if left_nan == 0 and right_nan != 0 else left_nan
#     right_nan = 1 if right_nan == 0 and left_nan != 0 else right_nan

#     sizes = [(left_groups[group_name] * right_groups[group_name]) for group_name in intersection]
#     sizes += [left_nan * right_nan]

#     left_size = [left_groups[group_name] for group_name in left_diff]
#     right_size = [right_groups[group_name] for group_name in right_diff]
#     if how == 'inner':
#         return sum(sizes)
#     elif how == 'left':
#         return sum(sizes + left_size)
#     elif how == 'right':
#         return sum(sizes + right_size)
#     return sum(sizes + left_size + right_size)
# group_by = ['instance_id', 'item_id', 'user_id']
# print(min([merge_size(result, Df2, group_by, how='left') for label in group_by]))
# # result = pd.merge(result, Df2, on=['instance_id', 'item_id', 'user_id'], how='left')

'''
?????????????
'''

# # ?????????????????????????

# train_origin = result
# train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'item_id']]

# # ?????????????
# train1 = train1.sort_values(['user_id', 'item_id', 'context_timestamp'], ascending=[1, 1, 1])

# # ????????????????????????
# rnColumn = train1.groupby(['user_id', 'item_id']).rank(method='min')
# train1['rnnn'] = rnColumn['context_timestamp']
# # ??-1??????????????????????
# train1['rnnn_1'] = rnColumn['context_timestamp'] - 1

# # train2?train1????????????how=?left?????????????
# # left_on?right_on????????????????
# # ???print(train2)??train2?????????
# # ??user_id?item_id???????????????rnnn_1?rnnn???????
# # ????user_id?item_id????????????????_x?_y??
# # train2????????????????????2???????????????????????????NaN
# train2 = train1.merge(train1, how='left', left_on=['user_id', 'item_id', 'rnnn_1'],
#       right_on=['user_id', 'item_id', 'rnnn'])

# print('??',datetime.datetime.now().strftime("%H:%M:%S"))

# ????????????????????
# train2['time_redc_user_item'] = train2['context_timestamp_x'] - train2['context_timestamp_y']

# # ?-1??NaN??????int64
# train2 = train2.fillna(-1).astype('int64')

# # ?????
# train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
# train2 = train2.rename(columns={'context_timestamp_x': 'context_timestamp'})

# # ?????
# train2 = train2.drop([#'rnnn_x','rnnn_y','rnnn_1_x','rnnn_1_y',
#     'context_timestamp_y', 'instance_id_y'], axis=1)

# result = pd.merge(train_origin, train2, on=['instance_id', 'item_id', 'user_id', 'context_timestamp'], how='left')
# print('??',datetime.datetime.now().strftime("%H:%M:%S"))

'''
?????????????
'''

# # ????????????????????????????????

# # ??????????????????????
# train_origin = result
# train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'shop_id', 'item_id']]

# # ?????????????
# train1 = train1.sort_values(['user_id', 'shop_id', 'context_timestamp'], ascending=[1, 1, 1])

# # ????????????????????
# rnColumn = train1.groupby(['user_id', 'shop_id']).rank(method='min')
# train1['rnn'] = rnColumn['context_timestamp']
# # ??-1??????????????????
# train1['rnn_1'] = rnColumn['context_timestamp'] - 1

# # train2?train1????????????how=?left?????????????
# # left_on?right_on????????????????
# # ???print(train2)??train2?????????
# # ??user_id?shop_id???????????????rnn_1?rnn???????
# # ????user_id?shop_id????????????????_x?_y??
# # train2??????????????????2???????????????????????????NaN
# train2 = train1.merge(train1, how='left', left_on=['user_id', 'shop_id', 'rnn_1'],
#       right_on=['user_id', 'shop_id', 'rnn'])

# print('??',datetime.datetime.now().strftime("%H:%M:%S"))

# ????????????????
# train2['time_redc_user_shop'] = train2['context_timestamp_x'] - train2['context_timestamp_y']

# # ?-1??NaN??????int64
# train2 = train2.fillna(-1).astype('int64')

# # ?????
# train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
# train2 = train2.rename(columns={'item_id_x': 'item_id'})
# train2 = train2.rename(columns={'shop_id_x': 'shop_id'})
# train2 = train2.rename(columns={'context_timestamp_x': 'context_timestamp'})

# # ?????
# train2 = train2.drop([#'rnn_x','rnn_y','rnn_1_x','rnn_1_y',
#     'context_timestamp_y', 'instance_id_y'], axis=1)

# # ??
# result = pd.merge(train_origin, train2, on=['instance_id', 'item_id', 'user_id', 'context_timestamp'], how='left')
# print('??',datetime.datetime.now().strftime("%H:%M:%S"))

'''
?????????????
'''

# # ???????
# # 1??????????????????????
# # 2??????????????????????????????????
# # ??????????????????
# train_origin = result
# train1 = train_origin[['context_timestamp', 'user_id', 'instance_id', 'item_id']]

# # ?????????????
# train1 = train1.sort_values(['user_id', 'context_timestamp'], ascending=[1, 1])

# # ????????????????
# rnColumn = train1.groupby('user_id').rank(method='min')
# train1['rn'] = rnColumn['context_timestamp']
# # ??-1????????????????
# train1['rn_1'] = rnColumn['context_timestamp'] - 1

# # train2?train1????????????how=?left?????????????
# # left_on?right_on????????????????
# # ???print(train2)??train2?????????
# # ??user_id???????????????rn_1?rn???????
# # ????user_id????????????????_x?_y??
# # train2??????????????2???????????????????????????NaN
# train2 = train1.merge(train1, how='left', left_on=['user_id', 'rn_1'], right_on=['user_id', 'rn'])

# # ????????????????
# train2['time_redc'] = train2['context_timestamp_x'] - train2['context_timestamp_y']

# # ?-1??NaN??????int64
# train2 = train2.fillna(-1).astype('int64')

# # ?????
# train2 = train2.rename(columns={'instance_id_x': 'instance_id'})
# train2 = train2.rename(columns={'item_id_x': 'item_id'})
# train2 = train2.rename(columns={'context_timestamp_x': 'context_timestamp'})

# # ?????????????????????????
# user_cnt_max = train2.groupby(['user_id']).max()['rn_x'].reset_index().rename(columns={'rn_x': 'user_cnt_max'})
# train2 = pd.merge(train2,user_cnt_max,'left',on=['user_id'])
# # ???????????????
# train2['user_remain_cnt'] = train2['user_cnt_max'] - train2['rn_x']
# # inplace=True??????resultframe
# train2.drop(['user_cnt_max'],inplace=True,axis=1)

# # ??
# result = pd.merge(train_origin, train2, on=['instance_id', 'item_id', 'user_id'], how='left')

# ??????
'''
????????????????????????????????????????
'''
# ??????????????????
result['sales_div_pv'] = result.item_sales_level / (1 + result.item_pv_level)
# na_action='ignore'????x?NaN?????
result['sales_div_pv'] = result.sales_div_pv.map(lambda x: int(10 * x), na_action='ignore')

# ??????????????????
number_click_day = result.groupby(['day']).size().reset_index().rename(columns={0:'number_click_day'})
result = pd.merge(result,number_click_day,'left',on=['day'])

# ???????????????????
number_click_hour = result.groupby(['hour']).size().reset_index().rename(columns={0:'number_click_hour'})
result = pd.merge(result,number_click_hour,'left',on=['hour'])

# ????????????????????????????????????????????
# nunique???????
temp = result.groupby('item_id')['user_age_level'].nunique().reset_index().rename(columns={'user_age_level': 'number_' + 'user_age_level' + '_query_item'})
result = pd.merge(result, temp, 'left', on=['item_id'])

# ??????????????????????
# ???item_category_list??????????????item_category_list_1?????????????
number_category_item = result.groupby(['item_category_list_2','item_id']).size().reset_index().rename(columns={0:'number_category_item'})
result = pd.merge(result,number_category_item,'left',on=['item_category_list_2','item_id'])

# ???????????????????????????
number_category2 = result.groupby(['item_category_list_2']).size().reset_index().rename(columns={0:'number_category2'})
result = pd.merge(result,number_category2,'left',on=['item_category_list_2'])

# ???????????????????????????
result['prob_item_id_category2'] = result['number_category_item']/result['number_category2']

# ??number_category2?number_category_item????
result = result.drop(['number_category2','number_category_item'],axis=1)

# ??????????????????????
ave_price_category_item = result.groupby(['item_category_list_2','item_id']).mean()['item_price_level'].reset_index().rename(columns={'item_price_level':'ave_price_category_item'})
result = pd.merge(result,ave_price_category_item,'left',on=['item_category_list_2','item_id'])

# ?????????????????????????
ave_price_category = result.groupby(['item_category_list_2']).mean()['item_price_level'].reset_index().rename(columns={'item_price_level':'ave_price_category'})
result = pd.merge(result,ave_price_category,'left',on=['item_category_list_2'])

# ????????????????????????????
result['prob_item_price_to_ave_category2'] = result['item_price_level']/result['ave_price_category']

# ????????????????????????????
ave_sales_price_category_item = result.groupby(['item_category_list_2','item_id','item_price_level']).mean()['item_sales_level'].reset_index().rename(columns={'item_sales_level':'ave_sales_price_category_item'})
result = pd.merge(result,ave_sales_price_category_item,'left',on=['item_category_list_2','item_id','item_price_level'])

# ???????????????????????????
ave_sales_level_category = result.groupby(['item_category_list_2']).mean()['item_sales_level'].reset_index().rename(columns={'item_sales_level':'ave_sales_level_category'})
result = pd.merge(result,ave_sales_level_category,'left',on=['item_category_list_2'])

# ???????????????????????
result['prob_ave_category_sales_item_sales'] = result['item_sales_level']/result['ave_sales_level_category']

# ?????????????????????
max_price_category = result.groupby(['item_category_list_2'])['item_price_level'].max().reset_index().rename(columns={'item_price_level':'max_price_category'})
result = pd.merge(result,max_price_category,'left',on=['item_category_list_2'])

# ????????????????????????????????
result['is_max_price_category'] = result['item_price_level']/result['max_price_category']
result['is_max_price_category'] = result['is_max_price_category'].map(lambda x: int(x), na_action='ignore')

# ?????????????????????
min_price_category = result.groupby(['item_category_list_2'])['item_price_level'].min().reset_index().rename(columns={'item_price_level':'min_price_category'})
result = pd.merge(result,min_price_category,'left',on=['item_category_list_2'])

# ????????????????????????????????????
result['is_min_price_category'] = result['min_price_category']/result['item_price_level']
result['is_min_price_category'] = result['is_min_price_category'].map(lambda x: int(x), na_action='ignore')

# ??max_price_category?min_price_category????
result = result.drop(['max_price_category','min_price_category'],axis=1)

# ?????????????????????
max_sales_category = result.groupby(['item_category_list_2'])['item_sales_level'].max().reset_index().rename(columns={'item_sales_level':'max_sales_category'})
result = pd.merge(result,max_sales_category,'left',on=['item_category_list_2'])

# ????????????????????????????????
result['is_max_sales_category'] = result['item_sales_level']/result['max_sales_category']
result['is_max_sales_category'] = result['is_max_sales_category'].map(lambda x: int(x), na_action='ignore')

# ?????????????????????
min_sales_category = result.groupby(['item_category_list_2'])['item_sales_level'].min().reset_index().rename(columns={'item_sales_level':'min_sales_category'})
result = pd.merge(result,min_sales_category,'left',on=['item_category_list_2'])

# ????????????????????????????????????
result['is_min_sales_category'] = result['min_sales_category']/result['item_sales_level']
result['is_min_sales_category'] = result['is_min_sales_category'].map(lambda x: int(x), na_action='ignore')

# ??max_sales_category?min_sales_category????
result = result.drop(['max_sales_category', 'min_sales_category'], axis=1)

'''
????????????????????????
'''
# # ??????????????????
# # ???rn_x???????????
# max_cnt_user_id = result.groupby(['user_id'])['rn_x'].max().reset_index().rename(columns={'rn_x':'max_cnt_user_id'})
# result = pd.merge(result,max_cnt_user_id,'left',on=['user_id'])

# # ???????????????????????????
# result['is_max_cnt_user_id'] = result['rn_x']/result['max_cnt_user_id']
# result['is_max_cnt_user_id'] = result['is_max_cnt_user_id'].map(lambda x: int(x), na_action='ignore')

# # ???????1
# min_cnt_user_id = result.groupby(['user_id'])['rn_x'].min().reset_index().rename(columns={'rn_x':'min_cnt_user_id'})
# result = pd.merge(result,min_cnt_user_id,'left',on=['user_id'])

# # ?????1 / ???????????
# result['is_min_cnt_user_id'] = result['min_cnt_user_id']/result['rn_x']
# result['is_min_cnt_user_id'] = result['is_min_cnt_user_id'].map(lambda x: int(x), na_action='ignore')

# # ??max_cnt_user_id?min_cnt_user_id????
# result = result.drop(['max_cnt_user_id', 'min_cnt_user_id'], axis=1)

# ????????? - ????
result['sales_minus_collected'] = result['item_sales_level'] - result['item_collected_level']

print(result.shape)
print(result['instance_id'].nunique())

result.head()

# ?????
result = result.drop(
    ['item_category_list', 'item_property_list', 'predict_category_property', 'time']
    #,'instance_id']
    , axis=1)

# result.drop_duplicates(inplace=True)

# ??
result.to_csv('./featureData.csv', sep=' ')

datapath = './featureData.csv'
data = pd.read_csv(datapath ,sep=' ')

filename = './serialize_constant'

with open(filename, 'rb') as f:
    serialize_constant = pickle.load(f)
    trainLen = serialize_constant['trainLen']
    trainlabel = serialize_constant['trainlabel']
    testInstanceID = serialize_constant['testInstanceID']

data = data.iloc[0 : trainLen, : ]
target = trainlabel

# ????????? 'LGBMmodel' 'XGBModel'
# XGBModel
dtrain = xgb.DMatrix(data=data.values, label=target.values)
progress = dict()
# xgbparamSetting()
param = {
            'learning_rate': 0.05,
            'eta': 0.4,
            'max_depth': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1,
            # 'lambda' : 0.1,
            'nthread': 4,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
# XGBbestNumRounds
num_round = 811
bst = xgb.train(param, dtrain, num_round, evals_result=progress)
bst.save_model('./xgbModelFinal')

# LGBMmodel
dtrain = lgb.Dataset(data=data.values, label=target.values)
progress = dict()
# LGBMparamSetting()
param = {
            'learning_rate': 0.01,
            'num_leaves': 32,
            # 'eta' : 0.4,
            'subsample': 0.35,
            'colsample_bytree': 0.3,
            'nthread': 4,
            # 'lambda_l1' : 0.1,
            'objective': 'binary',
            'metric': 'binary_logloss'
        }
# LGBMbestNumRounds
num_round = 2263
bst = lgb.train(param, dtrain, num_round, evals_result=progress)
bst.save_model('./lgbModelFinal')

datapath = './featureData.csv'
data = pd.read_csv(datapath ,sep=' ')
print(data['instance_id'].nunique())

data = data.iloc[trainLen : , : ] # ??????????

# # round1
# train_path_people_1='../../datasets/cut/round1/round1_train_cut_by_people.txt'
# train_path_type_1='../../datasets/cut/round1/round1_train_cut_by_type.txt'
# test_path_a_1='../../datasets/cut/round1/round1_ijcai_18_test_a_20180301.txt'
# test_path_b_1='../../datasets/cut/round1/round1_ijcai_18_test_b_20180418.txt'
# # round2
# train_path_type_2='../../datasets/cut/round2/round2_train_cut_by_type.txt'
# test_path_a_2='../../datasets/cut/round2/round2_test_a.txt'
# test_path_b_2='../../datasets/cut/round2/round2_test_b.txt'
# train=pd.read_table(train_path_type_1,delimiter=' ')
# test=pd.read_table(test_path_a_1,delimiter=' ')
# print(trainLen)
# print(train.shape)
# print(test.shape)

writefileName = './result.csv'

XGBmodel = xgb.Booster(model_file='./xgbModelFinal')
XGBpreds = XGBmodel.predict(xgb.DMatrix(data.values))
LGBMmodel = lgb.Booster(model_file='./lgbModelFinal')
LGBMpreds = LGBMmodel.predict(data.values)

preds = 0.5 * XGBpreds + 0.5 * LGBMpreds

sub = pd.DataFrame()
print(testInstanceID)
print(len(preds))
sub['instance_id'] = testInstanceID
sub['predicted_score'] = preds # ???????????????????????????????????

sub.to_csv(writefileName, sep=" ", index=False, line_terminator='\r')

pd.set_option('display.max_columns', None)  # ?
pd.set_option('display.max_rows', None)  # ?
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import time

import warnings

warnings.filterwarnings("ignore")

# ??????
# train_path='../../datasets/cut/round2/round2_train_cut_by_type.txt'
# # train_path_t='../../datasets/cut/round1/round1_train_cut_by_type.txt'
# # test_path_a='../../datasets/cut/round1/round1_ijcai_18_test_a_20180301.txt'
# # test_path_b='../../datasets/cut/round1/round1_ijcai_18_test_b_20180418.txt'
# datapath = '../../produce/mergeData.csv'

# data = pd.read_csv(train_path ,sep=' ')

# # ?timestamp???datetime?%Y-%m-%d %H:%M:%S?
# def timestamp_datetime(value):
#     format = '%Y-%m-%d %H:%M:%S'
#     value = time.localtime(value)
#     dt = time.strftime(format, value)
#     return dt # str

# # ???datetime64[ns]
# data['time'] = pd.to_datetime(data.context_timestamp.apply(timestamp_datetime))
# data['month'] = data.time.dt.month
# data['day'] = data.time.dt.day
# # data['hour'] = data.time.dt.hour

# data.groupby(['month', 'day']).count().to_csv('../../produce/count.csv')

datapath = './featureData.csv'
data = pd.read_csv(datapath ,sep=' ')

# ??????????????????
import re
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

filename = './serialize_constant'

with open(filename, 'rb') as f:
    serialize_constant = pickle.load(f)
    trainLen = serialize_constant['trainLen']
    trainlabel = serialize_constant['trainlabel']
    testInstanceID = serialize_constant['testInstanceID']

data = data.iloc[0 : trainLen, : ]
target = trainlabel

# model.modelFiveFoldEval(data, target) # 865 ??ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

data['target'] = np.array(target)

loglossList = []
avglogloss = 0
foldNum = 1

# print(data['day'].sort_values())

day24 = data[data['day'] == 24]

day18_23 = data[data['day'] < 24]
day19_23 = day18_23[data['day'] >= 18]

dataTrain = day19_23.drop(['target'], axis=1)
labelTrain = day19_23['target']

dataTest = day24.drop(['target'], axis=1)
labelTest = day24['target']

# print(dataTrain)
# print(labelTrain)
# print(dataTest)
# print(labelTest)
print('here')

dtrain = xgb.DMatrix(data=dataTrain.values, label=labelTrain.values)
dtest = xgb.DMatrix(data=dataTest.values, label=labelTest.values)

watchlist = [(dtrain, 'train'), (dtest, 'test')]
progress = dict()
# xgbparamSetting()
param = {
            'learning_rate': 0.05,
            'eta': 0.4,
            'max_depth': 3,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1,
            # 'lambda' : 0.1,
            'nthread': 4,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
# treeNum
num_boost_round = 3000

bst = xgb.train(param, dtrain, num_boost_round, watchlist, early_stopping_rounds=150, evals_result=progress)

bst.save_model('./xgbModel' + str(foldNum))

xgb.plot_importance(bst)

logList = progress['test']['logloss']

tmplogloss = np.min(np.array(logList))
avglogloss = avglogloss + tmplogloss
loglossList.append(logList)

num = len(loglossList)
val = len(loglossList[0])
result = []
for j in range(val):
    cal = 0
    for i in range(num):
        tmplist = loglossList[i]
        cal = cal + tmplist[j]
        cal = cal / num
        result.append(cal)
print(result)

dtrain = lgb.Dataset(dataTrain, label=labelTrain)
dtest = lgb.Dataset(dataTest, label=labelTest)

process = dict()
# LGBMparamSetting()
param = {
            'learning_rate': 0.01,
            'num_leaves': 32,
            # 'eta' : 0.4,
            'subsample': 0.35,
            'colsample_bytree': 0.3,
            'nthread': 4,
            # 'lambda_l1' : 0.1,
            'objective': 'binary',
            'metric': 'binary_logloss'
        }
# treeNum
num_boost_round = 3000

bst = lgb.train(param, dtrain, num_boost_round, valid_sets=dtest, early_stopping_rounds=500, evals_result=process)

bst.save_model('./lgbModel' + str(foldNum))

logList = process['valid_0']['binary_logloss']

# ????
score = bst.feature_importance()
name = bst.feature_name()
feature_score = sorted(zip(name, score), key=lambda a: a[1], reverse=True)
with open("./feature_score.txt", "a") as f:
    f.write(str(param) + '\n')
    f.write(str(bst.best_score) + str(bst.best_iteration) + '\n')
    f.write(str(feature_score) + '\n')

tmplogloss = np.min(np.array(logList))
avglogloss = avglogloss + tmplogloss
loglossList.append(logList)

num = len(loglossList)
val = len(loglossList[0])
result = []
for j in range(val):
    cal = 0
    for i in range(num):
        tmplist = loglossList[i]
        cal = cal + tmplist[j]
        cal = cal / num
        result.append(cal)
print(result)

