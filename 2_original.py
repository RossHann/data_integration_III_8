# ?????????????????????????????????????????????????
import matplotlib
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
from pasta.augment import inline
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
