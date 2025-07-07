#King County House Price Prediction 
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df=pd.read_csv('kc_house_data.csv') 

pd.set_option('display.max_columns', None)  # tüm sütunları gösterir
print(df.head())

#pip install ydata-profiling 13 altı sürüm istiyor 

#veri seti hakkında bilgi 
print(df.info())

#en düşük fiyatli ev, en yüksek fiyatlı ev, ortalama
print("en düşük ev fiyatı: ",df['price'].min())
print("en yüksek ev fiyatı: ",df['price'].max())
print(df['price'].mean())

# en yüksek fiyatlı evin özellikleri
print(df[df['price']==df['price'].max()])

#en fazla yatak odası sayısı 
print(df[df['bedrooms']==df['bedrooms'].max()].value_counts('bedrooms'))

#yapım yılı en eski en yaşlının enlem boylam bilgileri
print(df[df['date']==df['date'].min()][['lat','long']])

#en pahalı evin enlem boylam bilgileri
print(df[df['price']==df['price'].max()][['lat','long']])

# veri setindeki aykırı değerleri ayıklamak, aykırı değerleri çıkarmak 
df_outlier_col=df[['price','bedrooms','sqft_living','sqft_lot']]
outliers=df_outlier_col.quantile(q=.99) # %1 aykırı değer alınsın 
print(outliers) #bedroomsda 6 nın üstü değerler aykırı değerlerdir gibi gibi 

#şimdi bu değerleri kullanmamız lazım 

df_clean=df[df['price']<outliers['price']] #aykırı değerleri çıkarmış olduk 
print(df_clean['price'].max()) #aykırı olmayanlardan max a bakıcaz ki işe yaramış mı 

#aynısıını diğer alanlar için de yapalım bu sefer df_clean olarak yazıcaz 
df_clean=df_clean[df_clean['bedrooms']<outliers['bedrooms']]
print(df_clean['bedrooms'].max()) #6 ve üstü aykırı değerdi.şimdi max 5 oldu
df_clean=df_clean[df_clean['sqft_living']<outliers['sqft_living']]
print(df_clean['sqft_living'].max())
df_clean=df_clean[df_clean['sqft_lot']<outliers['sqft_lot']]
print(df_clean['sqft_lot'].max())

print(df_clean.describe()) # yeni değerler ortaya çıkar 

df_corr=df_clean.corr(numeric_only=True).sort_values('price',ascending=False)['price'].head(10)
# ev fiyatı ile diğer özellikler arasındaki korelsayonu sıralama
print(df_corr)

df_clean['age']=2015-df_clean['yr_built'] #evin yaşını hesaplıyoruz en son 2015 yılında satılmış
df_clean['restore']=np.where(df_clean['yr_renovated']==0,0,1)# restorasyon yapılmış mı yapılmamış mı sıfırsa edilmemiş 1 se edilmiş
df_clean['sqft_total']=df_clean['sqft_living']+df_clean['sqft_lot']
df_clean['sqft_basement']=np.where(df_clean['sqft_basement']==0,0,1) # bodrum var mı yok mu