# Project: Capstone Project
# Author:Rana Obulam
#### DataSource from kaggle : 
https://www.kaggle.com/datasets/gpandi007/usa-housing-dataset
#### DataSource in GITHUB: 
https://github.com/OBULAMRANA/Capstone20.1/blob/main/US_House_Data.csv
####  Python Jupiter NodeBook Code : 
https://github.com/OBULAMRANA/Capstone20.1/blob/main/Capstone20_1.ipynb

### Project Overview:
The  project is to identify  effective ways for predicting housing price based on the dataset. We will be training and tuning different set of classification & regression models to accurately predict the price. We will then evaluate and compare the models' performances to identify the best one, then further scrutinize it to find the most effective features that enhance performance.

### Dataset information
This dataset contains a wide variety of columns, offering a rich foundation for applying various encoding techniques for categorical data. By combining both numerical and categorical features, it becomes possible to build robust pipeline models for predicting house sale prices.

Given dataset has:

Total number of columns : 81
Total number of rows : 1460

<img width="795" height="411" alt="image" src="https://github.com/user-attachments/assets/e5625eb5-4d76-4e83-811b-5bbb7d282c5b" />

###  Data Cleaning and EDA
#### Date Understanding 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     588 non-null    object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
```
## Removing the data that is having missing values and having unique values that are not required.
```
Unique values in column MSZoning: ['RL' 'RM' 'C (all)' 'FV' 'RH']
Unique values in column Street: ['Pave' 'Grvl']
Unique values in column LotShape: ['Reg' 'IR1' 'IR2' 'IR3']
Unique values in column LandContour: ['Lvl' 'Bnk' 'Low' 'HLS']
Unique values in column Utilities: ['AllPub' 'NoSeWa']
Unique values in column LotConfig: ['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3']
Unique values in column LandSlope: ['Gtl' 'Mod' 'Sev']
Unique values in column BldgType: ['1Fam' '2fmCon' 'Duplex' 'TwnhsE' 'Twnhs']
Unique values in column ExterQual: ['Gd' 'TA' 'Ex' 'Fa']
Unique values in column ExterCond: ['TA' 'Gd' 'Fa' 'Po' 'Ex']
Unique values in column BsmtQual: ['Gd' 'TA' 'Ex' nan 'Fa']
Unique values in column BsmtCond: ['TA' 'Gd' nan 'Fa' 'Po']
Unique values in column BsmtExposure: ['No' 'Gd' 'Mn' 'Av' nan]
Unique values in column HeatingQC: ['Ex' 'Gd' 'TA' 'Fa' 'Po']
Unique values in column CentralAir: ['Y' 'N']
Unique values in column Electrical: ['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix' nan]
Unique values in column KitchenQual: ['Gd' 'TA' 'Ex' 'Fa']
Unique values in column FireplaceQu: [nan 'TA' 'Gd' 'Fa' 'Ex' 'Po']
Unique values in column GarageFinish: ['RFn' 'Unf' 'Fin' nan]
Unique values in column GarageQual: ['TA' 'Fa' 'Gd' nan 'Ex' 'Po']
Unique values in column GarageCond: ['TA' 'Fa' nan 'Gd' 'Po' 'Ex']
Unique values in column PavedDrive: ['Y' 'N' 'P']
Total categorical column in cdf dataset is: 22
Applying Ordinal encoding techniques for some of the Caterogical data columns
```

##  Heatmap on the dataset 

<img width="619" height="510" alt="image" src="https://github.com/user-attachments/assets/6618ab7d-a45b-4103-98d4-666b11015aa6" />

## Applying Ordinal encoding techniques for some of the Caterogical data columns

```
# Unique values in column Street: ['Pave' 'Grvl']
Street_map = {"Grvl": 0, "Pave": 1}
cdf['Street'] = cdf['Street'].map(Street_map)

# Unique values in column LotShape: ['Reg' 'IR1' 'IR2' 'IR3']
LotShape_map = {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3}
cdf['LotShape'] = cdf['LotShape'].map(LotShape_map)

# Unique values in column LandContour: ['Lvl' 'Bnk' 'Low' 'HLS']
LandContour_map = {"Low": 0, "Lvl": 1, "Bnk": 2, "HLS": 3}
cdf['LandContour'] = cdf['LandContour'].map(LandContour_map)

# Unique values in column Utilities: ['AllPub' 'NoSeWa']
Utilities_map = {"NoSeWa": 0, "AllPub": 1}
cdf['Utilities'] = cdf['Utilities'].map(Utilities_map)

# Unique values in column LotConfig: ['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3']
LotConfig_map = {"Inside": 0, "FR2": 1, "FR3": 2, "Corner":3, "CulDSac":4}
cdf['LotConfig'] = cdf['LotConfig'].map(LotConfig_map)
```
## headmap using the calculated correlation matrix
```
sns.set_style('whitegrid')
plt.figure(figsize=(20,20))
corr = cdf[numerical_columns_cdf].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm',mask=mask)
plt.tight_layout()
```
 <img width="1847" height="1990" alt="image" src="https://github.com/user-attachments/assets/8e1cfce7-6f48-4385-a2bf-8fca159b963b" />

 
## Interpritation of Evalution Matrics

### Evaluate the model: 

It calculates and prints three evaluation metrics: 

Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values. The value of 0.0228 indicates that model evaluation is prety decent. 

Root Mean Squared Error (RMSE): The square root of MSE, providing an error metric in the same units as the target variable. Value of RMSE 0.151 indicates that the value is low indicates that the model has less errors. 

R-squared (R2): Represents the proportion of variance in the target variable explained by the model. It is a generic matric and can be used for baseline model, and closer to 1 is better.


