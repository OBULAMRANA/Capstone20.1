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

Given dataset has 1460 entries with 81columns


Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	RL	65.0	8450	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500
1	2	20	RL	80.0	9600	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500
2	3	60	RL	68.0	11250	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500
3	4	70	RL	60.0	9550	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000
4	5	60	RL	84.0	14260	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000
5 rows × 81 columns


## House Price Prediction Using The Dataset
An overview of the question to be solved in the capstone :
housing price are more volatile and depends on different factors. This project is to identify more effective ways for predicting housing price

## Identification of the type of data that will be used to solve the question:
The dataset will contain the house related information like area, built year, lot size, number of bed rooms, rest rooms,sale price ...etc

## List of 1–3 techniques that could be used to answer the question identified.:
Exploratory Data Analysis (EDA): Data preparation, Cleaning the data, Understand the data, examine relationships using visualizations such as histograms, scatter plots, and heatmaps.

Principal Component Analysis (PCA) & : Reduce dimensionality to identify and retain the most significant features contributing to house price.

Clustering : Apply clustering to uncover groups of houses with similar catagories like built-year, price, area. These clusters can offer insight on sale/price patterns and may be used as additional features in the final predictive model.
