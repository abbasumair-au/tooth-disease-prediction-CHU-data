# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:12:17 2020

@author: HP
"""
import openpyxl
import pandas as pd
dataset=openpyxl.load_workbook('C:/Users/HP/Desktop/computational intelligence project/dataset_1_2.0.xlsx')
sheets=dataset.sheetnames
print(sheets)
data=pd.read_excel("C:/Users/HP/Desktop/computational intelligence project/dataset_1_2.0.xlsx",sheet_name="EpiParo ")
print(data) 
print(data.columns)
#print(data.describe())
#delete all rows which have null values
'''
-- first function to delete rows with null value
for co in data.columns:
    data_final=data[data[co].notna()]
    data=data_final
print(data_final)
data_final.to_excel('data_final.xlsx')
'''

#second function 
data_final=data.dropna(axis=0,how='any')
print(data_final)
data_final.to_excel('C:/Users/HP/Desktop/computational intelligence project/data_final.xlsx')
mydata=pd.read_excel("C:/Users/HP/Desktop/computational intelligence project/data_final.xlsx")
print(mydata)






