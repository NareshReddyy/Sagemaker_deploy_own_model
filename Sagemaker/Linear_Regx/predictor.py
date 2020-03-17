# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:32:40 2019

@author: naresh.gangiredd
"""

import os
import json
from sklearn.externals import joblib
import flask
#import boto3
#from boto3.s3.connection import S3Connection
#from botocore.exceptions import ClientError
#import pickle
import pandas as pd
import logging
import boto3
import numpy as np
import datetime
import re
import math
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
logging.info("Model Path" + str(model_path))

# Load the model components
regressor = joblib.load(os.path.join(model_path, 'Regx.pkl'))
logging.info("Regressor" + str(regressor))

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        #regressor
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    #input_json = json.dumps(input_json['input']['exp1'])
	req_body = input_json
	if not req_body:
			print("no input")
			status_code=400
	else:
		userdata = req_body['FusionRequest'][0]
		for k,v in userdata.items():
			userdata[k] = [v]
	Loads_data = pd.DataFrame.from_dict(userdata)
	Loads_data['Date'] = pd.to_datetime(Loads_data['Date'])
	#Date = datetime.datetime(2019,11,24)
	
	#READ PATH - S3
	bucket_name = 's3-pricing-2.0'
	sunteck_key = 'sunteck_30_days_data.csv'
	mode_key = 'mode_data_30_days_data.csv'
	market_mapping_key = 'USAllZipsMarketMapping - USAllZipsMarketMapping (1).csv'
	equipment_mapping_key ='EquipmentDescription_mapping.csv'
	actual_distance_key = 'Actual Distance.csv'
	ITS_data_key ='ITS_8_weeks_data.csv'
	DAT_data_key = 'DAT_8_week_data.csv'
	ziptozip_key = 'ZipToZipSummary.csv'
	markettomarket_key = 'MarketToMarketSummary.csv'
	ITS_data_key = 'ITS_MarketToMarketSummary.csv'
	DAT_data_key = 'DAT_MarketToMarketSummary.csv'
	modelnames_key ='ModelNames_new.csv'
	modelcoefficients_key = 'AllModelsCoefficients.csv'
	fuelprice_key = 'FuelPrice_Nation_Weekly.csv' 
	distance_lt_10 = 'Distance_10_avgPrice.csv'
	model_read_key = 'Pricing2_0_Model.pkl'
	
	#function to read data from s3
	def read_data_from_s3(bucket_name,key):
		obj = client.get_object(Bucket=bucket_name, Key=key)
		input_df = pd.read_csv(obj['Body'])
		return input_df

	#function to write data to s3
	def write_data_to_s3(bucket_name,write_key,filename):
		#with open(filename, 'r') as f:
		#    content = f.read()
		#client.put_object(Body = content, Bucket = bucket_name, Key = write_key)
		client.upload_file(filename, bucket_name, write_key)
	
	def GetMarketHistory(OriginMarketId, DestMarketId, Distance, Equipment, Date):
	  '''
	  function to get Market History(7day/30day) from Mode & Sunteck data for the given route, Equipment, date
	  function arguments 
		OriginMarketId : from market id
		DestMarketId : to market id
		Distance : Distance, to access summary file
		Equipment : EquipmentType
		Date : date 
	  '''
	  return_data = pd.DataFrame()
	  Distance_bin = np.where(Distance > 150, 1, 0)
	  selectedData = MarketToMarketFeatures[(MarketToMarketFeatures.OriginMarketID == OriginMarketId) & 
											(MarketToMarketFeatures.DestMarketID == DestMarketId) & 
											(MarketToMarketFeatures.Equipment == Equipment) & 
											(MarketToMarketFeatures.Distance_bin == Distance_bin) &
											(MarketToMarketFeatures.Date == Date)]  
	  if(len(selectedData) > 0):
		return_data = {'NoOfLoads_30days_MM' : selectedData.NoOfLoads_30days.iloc[0], 
					   'Price_30days_MM' : selectedData.RPM_30days.iloc[0] * Distance, 
					   'NoOfLoads_7days_MM' : selectedData.NoOfLoads_7days.iloc[0],
					   'Price_7days_MM' : selectedData.RPM_7days.iloc[0] * Distance, 
					   'NoOfLoads_MM' : selectedData.NoOfLoads_MM.iloc[0], 
					   'Price_MM' : selectedData.RPM_MM.iloc[0] * Distance, 
					   'Price2_MM' : selectedData.RPM2_MM.iloc[0] * (Distance**0.5)} 
	  return(return_data)

	def GetZipHistory(OriginZip, DestZip, Distance, Equipment, Date):
	  '''
	  function to get Zip History(7day/30day) from Mode & Sunteck data for the given route, Equipment, date
	  function arguments 
		OriginZip : from zipcode
		DestZip : to zipcode
		Distance : Distance
		Equipment : EquipmentType
		Date : date 
	  '''
	  return_data = pd.DataFrame()
	  Distance_bin = np.where(Distance > 150, 1, 0)
	  selectedData = ZipToZipFeatures[(ZipToZipFeatures.OriginZip == OriginZip) & 
									  (ZipToZipFeatures.DestZip == DestZip) & 
									  (ZipToZipFeatures.Equipment == Equipment) & 
									  (ZipToZipFeatures.Distance_bin == Distance_bin) &
									  (ZipToZipFeatures.Date == Date)]
	  if(len(selectedData) > 0):
		return_data = {'NoOfLoads_30days_ZZ' : selectedData.NoOfLoads_30days.iloc[0], 
					   'Price_30days_ZZ' : selectedData.RPM_30days.iloc[0] * Distance, 
					   'NoOfLoads_7days_ZZ' : selectedData.NoOfLoads_7days.iloc[0],
					   'Price_7days_ZZ' : selectedData.RPM_7days.iloc[0]  * Distance, 
					   'NoOfLoads_ZZ' : selectedData.NoOfLoads_ZZ.iloc[0], 
					   'Price_ZZ' : selectedData.RPM_ZZ.iloc[0] * Distance, 
					   'Price2_ZZ' : selectedData.RPM2_ZZ.iloc[0] * (Distance**0.5)}     
	  return return_data


	def GetDATfeatures(OriginMarketId, DestMarketId, Distance, Equipment, Date):

	  return_data = pd.DataFrame()
	  monday_date = Date - pd.Timedelta(days=Date.weekday())
	  Distance_bin = np.where(Distance > 150, 1, 0)
	  selectedData = DAT_data[(DAT_data.OriginMarketID == OriginMarketId) & 
									  (DAT_data.DestMarketID == DestMarketId) & 
									  (DAT_data.Equipment == Equipment) & 
									  (DAT_data.Distance_bin == Distance_bin) &
									  (DAT_data.FirstMonday <= monday_date) &
									  (DAT_data.FirstMonday >= (Date - pd.Timedelta(days=7*8)))]
	  selectedData = selectedData.sort_values(['FirstMonday'], ascending = False)
	  if(len(selectedData) > 0):
		return_data = {'NoofReports_DAT' : selectedData.NoofReports_DAT.iloc[0], 
					  'Price_DAT' : selectedData.RPM_DAT.iloc[0] * Distance}     
	  return(return_data)  

	def GetITSfeatures(OriginMarketId, DestMarketId, Equipment, Date):
	  '''
	  function to get ThirdParty(ITS) Market History(8 Weeks) for the given route, EquipmentType, date
	  function arguments
		OriginMarketId : from market id
		DestMarketId : to market id
		Distance : for distance bin >150, <=150 features
		EquipmentType : EquipmentType
		Date : date
	  '''
	  return_data = pd.DataFrame()
	  monday_date = Date - pd.Timedelta(days=Date.weekday())
	  selectedData = ITS_data[(ITS_data.OriginMarketID == OriginMarketId) & (ITS_data.DestMarketID == DestMarketId) & 
							  (ITS_data.Equipment == Equipment) & (ITS_data.FirstMonday <= monday_date) &
									  (ITS_data.FirstMonday >= (Date - pd.Timedelta(days=7*8)))]

	  selectedData = selectedData.sort_values(['FirstMonday'], ascending = False)
	  if(len(selectedData) > 0):
		return_data = {'NoofReports_ITS' : selectedData.NoofReports_ITS.iloc[0], 
					  'Price_ITS' : selectedData.RPM_ITS.iloc[0] * Distance}     
	  return(return_data)  

	#ModelNames = pd.read_csv("ModelNames_new.csv")
	ModelNames = read_data_from_s3(bucket_name,modelnames_key)
	ModelNames.rename(columns = {'Distance>150' : 'Distance_bin'}, inplace = True)
	ModelNames.drop(columns = ['Version'], inplace = True)
	def GetModelName(data):
	  data['Distance_bin'] = np.where(data.Distance > 150, 1, 0)
	  data['Set3(30day)'] = np.where(data.NoOfLoads_30days_MM > 0, 1, 0)
	  data['Set3(7day)'] = np.where(data.NoOfLoads_7days_MM > 0, 1, 0)
	  data['Set4(30day)'] = np.where(data.NoOfLoads_30days_ZZ > 0, 1, 0)
	  data['Set4(7day)'] = np.where(data.NoOfLoads_7days_ZZ > 0, 1, 0)
	  data['Set2'] = np.where(data.NoofReports_DAT > 0, 1, 0)
	  data['Set3(nDay)'] = np.where(data.NoOfLoads_MM > 0, 1, 0)
	  data['Set4(nDay)'] = np.where(data.NoOfLoads_ZZ > 0, 1, 0)

	  data = pd.merge(data, ModelNames, how = 'left', 
							on = ['Set2', 'Set3(7day)', 'Set3(30day)', 'Set4(7day)', 'Set4(30day)', 
								  'Set3(nDay)', 'Set4(nDay)',  'Distance_bin'])
	  if((data['ModelName'][0] == '1_EV') & (data['NoofReports_DAT_ITS'][0] > 0)):
		return {'ModelName' : '1.2_EV_DAT_ITS', 'Level' : 'Level 2'}
	  else:
		return {'ModelName' : data['ModelName'][0], 'Level' : data['Level'][0]}


	#ZipsMarketMapping = pd.read_csv("USAllZipsMarketMapping - USAllZipsMarketMapping.csv")
	ZipsMarketMapping = read_data_from_s3(bucket_name,market_mapping_key)
	def GetMarketId(Zipcode):
	  MarketId = ZipsMarketMapping[ZipsMarketMapping.Zipcode == Zipcode]
	  if(len(MarketId) == 1):
		return(MarketId[['MarketID']].iloc[0,0])
	  else:
		return(np.nan)

	def GetFuelPrice(Date):
	  '''
	  function to get fuel price of the given date week monday
	  '''
	  monday_date = Date - pd.Timedelta(days=Date.weekday())
	  FuelPrice = FuelRates[FuelRates.Date == monday_date]
	  if(len(FuelPrice) == 1):
		FuelPrice = FuelPrice[['FuelRate']].iloc[0,0]
	  else:
		FuelPrice = np.nan
	  return FuelPrice


	def Predict_Price(OriginZip, DestZip, Equipment, Date, NoOfStops, Distance):
	  '''
	  function to predict price : based data on the mode & sunteck data history features , thirsparty DAT & ITS history, 
	  it will select corresponsding model pickle file and predict price
	  function arguments
		OriginZip : Origin Zipcode
		DestZip : Dest Zipcode
		EquipmentType : EquipmentType
		Date : pickup planned date
		NoOfStops : No of intermiadiate stops
		Distance : Distance

	  output : predicted price, if any error in given features data it returns null value and also error message
	  '''

	  # 1. Get fuel price & correspoding MarketId for OriginZip and DestZip
	  # 2. Get third party DAT & ITS history
	  # 3. Get Mode & Sunteck Market level history
	  # 4. Get Mode & Sunteck Zip level history
	  # 5. Select corresponding model and predict price and return predicted price and confidence interval levels of price
		  

	  predictedPrice = np.nan
	  FuelRate = GetFuelPrice(Date)
	  if(pd.isnull(FuelRate) == False):
		history_data = pd.DataFrame({'Equipment' : [Equipment], 'Distance' : [Distance], 'FuelRate' : [FuelRate], 
							  'Year' : [Date.year + 1 - 2016], 'Month' : [Date.strftime("%B")], 'Weekday' : [Date.strftime("%A")], 
							  'NStops' : [NoOfStops]})
		OriginMarketId = GetMarketId(OriginZip)
		DestMarketId = GetMarketId(DestZip)

		features_names = Model_Features_Ref['7_EV_TD_MM7_MZ7'] + TD_1_2 + MM + MZ
		new_cols = list(set(features_names) - set(history_data.columns))
		history_data[new_cols] = pd.DataFrame([[np.nan] * len(new_cols)], index=history_data.index)

		if((pd.isnull(OriginMarketId) == False) & (pd.isnull(DestMarketId) == False)):

		  if(Distance <= 10):
			ZZ_avgPrice = Distance_10_avgPrice[(Distance_10_avgPrice.OriginZip == OriginZip) & 
											  (Distance_10_avgPrice.DestZip == DestZip)]
			if(len(ZZ_avgPrice) > 0):
			  predictedPrice = ZZ_avgPrice.Cost.iloc[0]
			  return(predictedPrice, 'Distance<=10_ZZ_avgPrice')     
			else:                            
			  return(300, 'Distance<=10_fixedPrice')         


		  # 2. Get third party history

		  # DAT features
		  DAT_History = GetDATfeatures(OriginMarketId, DestMarketId, Distance, Equipment, Date)
		  if(len(DAT_History) > 0):
			history_data.NoofReports_DAT.iloc[0] = DAT_History['NoofReports_DAT']
			history_data.Price_DAT.iloc[0] = DAT_History['Price_DAT']
		  else:
			# ITS features
			ITS_History = GetITSfeatures(OriginMarketId, DestMarketId, Equipment, Date)
			if(len(ITS_History) > 0):
			  history_data.NoofReports_DAT_ITS.iloc[0] = ITS_History['NoofReports_ITS']
			  history_data.Price_DAT_ITS.iloc[0] = ITS_History['Price_ITS']

		  MarketHistory = GetMarketHistory(OriginMarketId, DestMarketId, Distance, Equipment, Date)
		  if(len(MarketHistory) > 0):     
			history_data.NoOfLoads_30days_MM.iloc[0] = MarketHistory['NoOfLoads_30days_MM']
			history_data.Price_30days_MM.iloc[0] = MarketHistory['Price_30days_MM']
			history_data.NoOfLoads_7days_MM.iloc[0] = MarketHistory['NoOfLoads_7days_MM']
			history_data.Price_7days_MM.iloc[0] = MarketHistory['Price_7days_MM']
			history_data.NoOfLoads_MM.iloc[0] = MarketHistory['NoOfLoads_MM']
			history_data.Price_MM.iloc[0] = MarketHistory['Price_MM']
			history_data.Price2_MM.iloc[0] = MarketHistory['Price2_MM']

			ZipHistory = GetZipHistory(OriginZip, DestZip, Distance, Equipment, Date)
			if(len(ZipHistory) > 0):
			  history_data.NoOfLoads_30days_ZZ.iloc[0] = ZipHistory['NoOfLoads_30days_ZZ']
			  history_data.Price_30days_ZZ.iloc[0] = ZipHistory['Price_30days_ZZ']
			  history_data.NoOfLoads_7days_ZZ.iloc[0] = ZipHistory['NoOfLoads_7days_ZZ']
			  history_data.Price_7days_ZZ.iloc[0] = ZipHistory['Price_7days_ZZ']
			  history_data.NoOfLoads_ZZ.iloc[0] = ZipHistory['NoOfLoads_ZZ']
			  history_data.Price_ZZ.iloc[0] = ZipHistory['Price_ZZ']
			  history_data.Price2_ZZ.iloc[0] = ZipHistory['Price2_ZZ']

		  modelInfo = GetModelName(history_data)
		  modelName = modelInfo['ModelName']
		  modelLevel = modelInfo['Level']
		  
		  loaded_model = AllModelsPickleFile[modelName]    
		  coefficients = AllModelsCoefficients[AllModelsCoefficients.ModelName == modelName].copy()     
		  coefficients = coefficients[coefficients.Feature != 'Intercept']
		  X_test = pd.get_dummies(history_data[features_names])
		  new_cols = list(set(coefficients.Feature.tolist()) - set(X_test.columns))
		  X_test[new_cols] = pd.DataFrame([[0] * len(new_cols)], index=X_test.index)
		  X_test = X_test[coefficients.Feature.tolist()]
		  predictedPrice = round(loaded_model.predict(X_test)[0], 3)
		  
		  return(predictedPrice, modelName)

		else :
		  error_str = "either Origin Zip or Destination Zipcode does not exists in USA"
		  print(error_str)
	  else :
		error_str = "Fuel price does not exists"  
		print(error_str)
		  
	# Models feature names for reference
	EV = ['Equipment', 'Distance', 'FuelRate', 'Year', 'Month', 'Weekday', 'NStops']
	TD = ['Price_DAT', 'NoofReports_DAT']

	MM = ['NoOfLoads_MM', 'Price_MM', 'Price2_MM']
	MZ = ['NoOfLoads_ZZ', 'Price_ZZ', 'Price2_ZZ']

	MM30 = ['NoOfLoads_30days_MM', 'Price_30days_MM']
	MM7 = MM30 + ['NoOfLoads_7days_MM', 'Price_7days_MM']
	MZ30 = ['NoOfLoads_30days_ZZ', 'Price_30days_ZZ']
	MZ7 = MZ30 + ['NoOfLoads_7days_ZZ', 'Price_7days_ZZ']

	TD_1_2 = ['Price_DAT_ITS', 'NoofReports_DAT_ITS']


	Model_Features_Ref = {# Models for distances greater than 150 miles
						  '1_EV' : EV, '1.2_EV_DAT_ITS' : EV + TD_1_2, 
						  '2_EV_TD' : EV + TD, '3_EV_TD_MM7' : EV + TD + MM7, '4_EV_MM7' : EV + MM7, 
						  '5_EV_TD_MM30' : EV + TD + MM30, '6_EV_MM30' : EV + MM30, '7_EV_TD_MM7_MZ7' : EV + TD + MM7 + MZ7,
						  '8_EV_MM7_MZ7' : EV + MM7 + MZ7, '9_EV_TD_MM7_MZ30' : EV + TD + MM7 + MZ30, 
						  '10_EV_MM7_MZ30' : EV + MM7 + MZ30, '11_EV_TD_MM30_MZ30' : EV + TD + MM30 + MZ30,
						  '12_EV_MM30_MZ30' : EV + MM30 + MZ30,
						  # Models for distances (10,150] miles
						  '13_EV' : EV, '14_EV_TD' : EV + TD, '15_EV_MM' : EV + MM, '16_EV_TD_MM' : EV + TD + MM, 
						  '17_EV_MZ' : EV + MZ, '18_EV_TD_MZ' : EV + TD + MZ
						  }
		

	#results_path = '/content/drive/My Drive/Pricing 2.0/Model building/ModelTraining/Test_Final'
	response = client.get_object(Bucket=bucket_name, Key=model_read_key)
	body = response['Body'].read()
	#data = pickle.loads(body)
	AllModelsPickleFile = pickle.loads(body)
	#AllModelsCoefficients = pd.read_csv("AllModelsCoefficients.csv")	
	AllModelsCoefficients = read_data_from_s3(bucket_name,modelcoefficients_key)

		
	# Mode&Sunteck Zip to Zip summary features
	#ZipToZipFeatures = pd.read_csv("ZipToZipSummary.csv")
	ZipToZipFeatures = read_data_from_s3(bucket_name,ziptozip_key)
	ZipToZipFeatures['Date'] = pd.to_datetime(ZipToZipFeatures['Date'], format = "%Y-%m-%d")

	# Mode&Sunteck Market to Market summary features
	#MarketToMarketFeatures = pd.read_csv("MarketToMarketSummary.csv")
	MarketToMarketFeatures = read_data_from_s3(bucket_name,markettomarket_key)
	MarketToMarketFeatures['Date'] = pd.to_datetime(MarketToMarketFeatures['Date'], format = "%Y-%m-%d")

	# Fuel prices
	#FuelRates = pd.read_csv('FuelPrice_Nation_Weekly.csv')
	FuelRates = read_data_from_s3(bucket_name,fuelprice_key)
	FuelRates['Date'] = pd.to_datetime(FuelRates['Date'], format = "%m/%d/%Y")
	FuelRates.columns = ['Date', 'FuelRate']

	# Thirdparty(DAT) Market to Market summary features
	#DAT_data = pd.read_csv("DAT_MarketToMarketSummary.csv")
	DAT_data = read_data_from_s3(bucket_name,DAT_data_key)
	DAT_data['FirstMonday'] = pd.to_datetime(DAT_data['FirstMonday'], format = "%Y-%m-%d")
	DAT_data.rename(columns = {'OriginMarket' : 'OriginMarketID', 'DestMarket' : 'DestMarketID', 'TruckType' : 'Equipment',
							   'distance_bin' : 'Distance_bin'}, inplace = True)
	DAT_data = DAT_data[['FirstMonday', 'OriginMarketID', 'DestMarketID', 'Equipment', 'Distance_bin', 'NoofReports_DAT',	
						 'RPM_DAT']]
	DAT_data['Distance_bin'] = np.where(DAT_data['Distance_bin'] == '>150', 1, 0)

	# Thirdparty(ITS) Market to Market summary features
	#ITS_data = pd.read_csv("ITS_MarketToMarketSummary.csv")
	ITS_data = read_data_from_s3(bucket_name,ITS_data_key)
	ITS_data['FirstMonday'] = pd.to_datetime(ITS_data['FirstMonday'], format = "%Y-%m-%d")
	print(ITS_data.columns)
	ITS_data.rename(columns = {'OriginMarketId' : 'OriginMarketID', 'DestMarketId' : 'DestMarketID', 
							   'TruckType' : 'Equipment'}, inplace = True)
	ITS_data['Distance_bin'] = 1
	ITS_data = ITS_data[['FirstMonday', 'OriginMarketID', 'DestMarketID', 'Equipment', 'Distance_bin', 'NoofReports_ITS',	
						 'RPM_ITS']]

	ZipToZipFeatures.rename(columns = {'NoOfLoads' : 'NoOfLoads_ZZ', 'RPM' : 'RPM_ZZ', 'RPM2' : 'RPM2_ZZ'}, inplace = True)
	MarketToMarketFeatures.rename(columns = {'NoOfLoads' : 'NoOfLoads_MM', 'RPM' : 'RPM_MM', 'RPM2' : 'RPM2_MM'}, inplace = True)

	#Distance_10_avgPrice = pd.read_csv('Distance_10_avgPrice.csv')
	Distance_10_avgPrice = read_data_from_s3(bucket_name,distance_lt_10)
	
    predictions = Predict_Price(int(Loads_data.iloc[0]['OriginZip']), int(Loads_data.iloc[0]['DestZip']),Loads_data.iloc[0]['Equipment'], Loads_data.iloc[0]['Date'], 0, int(Loads_data.iloc[0]['Distance']))
    #predictions = float(regressor.predict([[input]]))

    # Transform predictions to JSON
    result = {
        'output': predictions
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')