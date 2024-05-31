import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

cars_df = pd.read_csv('data/cars_after_eda.csv')
orig_cars_df = cars_df.copy()

columns = ['brand', 'model', 'model_variants', 'seller_type', 'transmission', 'fuel', 'owner', 'km_driven_bin', 'engine_bin', 'mileage_bin', 'max_power_bin']
cars_df_labels = cars_df[columns + ['year']]

lEnc = LabelEncoder()

mappings = list()

for col in columns:
    encoded_labels = lEnc.fit_transform(cars_df[col])
    cars_df[col] = encoded_labels
    cur_dict = dict(zip(lEnc.classes_, range(len(lEnc.classes_))))
    mappings.append(cur_dict)


X = cars_df.drop(['name','selling_price', 'km_driven', 'engine_cc', 'mileage_kmpl', 'max_power_bhp'], axis=1)
Y = cars_df['selling_price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

columns = ['brand', 'model', 'seller_type', 'transmission', 'fuel', 'owner', 'model_variants']
models = {'LR': 'LinearRegressor', 'XGB': 'Extra gradient boosting', 'RFR': 'Random forest regression', 'Bagging': 'BaggingRegressor', 'ETR': 'ExtraTreesRegressor'}

st.set_page_config(layout="wide", page_title="Car Price Prediction")

@st.cache_resource
def getLRModel(testPredictions, modelsPredictMetrics):
    # Loading the Linear Regression model
    lRegModel = LinearRegression()
    lRegModel.fit(X_train, Y_train)
    lrPredicts = lRegModel.predict(X_test)
    lrScore = r2_score(Y_test, lrPredicts)
    lrError = np.sqrt(mean_squared_error(Y_test, lrPredicts))
    print('R2-Score LR ', lrScore)
    print('Error square LR ', lrError)
    modelsPredictMetrics.append(['Linear Regression', lrScore, lrError])
    testPredictions[models['LR']] = lrPredicts
    return lRegModel

@st.cache_resource
def getXGBModel(testPredictions, modelsPredictMetrics):
    # Create a XGB model and train created model.
    xgb = XGBRegressor()
    xgb.fit(X_train, Y_train)
    xgbPredicts = xgb.predict(X_test)
    xgbScore = r2_score(Y_test, xgbPredicts)
    xgbError = np.sqrt(mean_squared_error(Y_test, xgbPredicts))
    print('R2-Score XGB ', xgbScore)
    print('Error square XGB ', xgbError)
    modelsPredictMetrics.append(['Extra gradient boosting', xgbScore, xgbError])
    testPredictions[models['XGB']] = xgbPredicts
    return xgb

@st.cache_resource
def getRFRModel(testPredictions, modelsPredictMetrics):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train)
    rfrPredicts = rfr.predict(X_test)
    rfrScore = r2_score(Y_test, rfrPredicts)
    rfrError = np.sqrt(mean_squared_error(Y_test, rfrPredicts))
    print('R2-Score RFR ', rfrScore)
    print('Error square RFR ', rfrError)
    modelsPredictMetrics.append(['Random forest regression', rfrScore, rfrError])
    testPredictions[models['RFR']] = rfrPredicts
    return rfr

@st.cache_resource
def getBaggingModel(testPredictions, modelsPredictMetrics):
    bagging = BaggingRegressor()
    bagging.fit(X_train, Y_train)
    baggingPredicts = bagging.predict(X_test)
    baggingScore = r2_score(Y_test, baggingPredicts)
    baggingError = np.sqrt(mean_squared_error(Y_test, baggingPredicts))
    print('R2-Score Bagging ', baggingScore)
    print('Error square Bagging ', baggingError)
    modelsPredictMetrics.append(['BaggingRegressor', baggingScore, baggingError])
    testPredictions[models['Bagging']] = baggingPredicts
    return bagging

@st.cache_resource
def getETRModel(testPredictions, modelsPredictMetrics):
    etr = ExtraTreesRegressor()
    etr.fit(X_train, Y_train)
    etrPredicts = etr.predict(X_test)
    etrScore = r2_score(Y_test, etrPredicts)
    etrError = np.sqrt(mean_squared_error(Y_test, etrPredicts))
    print('R2-Score ExtraTreesRegressor ', etrScore)
    print('Error square ExtraTreesRegressor ', etrError)
    modelsPredictMetrics.append(['ExtraTreesRegressor', etrScore, etrError])
    testPredictions[models['ETR']] = etrPredicts
    return etr

def initModels():
    print('Inits models')
    testPredictions = dict()
    modelsPredictMetrics = list()
    getLRModel(testPredictions, modelsPredictMetrics)
    getRFRModel(testPredictions, modelsPredictMetrics)
    getXGBModel(testPredictions, modelsPredictMetrics)
    getBaggingModel(testPredictions, modelsPredictMetrics)
    getETRModel(testPredictions, modelsPredictMetrics)
    if 'testPredictions' not in st.session_state:
        st.session_state['testPredictions'] = testPredictions
    if 'modelsPredictMetrics' not in st.session_state:
        st.session_state['modelsPredictMetrics'] = modelsPredictMetrics

def plotForTab(tab, headn:str, x, y):
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots(figsize = (3,3))
    plt.subplot(2, 2, 1)
    plt.hist(arr, bins=20)
    plt.subplot(2, 2, 2)
    plt.boxplot(arr)
    plt.subplot(2, 2, 3)
    #plt.pie(cars_df.groupby(x)[y].mean())
    plt.boxplot(arr)
    plt.subplot(2, 2, 4)
    plt.plot(cars_df[x].to_xarray(), np.array(cars_df[y]), 'o-')
    plt.savefig('x',dpi=400)
    tab.pyplot(fig) 
    os.remove('x.png') 


def plotForTabParam(tab, input_year):
    title = 'based on highest selling'
    fuel_data = orig_cars_df.query('year == @input_year').groupby("fuel").size()
    transmission_data = orig_cars_df.query('year == @input_year').groupby("transmission").size()
    owner_data = orig_cars_df.query('year == @input_year').groupby("owner").size() 
    seller_data = orig_cars_df.query('year == @input_year').groupby("seller_type").size() 
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axs[0, 0].set_title('Fuel')
    axs[0, 0].bar(fuel_data.index.to_list(), fuel_data.to_list())
    axs[0, 1].set_title('Owner')
    axs[0, 1].bar(owner_data.to_list(), owner_data.to_list())   
    #axs[0, 1].scatter(top_10.index.to_list(), top_10.to_list())
    #axs[1, 0].plot(transmission_data.index.to_list(), transmission_data.to_list())
    axs[1, 0].set_title('Transmission')
    axs[1, 0].bar(transmission_data.index.to_list(), transmission_data.to_list())
    axs[1, 1].set_title('Seller')
    axs[1, 1].bar(seller_data.index.to_list(), seller_data.to_list())
    fig.suptitle('Data of year ' + str(input_year))
    tab.pyplot(fig)

# Builds different plots to top 10 brands (fuel, transmission, seller, )
def plotForTabParamYearSP(tab, input_year):
    tab.subheader('Top 10 Brands of year ' + str(input_year))
    tab.write('based on highest selling')
    tab.bar_chart(orig_cars_df.query('year == @input_year').groupby("brand").size().nlargest(10))
    tab.write('based on average selling price')
    tab.line_chart(orig_cars_df.query('year == @input_year').groupby("brand")['selling_price'].mean().nlargest(10))
    #tab.scatter_chart(orig_cars_df.query('year == @input_year').groupby("brand").size().nlargest(10))

# Builds different plots to another features (fuel, transmission, seller, )
def plotForTabParamYearAF(tab, input_year):
    tab.subheader('Top 10 Brands of year ' + str(input_year))
    tab.write('based on fuel type')
    tab.bar_chart(orig_cars_df.query('year == @input_year').groupby("fuel").size())
    tab.write('based on transmission type')
    tab.line_chart(orig_cars_df.query('year == @input_year').groupby("transmission").size())
    tab.write('based on seller type')
    tab.bar_chart(orig_cars_df.query('year == @input_year').groupby("seller_type").size())
    tab.write('based on owner type')
    tab.bar_chart(orig_cars_df.query('year == @input_year').groupby("owner").size())

def printPredictInputData(brand, model, model_variant, year, km_driven, engine_size, mileage, max_power, fuel, seller, transmission, owner, seats, pred_model):
    print('Passed data year ', year) 
    print('Passed data fuel ', fuel)
    print('Passed data seller_type ', seller) 
    print('Passed data transmission ', transmission)
    print('Passed data owner ', owner) 
    print('Passed data seats ', seats) 
    print('Passed data brand ', brand)
    print('Passed data model ', model) 
    print('Passed data model_variant ', model_variant)
    print('Passed data km_driven_bin ', km_driven)
    print('Passed data engine_bin ', engine_size)
    print('Passed data mileage_bin ', mileage)
    print('Passed data max_power_bin ', max_power)

def testPrediction():
    input_data_test = pd.DataFrame([[2017,1,1,0,0,5.0,2,6,559,38,8,1,1]],
                                columns=['year','fuel','seller_type','transmission','owner','seats',
                                         'brand','model','model_variants',
                                         'km_driven_bin','engine_bin','mileage_bin','max_power_bin'])

def predict_carprice(brand, model, model_variant, year, km_driven, engine_size, mileage, max_power, fuel, seller, transmission, owner, seats):     
    
    input_data = pd.DataFrame([[year, fuel, seller, transmission, owner, seats,
                                brand, model, model_variant, 
                                km_driven, engine_size, mileage, max_power]],
                                 columns=['year','fuel','seller_type','transmission','owner','seats',
                                         'brand','model','model_variants',
                                         'km_driven_bin','engine_bin','mileage_bin','max_power_bin'])
    
    predictionResult = dict()
    for curModel in models:
        if curModel == 'XGB':
            predictModel = getXGBModel(st.session_state['testPredictions'], st.session_state['modelsPredictMetrics']) 
            prediction = predictModel.predict(input_data)
            predictionResult[models[curModel]] = prediction[0]
        elif curModel == 'RFR':
            predictModel = getRFRModel(st.session_state['testPredictions'], st.session_state['modelsPredictMetrics'])
            prediction = predictModel.predict(input_data)
            predictionResult[models[curModel]] = prediction[0]
        elif curModel == 'Bagging':
            predictModel = getBaggingModel(st.session_state['testPredictions'], st.session_state['modelsPredictMetrics'])
            prediction = predictModel.predict(input_data)
            predictionResult[models[curModel]] = prediction[0]
        else:
            predictModel = getETRModel(st.session_state['testPredictions'], st.session_state['modelsPredictMetrics'])
            prediction = predictModel.predict(input_data)
            predictionResult[models[curModel]] = prediction[0]

    return predictionResult

def printDataForPrediction(tab, brand, model, model_var, year):
    tab.subheader('Data based on dataset:')
     # selection parameters 
    container = tab.container(border=True)
    col1, col2, col3 = container.columns(3)

    year_list = sorted(cars_df_labels.year.unique().tolist())
    year_list.insert(0, 0)
    brand_list = sorted(cars_df_labels.brand.unique().tolist())
    brand_list.insert(0, 'Select')
    with col1:
        brand_ix = brand_list.index(brand)
        brand_param = col1.selectbox("Selected brand", brand_list, index=brand_ix, disabled=True)
        year_param = col1.selectbox("Select year:", year_list, key='year_dataset')  
    with col2:
        model_list = sorted(cars_df_labels.query('brand == @brand_param')['model'].unique().tolist())
        model_ix = model_list.index(model)
        model_param = col2.selectbox("Selected model", model_list, index=model_ix, disabled=True)

    with col3:
        model_var_list = sorted(cars_df_labels.query('model == @model_param')['model_variants'].unique().tolist())
        model_var_id = model_var_list.index(model_var)
        model_var_param = col3.selectbox('Selected model variant', model_var_list, index=model_var_id, disabled=True)      

    if brand != 'Select':
        data = orig_cars_df.query("brand == @brand and model == @model and model_variants == @model_var")
    if brand != 'Select' and year != 0:
        data = orig_cars_df.query("brand == @brand and model == @model and model_variants == @model_var and year == @year_param")
    #pd.options.display.float_format = '${:,.2f}'.format
    data['year'] = data['year'].map(str)
    tab.write(data)  
    tab.subheader('Used Cars Price Prediction')
    #tab.write('Models comparison, visualization the relationship between the actual car prices and the prices predicted by our models, providing insights into their accuracy and generalization ability.')
    # TODO
    tab.write('Prediction data to be displayed...')

def lineGraph_prediction(n, y_actual, y_predicted):
    y = y_actual
    y_total = y_predicted
    number = n
    aa=[x for x in range(number)]
    fig = plt.figure(figsize=(25,10)) 
    plt.plot(aa, y[:number], marker='.', label="actual")
    plt.plot(aa, y_total[:number], 'b', label="prediction")
    plt.xlabel('Price prediction of first {} used cars'.format(number), size=15)
    plt.legend(fontsize=15)
    #plt.show()

def plotline_prediction(tab, n, y_actual, y_predicted):
    #y = y_actual
    #y_total = y_predicted
    number = n
    #aa=[x for x in range(number)] 
    data = [Y_test.values[:number], y_predicted[:number]]
    #group_labels = ['actual', 'prediction']
    tab.plotly_chart(px.line(data), use_container_width=True)

def refreshModels(tab):
    testPredictions = st.session_state['testPredictions']
    modelsPredictMetrics = st.session_state['modelsPredictMetrics']
    
    tab.subheader('Prediction accuracy for models by R2 / MSE criterions')
    pd.options.display.float_format = '${:,.2f}'.format

    prediction_metrics_df = pd.DataFrame(modelsPredictMetrics, columns=(['Model', 'r2 Score', 'MSE']))
    #tab.dataframe(prediction_df.style.highlight_max(axis=0))
    sorted_metrics_df = prediction_metrics_df.sort_values(by='r2 Score', ascending=False)
    tab.dataframe(sorted_metrics_df, hide_index=True, width=1200)
    #my_table = tab.table(prediction_df.sort_values(by='r2 Score', ascending=False))

    modelsKeys = testPredictions.keys()
    print(modelsKeys)
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharey=True)
    axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0]]
    ind = 0
    
    #for key in testPredictions.keys():
    #    axs_list[ind].set_title("by " + key)
    #    axs_list[ind].set_xlabel("Actual price")
    #   axs_list[ind].set_ylabel("Predicted price")
    #    axs_list[ind].scatter(Y_test, Y_test, color='blue', label='Actual Selling Price')
    #    axs_list[ind].scatter(Y_test, testPredictions[key], color='green',label='Predicted Selling Price')
    #   ind = ind + 1

    tab.subheader(':blue[Actual prices] vs. :green[predicted prices]')
    line_selected = tab.checkbox("Use line charts)")
    if not line_selected:
        for key in testPredictions.keys():
            axs_list[ind].set_title("by " + key)
            axs_list[ind].set_xlabel("Actual price")
            axs_list[ind].set_ylabel("Predicted price")
            axs_list[ind].scatter(Y_test, Y_test, color='blue', label='Actual Selling Price')
            axs_list[ind].scatter(Y_test, testPredictions[key], color='green',label='Predicted Selling Price')
            ind = ind + 1
        tab.pyplot(fig)
    else:
        #ind = 0
        for key in testPredictions.keys():
            plotline_prediction(tab, 50, Y_test.values, testPredictions[key])          
            #axs_list[ind].set_title("by " + key)
            #axs_list[ind].set_xlabel("Actual price")
            #axs_list[ind].set_ylabel("Predicted price")
            #axs_list[ind].line(Y_test, Y_test, color='blue', label='Actual Selling Price')
            #axs_list[ind].line(Y_test, testPredictions[key], color='green',label='Predicted Selling Price')
            #ind = ind + 1
        #tab.pyplot(fig)
            #data = [Y_test.values, testPredictions[key]]
            #group_labels = ['Group 1', 'Group 2']
            #tab.plotly_chart(px.line(data), use_container_width=True)
            
            #lineGraph_prediction(50, Y_test.values, testPredictions[key])  
            #axs_list[ind].set_title("by " + key)
            #axs_list[ind].set_xlabel("Actual price")
            #axs_list[ind].set_ylabel("Predicted price")
            #data = pd.DataFrame(Y_test.values, testPredictions[key], columns=["a", "b"])
            #tab.line_chart(Y_test.values)
            #tab.line_chart(testPredictions[key])
            #tab.line_chart(Y_test, Y_test, color='blue', label='Actual Selling Price')
            #tab.line_chart(Y_test, testPredictions[key], color='green',label='Predicted Selling Price')
            #ind = ind + 1

def printDataBrands(tab, subbrand, year, selected_list, selected_plot):
    tab.subheader('Brands data:')
    #
    if selected_list[0]:
        if subbrand != 'Select':
            chart_data = orig_cars_df.query("brand == @subbrand")['km_driven']
        if subbrand != 'Select' and year != 0:
            chart_data = orig_cars_df.query("brand == @subbrand and year == @year")['km_driven']
        tab.scatter_chart(chart_data)

def buildMappingData(tab):
    container = tab.container(border=True)
    col1, col2, col3 = container.columns(3)
    tab.write('Mapping data')
    with col1:
        col1.write(mappings[0])
    with col2:
        col2.write(mappings[5])
    with col3:
        col3.write(mappings[6])

@st.experimental_fragment
def selectCurrency(tab, predictionResult, predictionResultEuro):
    if predictionResult:      
        currency = tab.radio("Select currency (Exchange rate 1 INR = 0.01110 EUR)", ["₹ INR", "€ EUR"])
        tab.success(f'Result of prediction:')
        print("selected currency ", currency)
        predictionsMessage = ''
        if currency.endswith('INR'):           
            for key in predictionResult.keys():
                predictionsMessage = predictionsMessage + f'{key}:  {predictionResult[key]}\n' + '\r\n'
        else:
            for key in predictionResultEuro.keys():
                predictionsMessage = predictionsMessage + f'{key}:  {predictionResultEuro[key]}\n' + '\r\n'    
        tab.success(predictionsMessage)

def frStrKey(strKey):
    tNmb = 25
    return strKey + (' ' * (tNmb - len(strKey))) + ':'

def vizualizePredictionResults(tab, predictionResult):
    modelsPredictMetrics = st.session_state['modelsPredictMetrics']

    prediction_metrics_df = pd.DataFrame(modelsPredictMetrics, columns=(['Model', 'r2 Score', 'MSE']))
    sorted_metrics_df = prediction_metrics_df.sort_values(by='r2 Score', ascending=False)
    print('sorted_metrics_df')
    print(sorted_metrics_df)
    print('Model ', sorted_metrics_df.head(1)['Model'])
    best_model_val = sorted_metrics_df.head(1)['Model'].values[0]
    print('best_model_val  ', best_model_val)

    if predictionResult:   
        helpMessage = ''
        for key in predictionResult.keys():
            formatted1 = "₹  {0:.2f}".format(predictionResult[key][0])
            formatted2 = "€  {0:.2f}".format(predictionResult[key][1])
            helpMessage = helpMessage + frStrKey(key) + f'{formatted1}   ( {formatted2} )\n'  + '\r\n'           
        tab.toggle("Show all prediction variants", help=helpMessage, disabled=True)
        tab.success(f'Result of prediction:')
        predictionsMessage = ''
       
        for key in predictionResult.keys():
            if best_model_val == key:
                formatted2 = "€  {0:.2f}".format(predictionResult[key][1])
                formatted1 = "₹  {0:.2f}".format(predictionResult[key][0])
                predictionsMessage = predictionsMessage + frStrKey(key) + f'{formatted1}   ( {formatted2} )\n'  + '\r\n'
       
        tab.success(predictionsMessage)


def main():
    st.sidebar.image("CarPricePrediction.jpg")
    
    brand_v = st.sidebar.selectbox('Brand / Manufacturer', sorted(mappings[0].keys()))

    model_v = st.sidebar.selectbox('Car model', sorted(cars_df_labels.query('brand == @brand_v')['model'].unique().tolist()))
    model_var_v = st.sidebar.selectbox('Car model variant', sorted(cars_df_labels.query('model == @model_v')['model_variants'].unique().tolist()))
    year_v = st.sidebar.selectbox('Year', sorted(cars_df_labels.year.unique().tolist()))

    kmdriven_list = [x for x in cars_df_labels.km_driven_bin.unique().tolist() if type(x) == str]
    kmdriven_v = st.sidebar.selectbox('Driven km', sorted(kmdriven_list, key=lambda x: int(x.split()[0])))
    engine_list = [x for x in cars_df_labels.engine_bin.unique().tolist() if type(x) == str]
    engine_v = st.sidebar.selectbox('Engine size', sorted(engine_list, key=lambda x: int(x.split()[0])))
    mileage_list = [x for x in cars_df_labels.mileage_bin.unique().tolist() if type(x) == str]
    mileage_v = st.sidebar.selectbox('Mileage (kilometres for one litre of fuel)', sorted(mileage_list, key=lambda x: int(x.split()[0])))
    max_power_list = [x for x in cars_df_labels.max_power_bin.unique().tolist() if type(x) == str]
    max_power_v = st.sidebar.selectbox('Max power', sorted(max_power_list, key=lambda x: int(x.split()[0])))
    seats_v = st.sidebar.selectbox('Seats', sorted(cars_df.seats.unique().tolist()))

    transmission_v = st.sidebar.select_slider('Transmission', cars_df_labels.transmission.unique().tolist())
    fuel_v = st.sidebar.select_slider('Fuel', cars_df_labels.fuel.unique().tolist())
    owner_v = st.sidebar.select_slider('Owner', cars_df_labels.owner.unique().tolist())
    seller_v = st.sidebar.select_slider('Seller_type', cars_df_labels.seller_type.unique().tolist())

    st.image('Used_cars.png') 
    selectCurrency(None, None, None)
    # selection parameters 
    container = st.container(border=True)
    col1, col2 = container.columns(2)
    with col1:
        year_list = sorted(cars_df_labels.year.unique().tolist())
        year_list.insert(0, 0) 
        year_param = col1.selectbox("Generate data for year:", year_list, key='year_param')
        
    labels = ["Dataset", "Models", "Brands data"]
    if year_param != 0:
        labels.append(str(year_param) + '  sp')
        labels.append(str(year_param) + '  af')

    #
    initModels()

    tabslist = st.tabs(labels)
  
    for label, tab in zip(labels, tabslist):
        if label.endswith('sp'):     
            plotForTabParamYearSP(tab, year_param)
        elif label.endswith('af'):
            plotForTabParamYearAF(tab, year_param)
        elif label.endswith('test'):
            plotForTabParam(tab, year_param)    
        elif label == "Dataset":  
            dataset_tab = tab           
            printDataForPrediction(tab, brand_v, model_v, model_var_v, year_param)
        elif label == "Models":  
            refreshModels(tab)
        elif label == "Brands data": 
            container = tab.container(border=True)
            col1, col2, col3 = container.columns(3)
            with col1:          
                brand_sel = col1.selectbox("Select brand", sorted(mappings[0].keys()))
                year_sel = col1.selectbox("Select year:", year_list, key='year_sel')
            with col2: 
                selected_list = list()
                driven_km_sel = col2.checkbox("Driven kms")
                selected_list.append(driven_km_sel)
                engine_sel = col2.checkbox("Engine size")
                selected_list.append(engine_sel)
                mileage_sel = col2.checkbox("Mileage")
                selected_list.append(mileage_sel)
                max_power_sel = col2.checkbox("Max power")
                selected_list.append(max_power_sel)
            with col3: 
                selected_plot = col3.radio("Select plot type", ["Line", "Box", "Scatter"])

            printDataBrands(tab, brand_sel, year_sel, selected_list, selected_plot)        
        else:
            plotForTab(tab, 'Charts for '+label, label, ["selling_price"])
    #
   # currency = st.sidebar.radio("Select currency (Exchange rate 1 INR = 0.01110 EUR)", ["₹ INR", "€ EUR"])

    if st.sidebar.button("Predict car price"):
        predictionResult = predict_carprice(mappings[0].get(brand_v), mappings[1].get(model_v), mappings[2].get(model_var_v), year_v, 
                                  mappings[7].get(kmdriven_v), mappings[8].get(engine_v), mappings[9].get(mileage_v), mappings[10].get(max_power_v),
                                  mappings[5].get(fuel_v), mappings[3].get(seller_v), mappings[4].get(transmission_v), mappings[6].get(owner_v), seats_v)
        
        predictionResultEuro = dict()
        for key in predictionResult.keys():
            predictionResultEuro[key] = predictionResult[key] * 0.01110
            values_list = list()
            values_list.append(predictionResult[key])
            values_list.append(predictionResult[key] * 0.01110)
            predictionResult[key] = values_list 
        #
        #selectCurrency(dataset_tab, predictionResult, predictionResultEuro)
        vizualizePredictionResults(dataset_tab, predictionResult)
if __name__=='__main__':
    main()