import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
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

#from sklearn.pipeline import Pipeline

#pipe = Pipeline([('XGBRegressor', xgb)])
#pipe.fit(X_train, Y_train)


columns = ['brand', 'model', 'seller_type', 'transmission', 'fuel', 'owner', 'model_variants']

models = {'XGB': 'Extra gradient boosting', 'RFR': 'Random forest regression', 'Bagging': 'BaggingRegressor', 'ETR': 'ExtraTreesRegressor'}

#model = pickle.load(open('data/trained_pipeline_cars_xgb.pkl','rb'))



@st.cache_resource
def getXGBModel():
    # Create a XGB model and train created model.
    xgb = XGBRegressor()
    xgb.fit(X_train, Y_train)
    xgbPredicts = xgb.predict(X_test)
    xgbScore = r2_score(Y_test, xgbPredicts)
    xgbError = np.sqrt(mean_squared_error(Y_test, xgbPredicts))
    print('R2-Score XGB ', xgbScore)
    print('Error square XGB ', xgbError)
    return xgb

@st.cache_resource
def getRFRModel():
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train)
    rfrPredicts = rfr.predict(X_test)
    rfrScore = r2_score(Y_test, rfrPredicts)
    rfrError = np.sqrt(mean_squared_error(Y_test, rfrPredicts))
    print('R2-Score RFR ', rfrScore)
    print('Error square RFR ', rfrError)
    return rfr

@st.cache_resource
def getBaggingModel():
    bagging = BaggingRegressor()
    bagging.fit(X_train, Y_train)
    baggingPredicts = bagging.predict(X_test)
    baggingScore = r2_score(Y_test, baggingPredicts)
    baggingError = np.sqrt(mean_squared_error(Y_test, baggingPredicts))
    print('R2-Score Bagging ', baggingScore)
    print('Error square Bagging ', baggingError)
    return bagging

def getETRModel():
    etr = ExtraTreesRegressor()
    etr.fit(X_train, Y_train)
    etrPredicts = etr.predict(X_test)
    etrScore = r2_score(Y_test, etrPredicts)
    etrError = np.sqrt(mean_squared_error(Y_test, etrPredicts))
    print('R2-Score ExtraTreesRegressor ', etrScore)
    print('Error square ExtraTreesRegressor ', etrError)
    return etr

def plotForTabEcht(tab, headn:str, x, y):
    tab.subheader(headn)
    tab.line_chart(cars_df, x=x, y=y, color=["#FF0000"])
    tab.bar_chart(cars_df, x=x, y=y)

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


def plotForTabTest(tab, headn:str, x, y):
    imgs = [np.random.random((50,50)) for _ in range(4)]
    fig1 = plt.figure(figsize = (3,3))
    plt.subplot(2, 2, 1)
    plt.imshow(imgs[0])
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(imgs[1])
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(imgs[2])
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(imgs[3])
    plt.axis('off')
    plt.subplots_adjust(wspace=.025, hspace=.025)
    plt.savefig('x',dpi=400)
    tab.image('x.png')
    tab.pyplot(fig1)
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

def plotForTabParamYearSP(tab, input_year):
    tab.subheader('Top 10 Brands of year ' + str(input_year))
    tab.write('based on highest selling')
    tab.bar_chart(orig_cars_df.query('year == @input_year').groupby("brand").size().nlargest(10))
    tab.write('based on average selling price')
    tab.line_chart(orig_cars_df.query('year == @input_year').groupby("brand")['selling_price'].mean().nlargest(10))
    #tab.scatter_chart(orig_cars_df.query('year == @input_year').groupby("brand").size().nlargest(10))

def plotForTabParamYearAF(tab, input_year):
    tab.subheader('Top 10 Brands of year ' + str(input_year))
    tab.write('based on fuel type')
    tab.bar_chart(orig_cars_df.query('year == @input_year').groupby("fuel").size())
    tab.write('based on transmission type')
    tab.line_chart(orig_cars_df.query('year == @input_year').groupby("transmission").size())

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

def predict_carprice(brand, model, model_variant, year, km_driven, engine_size, mileage, max_power, fuel, seller, transmission, owner, seats, pred_model):     
    
    input_data = pd.DataFrame([[year, fuel, seller, transmission, owner, seats,
                                brand, model, model_variant, 
                                km_driven, engine_size, mileage, max_power]],
                                 columns=['year','fuel','seller_type','transmission','owner','seats',
                                         'brand','model','model_variants',
                                         'km_driven_bin','engine_bin','mileage_bin','max_power_bin'])
    
    print('Passed prediction model', pred_model)
    if pred_model == 'XGB':
        predictModel = getXGBModel() 
        prediction = predictModel.predict(input_data)
    elif pred_model == 'RFR':
        predictModel = getRFRModel()
        prediction = predictModel.predict(input_data)
    elif pred_model == 'Bagging':
        predictModel = getBaggingModel()
        prediction = predictModel.predict(input_data)
    else:
        predictModel = getETRModel()
        prediction = predictModel.predict(input_data)
    return int(prediction)

def printDataset(tab, brand, model, model_var, year):
    tab.subheader('Data:')
    data = 'no content'
    if brand != 'Select':
        data = orig_cars_df.query("brand == @brand and model == @model and model_variants == @model_var")
    if brand != 'Select' and year != 0:
        data = orig_cars_df.query("brand == @brand and model == @model and model_variants and year == @year")
    tab.write(data)
    

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

def main():
    st.sidebar.title("Car Price Prediction")
    
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
   
    # selection parameters 
    container = st.container(border=True)
    col1, col2, col3 = container.columns(3)

    year_list = sorted(cars_df_labels.year.unique().tolist())
    year_list.insert(0, 0)
    brand_list = sorted(cars_df_labels.brand.unique().tolist())
    brand_list.insert(0, 'Select')
    with col1:
        brand_ix = brand_list.index(brand_v)
        brand_param = col1.selectbox("Selected brand", brand_list, index=brand_ix, disabled=True)
        #year_param = col1.selectbox("Select year:", year_list)
        
    with col2:
        model_list = sorted(cars_df_labels.query('brand == @brand_param')['model'].unique().tolist())
        model_ix = model_list.index(model_v)
        model_param = col2.selectbox("Selected model", model_list, index=model_ix, disabled=True)
       
    with col3:
        model_var_list = sorted(cars_df_labels.query('model == @model_param')['model_variants'].unique().tolist())
        model_var_id = model_var_list.index(model_var_v)
        model_var_param = col3.selectbox('Selected model variant', model_var_list, index=model_var_id, disabled=True)      
    #

    container = st.container(border=True)
    col4, col5 = container.columns(2)
    with col4:          
        brand_sel = col4.selectbox("Select brand", brand_list)
        year_param = col4.selectbox("Select year:", year_list)
    with col5: 
        driven_km_sel = col5.checkbox("Driven kms")
        engine_sel = col5.checkbox("Engine size")
        mileage_sel = col5.checkbox("Mileage")
        max_power_sel = col5.checkbox("Max power")
    labels = ["Dataset"]
    if year_param != 0:
        labels.append(str(year_param) + '  sp')
        labels.append(str(year_param) + '  af')
        labels.append(str(year_param) + '  test')

    #
    tabslist = st.tabs(labels)
  
    for label, tab in zip(labels, tabslist):
        if label.endswith('sp'):     
            plotForTabParamYearSP(tab, year_param)
        elif label.endswith('af'):
            plotForTabParamYearAF(tab, year_param)
        elif label.endswith('test'):
            plotForTabParam(tab, year_param)    
        elif label == "Dataset":  
            printDataset(tab, brand_param, model_param, model_var_param, year_param)
             # selection parameters             
        else:
            plotForTab(tab, 'Charts for '+label, label, ["selling_price"])
    #
    
    prediction_model_v = st.sidebar.selectbox("Select prediction model", models.values())

    if st.sidebar.button("Predict car price"):

        output = predict_carprice(mappings[0].get(brand_v), mappings[1].get(model_v), mappings[2].get(model_var_v), year_v, 
                                  mappings[7].get(kmdriven_v), mappings[8].get(engine_v), mappings[9].get(mileage_v), mappings[10].get(max_power_v),
                                  mappings[5].get(fuel_v), mappings[3].get(seller_v), mappings[4].get(transmission_v), mappings[6].get(owner_v), 
                                  seats_v, list(models.keys())[list(models.values()).index(prediction_model_v)])
        st.sidebar.success(f'Result: {output}.')
        st.sidebar.write('predicted by ', prediction_model_v)

if __name__=='__main__':
    main()