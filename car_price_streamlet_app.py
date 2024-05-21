import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

cars_df = pd.read_csv('data/cars_after_eda.csv')

columns = ['brand', 'model', 'seller_type', 'transmission', 'fuel', 'owner', 'model_variants']

model = pickle.load(open('data/trained_pipeline_cars_xgb.pkl','rb'))

def predict_carprice(variance, skewness, curtosis, entropy):

    input = np.array([[variance, skewness, curtosis, entropy]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)

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

def plotForTabParam(tab, headn:str, x, y):
    tab.subheader(headn)
    tab.bar_chart(cars_df.query('year == @x'), x='fuel', y=y)
    tab.scatter_chart(cars_df.query('year == @x'), x='brand', y=y, color=['#FF0000'])

def main():
    st.sidebar.title("Car Price ")
    
    brandv = st.sidebar.selectbox('Brand / Manufacturer', sorted(cars_df.brand.unique().tolist()))
    modelv = st.sidebar.selectbox('Car model', sorted(cars_df.query('brand == @brandv')['model'].unique().tolist()))
    yearv = st.sidebar.selectbox('Year', sorted(cars_df.year.unique().tolist()))
    #

    kmdrivenv = st.sidebar.selectbox('Driven km', sorted([x for x in cars_df.km_driven_bin.unique().tolist() if type(x) == str], key=lambda x: int(x.split()[0])))
    enginev = st.sidebar.selectbox('Engine size', sorted([x for x in cars_df.engine_bin.unique().tolist() if type(x) == str]))
    mileage_v = st.sidebar.selectbox('Mileage (kilometres for one litre of fuel)', sorted([x for x in cars_df.mileage_bin.unique().tolist() if type(x) == str]))

    st.sidebar.select_slider('Transmission', cars_df.transmission.unique().tolist())
    st.sidebar.select_slider('Fuel', cars_df.fuel.unique().tolist())
    st.sidebar.select_slider('Owner', cars_df.owner.unique().tolist())
    st.sidebar.select_slider('Seller_type', cars_df.seller_type.unique().tolist())

    st.image('Used_cars.png') 
   
    # selection parameters 
    container = st.container(border=True)
    col1, col2, col3 = container.columns(3)

    year_list = sorted(cars_df.year.unique().tolist())
    year_list.insert(0, 0)
    brand_list = sorted(cars_df.brand.unique().tolist())
    brand_list.insert(0, 'Select')
    with col1:
        year_param = col1.selectbox("Select year:", year_list)
        brand_param = col1.selectbox("Select band:", brand_list)
    with col2:
        transmission_list = ['unknown'] + cars_df.transmission.unique().tolist()
        transmission_param = col2.radio("Transmission", transmission_list)
    with col3:
        fuel_list = ['unknown'] + cars_df.fuel.unique().tolist()
        fuel_param = col3.radio("Fuel", fuel_list)   
    #

    labels = ["year", "fuel", "owner", "seller_type"]
    if year_param != 0:
        labels.append(str(year_param) + '  sp')

    #
    tabslist = st.tabs(labels)
  
    for label, tab in zip(labels, tabslist):
        if label.endswith('sp'):
            plotForTabParam(tab, 'Info for '+label.split()[0], label.split()[0], ["selling_price"])      
        else:
            plotForTab(tab, 'Charts for '+label, label, ["selling_price"])
    #
    



    if st.sidebar.button("Predict car price"):
        output = predict_carprice(variance, skewness, curtosis, entropy)
        st.sidebar.success(f'Result: {output}.')
        st.sidebar.write('1 = banknote is genuine, 0 = banknote is forged')

if __name__=='__main__':
    main()