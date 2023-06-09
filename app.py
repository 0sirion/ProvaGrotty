import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlem
import streamlit as st
import seaborn as sns

def main():
    url="https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv"
    data=pd.read_csv(url)
    data=data.iloc[:-1]
    data=data.apply(pd.to_numeric)
    
    fig=plt.figure(figsize=(10,8))
    plt.title('case')
    sns.heatmap(data.corr(),annot=True , cmap="Blues")
    st.pyplot(fig)

#########################################################
    crim=st.number_input('inserisci crim',1,10000,500)
    zn=st.number_input('inserisci zn',1,10000,500)
    indus=st.number_input('inserisci indus',1,10000,500)
    chas=st.number_input('inserisci indus',0,10,0)
    nox=st.number_input('inserisci nox',0.0,1.0,0.5)
    rm=st.number_input('inserisci rm',1,10,5)
    age=st.number_input('inserisci age',18,100,30)
    dis=st.number_input('inserisci dis',1,10,2)
    rad=st.number_input('inserisci rad',1,3,1)
    tax=st.number_input('inserisci tax',1,500,100)
    ptratio=st.number_input('inserisci ptratio',1,50,10)
    b=st.number_input('inserisci b',1,300,100)
    

    lstat=st.number_input('inserisci istat',1,10,5)
    load_model=mlem.api.load('model_.mlem')
    pred=load_model.predict([[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat,]])
    st.write(f"il prezzo é: euro{round(pred[0],2)}")

if __name__ == '__main__':
    main()