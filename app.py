import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlem
import streamlit as st

def main():
    data = pd.read_csv("https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv")

    data = data.iloc[:-1 , :]
    data = data.astype(float)
    # print(data.describe())
    ###########################################################################################################
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.title('Prova Machine Learning 5 Maggio 2023')

    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k,v in data.items():
        sns.distplot(v, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    #####################################################################
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.distplot(data['medv'], bins=30)
    st.pyplot(fig)

    #######**********#CORRELATION MATRIX****** ########
    data.corr()
    correlation_matrix = data.corr()
    correlation_matrix
    sns.heatmap(data=correlation_matrix, annot=True)



    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), ax=ax, annot=True)
    st.write(fig)

if __name__ == '__main__':
    main()