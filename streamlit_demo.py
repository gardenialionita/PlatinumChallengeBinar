import streamlit as st
import requests, re
import pandas as pd
import numpy as np
import sqlite3
import json

db_conn = sqlite3.connect('predict_data.db', check_same_thread=False)
db_conn.text_factory = bytes
mycursor = db_conn.cursor()
table_name = 'master_data'



def cleansing(sent):
    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()
    string = sent.lower()
    # Menghapus emoticon dan tanda baca menggunakan "RegEx" dengan script di bawah
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

def call_api(text, path):
    url = f"http://127.0.0.1:5555/{path}/v1"
    data_payload = {
        "text":text
    }
    response = requests.post(url,json=data_payload)
    result = response.json()

    return result


def predict_text():
    st.subheader("Predict Data from Input Text")

    input_text = st.text_input('Masukkan kalimat dalam Bahasa Indonesia')
    predict_button = st.button('predict')

    if  predict_button:
        if option=='LSTM':
            result = call_api(input_text,'predict_sentiment_lstm')
            st.write(result)

        elif option=='ANN':
            result = call_api(input_text,'predict_sentiment')
            st.write(result)

        else:
            st.write('Terjadi kesalahan, silakan refresh halaman.')

def predict_file():
    st.subheader("Predict Data from File")
    
    data_file = st.file_uploader("Upload CSV",type=['csv'])

    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.write(file_details)
        data_csv = pd.read_csv(data_file,encoding = "latin-1")


        st.write('Preview Data')
        st.dataframe(data_csv.iloc[:, 0])
        predictfile_button = st.button('predict file')

        query_text = 'delete from master_data'
        mycursor.execute(query_text)
        db_conn.commit()

        first_column = data_csv.iloc[:, 0]
        for input_text in first_column:

            data_clean = cleansing(input_text)
            if  predictfile_button:
                if option=='LSTM':
                    result = call_api(input_text,'predict_sentiment_lstm')
                    result = json.dumps(result['result_sentiment'])
                    insert_tweet = 'insert into master_data (data_raw, data_clean, data_sentiment) values(?, ?, ?)'
                    value = (input_text, data_clean, result)
                    mycursor.execute(insert_tweet, value)
                    db_conn.commit()
                elif option=='ANN':
                    result = call_api(input_text,'predict_sentiment_lstm')
                    result = json.dumps(result['result_sentiment'])
                    insert_tweet = 'insert into master_data (data_raw, data_clean, data_sentiment) values(?, ?, ?)'
                    value = (input_text, data_clean, result)
                    mycursor.execute(insert_tweet, value)
                    db_conn.commit()
                else:
                    st.write('Belum pilih model, silakan coba lagi')

        query_text = 'select * from master_data'
        select_data = mycursor.execute(query_text)
        show_data = [
            dict(data_id=row[0], data_raw=row[1], data_clean= row[2], data_sentiment = row[3])
            for row in select_data.fetchall()
        ]

        st.write(show_data)


st.title('Predict Sentiment Analysis in Bahasa Indonesia')
st.image("assets/sentiment-analysis.png",width=300)
option = st.selectbox(
'pakai model apa?',
('LSTM', 'ANN'))
predict_text()
predict_file()

st.markdown('Contributor:')
st.write('Adinda Reyna Maulidia | Gardenia Lionita | Muhamad Thoriq')

