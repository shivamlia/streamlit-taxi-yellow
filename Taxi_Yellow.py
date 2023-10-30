import pickle
import streamlit as st

model = pickle.load(open('Taxi_Yellow.sav', 'rb'))

st.title('Yellow Taxi Trip')

col1, col2 = st.columns(2)

with col1:
    passenger_count = st.number_input('passenger_count(Jumlah penumpang)')
    trip_distance = st.number_input('trip_distance (Jarak perjalanan) ')
    PULocationID = st.number_input('PULocationID (Zona taximeter digunakan)')
    DOLocationID = st.number_input('DOLocationID (Zona taxi meter dilepas)')

with col2:
    payment_type = st.number_input('payment_type (Jenis pembayaran)')
    fare_amount = st.number_input('fare_amount (Tarif pembayaran)')
    mta_tax = st.number_input('mta_tax (Pajak MTA)')
    tip_amount = st.number_input('tip_amount (Jumlah tip)')

predict = ''

if st.button('Estimasi Trip Yellow Taxi'):
  predict = model.predict(
      [[passenger_count,	trip_distance, PULocationID,	DOLocationID,	payment_type,	fare_amount, mta_tax,	tip_amount]]

  )
  st.write ('Estimasi Harga Trip Yellow Taxi dalam dolar $: ', predict)
  st.write ('Estimasi Harga Trip Yellow Taxi IDR (Juta) :', predict*19000)
