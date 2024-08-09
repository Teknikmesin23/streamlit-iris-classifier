import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Memuat dataset iris
iris = load_iris()
X = iris.data
y = iris.target

# Melatih model RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Antarmuka pengguna dengan Streamlit
st.title("Klasifikasi Bunga Iris")
st.write("Masukkan fitur-fitur bunga untuk memprediksi spesiesnya:")

sepal_length = st.slider("Panjang Sepal", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.slider("Lebar Sepal", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.slider("Panjang Petal", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.slider("Lebar Petal", min_value=0.0, max_value=10.0, value=1.0)

# Menyusun fitur yang dimasukkan pengguna
features = [[sepal_length, sepal_width, petal_length, petal_width]]

# Membuat prediksi
prediction = model.predict(features)
prediction_proba = model.predict_proba(features)

# Menampilkan hasil prediksi
st.write(f"Prediksi: **{iris.target_names[prediction[0]]}**")
st.write("Probabilitas prediksi:")
st.write(f"- Setosa: {prediction_proba[0][0]:.2f}")
st.write(f"- Versicolor: {prediction_proba[0][1]:.2f}")
st.write(f"- Virginica: {prediction_proba[0][2]:.2f}")
