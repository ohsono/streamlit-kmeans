import streamlit as st
import requests
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

st.title('Iris k-means clustering')

# load the iris dataset
# load the iris dataset with sklearn
dataset_iris = load_iris()

x= pd.DataFrame(dataset_iris.data, columns=dataset_iris.feature_names)
feature_names = dataset_iris.feature_names

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

st.sidebar.header('Input Features')
n_clusters = st.sidebar.slider("Cluster count", min_value=1, max_value=10, value=3)
sepal_length = st.sidebar.slider('Pick a Sepal Length', 4.0, 8.0)
petal_width = st.sidebar.slider('Pick a Petal Width', 0.0, 2.5)

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(x)

# Create DataFrame for plotting
df_plot = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = clusters.astype(str)  # Convert to string for categorical coloring

# Plot
fig = px.scatter(
    df_plot, x="PC1", y="PC2", color="Cluster",
    labels={"Cluster": "Cluster ID"},
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

#Convert to float
sepal_length=float(sepal_length)
petal_width = float(petal_width)

input_data = {
            "petal_width":petal_width, 
             "sepal_length":sepal_length
            
              }  
#st.write(input_data)
response = requests.post('https://plumber-app-521739183727.us-central1.run.app/predict_petal_length', json=input_data, headers = {"content-type":"application/json"})
#st.write(response.json())
prediction = response.json()
prediction = list(prediction.values())[0]
st.write(f"The Predicted Petal Length for your selected Sepal Length and Petal Width is: {prediction}")
