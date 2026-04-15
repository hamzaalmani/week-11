import streamlit as st
import numpy as np
from apputil import kmeans

st.title("K-Means Demo")

k = st.slider("Number of clusters (k)", 2, 10, 3)

# random sample data
X = np.random.rand(100, 2)

centroids, labels = kmeans(X, k)

st.write("Centroids:")
st.write(centroids)

st.write("Cluster labels (first 10):")
st.write(labels[:10])