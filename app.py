import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

st.title('Mammographic Masses Classification')

# 1. Load and clean data
data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'], names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
st.write('### Raw Data', data.head())
st.write('Missing values per column:', data.isnull().sum())

# Drop rows with missing data
data = data.dropna()
st.write('### Cleaned Data', data.describe())

# 2. Feature selection
features = ['age', 'shape', 'margin', 'density']
X = data[features].values
y = data['severity'].values

# 3. Feature scaling
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Model selection
st.sidebar.header('Model Selection')
model_name = st.sidebar.selectbox('Choose model', ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'Logistic Regression'])

params = {}
if model_name == 'KNN':
    k = st.sidebar.slider('Number of neighbors (K)', 1, 50, 10)
    params['n_neighbors'] = k
if model_name == 'SVM':
    kernel = st.sidebar.selectbox('Kernel', ['linear', 'rbf', 'sigmoid', 'poly'])
    params['kernel'] = kernel
    C = st.sidebar.slider('C (Regularization)', 0.01, 10.0, 1.0)
    params['C'] = C

# 5. Model instantiation
if model_name == 'Decision Tree':
    clf = DecisionTreeClassifier(random_state=1)
    X_model = X_scaled
elif model_name == 'Random Forest':
    clf = RandomForestClassifier(n_estimators=10, random_state=1)
    X_model = X_scaled
elif model_name == 'SVM':
    clf = svm.SVC(kernel=params['kernel'], C=params['C'])
    X_model = X_scaled
elif model_name == 'KNN':
    clf = neighbors.KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    X_model = X_scaled
elif model_name == 'Naive Bayes':
    scaler_nb = preprocessing.MinMaxScaler()
    X_nb = scaler_nb.fit_transform(X)
    clf = MultinomialNB()
    X_model = X_nb
elif model_name == 'Logistic Regression':
    clf = LogisticRegression()
    X_model = X_scaled

# 6. Cross-validation
cv_scores = cross_val_score(clf, X_model, y, cv=10)
st.write(f'### {model_name} Mean CV Accuracy:', np.round(cv_scores.mean(), 3))

# 7. Compare all models
if st.checkbox('Compare all models'):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=1),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=1),
        'SVM (linear)': svm.SVC(kernel='linear', C=1.0),
        'KNN (k=10)': neighbors.KNeighborsClassifier(n_neighbors=10),
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression()
    }
    scores = []
    for name, model in models.items():
        if name == 'Naive Bayes':
            scaler_nb = preprocessing.MinMaxScaler()
            X_nb = scaler_nb.fit_transform(X)
            score = cross_val_score(model, X_nb, y, cv=10).mean()
        else:
            score = cross_val_score(model, X_scaled, y, cv=10).mean()
        scores.append(score)
    st.bar_chart(pd.DataFrame({'Model': list(models.keys()), 'Accuracy': scores}).set_index('Model'))

# 8. Prediction demo
st.write('---')
st.write('## Try a Prediction')
age = st.number_input('Age', min_value=18, max_value=100, value=50)
shape = st.selectbox('Shape', [1, 2, 3, 4], format_func=lambda x: ['round', 'oval', 'lobular', 'irregular'][x-1])
margin = st.selectbox('Margin', [1, 2, 3, 4, 5], format_func=lambda x: ['circumscribed', 'microlobulated', 'obscured', 'ill-defined', 'spiculated'][x-1])
density = st.selectbox('Density', [1, 2, 3, 4], format_func=lambda x: ['high', 'iso', 'low', 'fat-containing'][x-1])

input_features = np.array([[age, shape, margin, density]])
if model_name == 'Naive Bayes':
    input_features_scaled = preprocessing.MinMaxScaler().fit(X).transform(input_features)
else:
    input_features_scaled = scaler.transform(input_features)

if st.button('Predict'):
    clf.fit(X_model, y)
    pred = clf.predict(input_features_scaled)[0]
    st.write('### Prediction:', 'Malignant' if pred == 1 else 'Benign') 