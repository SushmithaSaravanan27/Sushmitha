from zipfile import ZipFile
with ZipFile('/content/archive.zip', 'r') as Z:
  Z.extractall("/content/archive")
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-2-85c423d171bd> in <cell line: 2>()
      1 from zipfile import ZipFile
----> 2 with ZipFile('/content/archive.zip', 'r') as Z:
      3   Z.extractall("/content/archive")

/usr/lib/python3.10/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps)
   1252             while True:
   1253                 try:
-> 1254                     self.fp = io.open(file, filemode)
   1255                 except OSError:
   1256                     if filemode in modeDict:

FileNotFoundError: [Errno 2] No such file or directory: '/content/archive.zip'
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
data = pd.read_csv("/content/archive/healthcare-dataset-stroke-data.csv")
data.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['work_type'] = le.fit_transform(data['work_type'])
data['work_type'].unique()
import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
gender_counts = data['gender'].value_counts()
custom_colors = ['#9C9C9C' , '#252E6C',  '#893F3D']

fig1 = px.bar(gender_counts, x=gender_counts.index, y=gender_counts.values, color=gender_counts.index,
              color_discrete_map={gender: color for gender, color in zip(gender_counts.index, custom_colors)})
fig1.update_layout(title='Gender Distribution')
fig1.show()
age_stroke_counts = data.groupby(["age", "stroke"]).size().reset_index(name="Count")
fig_bubble_age_stroke = px.scatter(age_stroke_counts, x="age", y="Count", size="Count", color="stroke",
                                   title="Distribution of Stroke Across Age Groups",
                                   labels={"age": "Age", "Count": "Count", "stroke": "Stroke"},
                                   color_discrete_sequence=['#252E6C', '#BC3030'] )

fig_bubble_age_stroke.update_layout(xaxis_title="Age", yaxis_title="Count")
fig_bubble_age_stroke.show()
stroke_counts = data['stroke'].value_counts().reset_index()
stroke_counts.columns = ['Stroke', 'Count']
custom_colors = ['#252E6C', '#BC3030']
fig = px.pie(stroke_counts, names='Stroke', values='Count',
             title='Distribution of Stroke Status',
             color_discrete_sequence=custom_colors,
             hole=0.3
             )
fig.show()
gender_stroke_counts = data.groupby(['gender', 'stroke']).size().unstack()
gender_stroke_percentage = (gender_stroke_counts[1] / (gender_stroke_counts[0] + gender_stroke_counts[1])) * 100

fig = px.pie(names=gender_stroke_percentage.index, values=gender_stroke_percentage.values,
             title="Percentage of Stroke Cases by Gender",
             color_discrete_sequence=['#252E6C', '#9C9C9C'])

fig.update_traces(textinfo="percent+label", pull=[0.1, 0], marker=dict(line=dict(color="white", width=2)))
fig.show()
hypertension_count = data['hypertension'].value_counts()
custom_colors = ['#252E6C', '#9C9C9C']

fig = px.pie(
    values=hypertension_count,
    names=hypertension_count.index,
    hole=0.3,
    title='Distribution of Patients with and without Hypertension',
    color_discrete_sequence=custom_colors,
)

fig.update_traces(textinfo='percent+label', pull=[0, 0.1])
fig.show()
stroke_rate_with_hypertension = (data[data['hypertension'] == 1]['stroke'].mean()) * 100
stroke_rate_without_hypertension = (data[data['hypertension'] == 0]['stroke'].mean()) * 100

data_1 = pd.DataFrame({'Hypertension': ['With Hypertension', 'Without Hypertension'],
                     'Stroke Rate': [stroke_rate_with_hypertension, stroke_rate_without_hypertension]})
fig = px.bar(data_1, x='Hypertension', y='Stroke Rate',
             text='Stroke Rate', title='Stroke Rate by Hypertension',
             labels={'Hypertension': 'Hypertension Status', 'Stroke Rate': 'Stroke Rate (%)'})

fig.update_traces(marker_color=[ "#123F6A" ,  "#89AED2"])
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.show()
smoking_status_counts = data['smoking_status'].value_counts().reset_index()
smoking_status_counts.columns = ['smoking_status', 'count']
custom_colors = ['#0E2B59', '#355384', '#7793BE']

fig = px.bar(
    smoking_status_counts,
    x='smoking_status',
    y='count',
    title='Smoking Status Distribution',
    color='smoking_status',
    color_discrete_sequence=custom_colors
)

fig.show()
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix,  f1_score, precision_score, recall_score
from sklearn import model_selection
import numpy as np
def model_scores(model, x, y):
  F1_score = []
  precision = []
  recall = []
  error = []
  kappa = []
  def classification_report_with_accuracy_score(y_true, y_pred):
    F1_score.append(f1_score(y_true, y_pred, average='micro'))
    precision.append(precision_score(y_true, y_pred, average = 'micro'))
    recall.append(recall_score(y_true, y_pred, average = 'micro'))
    error.append(1-accuracy_score(y_true, y_pred))
    return accuracy_score(y_true, y_pred)

  kfold = model_selection.KFold(n_splits=5,shuffle = True)
  results = model_selection.cross_val_score(model, x, y, cv = kfold, \
                scoring=make_scorer(classification_report_with_accuracy_score))
  print('Accuracy :', np.average(results)*100)
  print('recall : ', np.average(recall)*100)
  print('Precision : ', np.average(precision)*100)
  print('f1_score :', np.average(F1_score)*100)
  print('error : ', np.average(error)*100)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,HistGradientBoostingClassifier,GradientBoostingClassifier, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier

vc = VotingClassifier(estimators = [('abc', AdaBoostClassifier()),('ets', ExtraTreesClassifier()),('rf',RandomForestClassifier()),('dt',DecisionTreeClassifier()), ('gbc', GradientBoostingClassifier()),('hgb', HistGradientBoostingClassifier()),('knn',KNeighborsClassifier()), ('svc',SVC(probability=True)), ('nb',GaussianNB()), ('lr',LogisticRegression())],voting='soft')
vc.fit(X_sm, y_sm)
