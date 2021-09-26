import dalex as dx

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

data = dx.datasets.load_titanic()

X = data.drop(columns='survived')
y = data.survived

data.head(10)

numerical_features = ['age', 'fare', 'sibsp', 'parch']
numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categorical_features = ['gender', 'class', 'embarked']
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500, random_state=0)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])

clf.fit(X, y)

exp = dx.Explainer(clf, X, y)

john = pd.DataFrame({'gender': ['male'],
                       'age': [25],
                       'class': ['1st'],
                       'embarked': ['Southampton'],
                       'fare': [72],
                       'sibsp': [0],
                       'parch': 0},
                      index = ['John'])

mary = pd.DataFrame({'gender': ['female'],
                     'age': [35],
                     'class': ['3rd'],
                     'embarked': ['Cherbourg'],
                     'fare': [25],
                     'sibsp': [0],
                     'parch': [0]},
                     index = ['Mary'])

exp.predict(X)[0:10]
exp.predict(john)
exp.predict(mary)

bd_john = exp.predict_parts(john, type='break_down', label=john.index[0])
bd_interactions_john = exp.predict_parts(john, type='break_down_interactions', label="John+")

sh_mary = exp.predict_parts(mary, type='shap', B = 10, label=mary.index[0])

bd_john.result

bd_john.plot(bd_interactions_john)

sh_mary.result.loc[sh_mary.result.B == 0, ]

sh_mary.plot(bar_width = 16)

exp.predict_parts(john, type='shap', B = 10, label=john.index[0]).plot(max_vars=5)

cp_mary = exp.predict_profile(mary, label=mary.index[0])
cp_john = exp.predict_profile(john, label=john.index[0])

cp_mary.result.head()

cp_mary.plot(cp_john)

cp_john.plot(cp_mary, variable_type = "categorical")

mp = exp.model_performance(model_type = 'classification')
mp.result

mp.result.auc[0]

mp.plot(geom="roc")

    vi = exp.model_parts()
vi.result

vi.plot(max_vars=5)

vi_grouped = exp.model_parts(variable_groups={'personal': ['gender', 'age', 'sibsp', 'parch'],
                                     'wealth': ['class', 'fare']})
vi_grouped.result

vi_grouped.plot()

pdp_num = exp.model_profile(type = 'partial', label="pdp")

ale_num = exp.model_profile(type = 'accumulated', label="ale")

pdp_num.plot(ale_num)

pdp_cat = exp.model_profile(type = 'partial', variable_type='categorical',
                            variables = ["gender","class"], label="pdp")

ale_cat = exp.model_profile(type = 'accumulated', variable_type='categorical',
                            variables = ["gender","class"], label="ale")

ale_cat.plot(pdp_cat)

