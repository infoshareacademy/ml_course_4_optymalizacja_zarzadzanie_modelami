import pandas as pd
from help_function import load_model, predict_df, predict_user

model = load_model(path = r'C:\dane\PATRYK\kurs\ml_course\4_Optymalizacja_zarzadzanie_modelami\8_zarzadzanie_modelami\model_restaurant_revenue.joblib')
predict_user(model=model)
input("Kliknij dowolny klawisz, aby zakończyć")