import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


reference_data = datasets.load_iris(as_frame=True).frame
#current_data = pd.read_csv('prediction_database.csv',usecols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'prediction'])

report = Report(metrics=[DataDriftPreset()])
report.run(
    current_data=reference_data.iloc[:60],
    reference_data=reference_data.iloc[60:],
    column_mapping=None,
)
report.save_html('report.html')
