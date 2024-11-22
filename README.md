<h1 align="center">Coronary Heart Disease Prediction</h1>
This is the Jupyter Notebook and the Dataset for the mentioned Classification Predictive Modeling


### About the dataset:

Data yang digunakan dalam penelitian ini adalah Starter: Cardiac data NHANES yang diambil dari Kaggle, yang berisi 52 atribut dan 37.079 catatan. NHANES (National Health and Nutrition Examination Survey) adalah survei yang dilakukan oleh National Center Health Statistics (NCHS). NCHS adalah bagian dari Centers for Disease Control and Prevention (CDC) di Amerika Serikat untuk memahami kondisi kesehatan dan gizi penduduk Amerika Serikat. Pada penelitian ini menggunakan data NHANES yang diperoleh dari tahun 1999-2000 hingga 2017-2018. Data diperoleh dalam tiga kategori data demografi, data pemeriksaan dan data laboratorium. Variabel demografi mencakup usia dan jenis kelamin partisipan survei pada saat pemeriksaan. Berat badan partisipan, tekanan darah dan indeks massa tubuh (BMI) dari data pemeriksaan juga dianggap sebagai seperangkat variabel faktor risiko untuk mempelajari pengaruhnya terhadap penyakit kardiovaskular.

### Objective: To build a classification model that classify Coronary Heart Disease in a subject.

### I have performed the following steps: 

1. Read the file and displayed its columns.
2. Handled missing values, Outliers and Duplicate Data.
3. Calculated basic statistics of the data (count, mean, std, etc), did exploratory analysis and described my observations.
4. Resampled the imbalanced dataset by oversampling the positive cases.
5. Selected columns that will probably be important to predict heart disease.
6. Created training and testing sets (using 60% of the data for the training and reminder for testing) and scaled the data using MinMaxScaler.
7. Built 5 different machine learning models to predict TenYearCHD:
    *  Logistic Regression - 67.56% Accuracy
    *  kNN Classification - 89.98% Accuracy
    *  Random Forest Classification - 90.39% Accuracy
    *  Decision Tree Classification - 86.66% Accuracy
    *  Gradient Boosting Classification - 71.46% Accuracy
8. Hyperparameter tuned the RandomForestClassification - 95.77% and GradientBoostingClassification - 95.21%.
9. Evaluated each model (f1 score, Accuracy, Precision ,Recall and Confusion Matrix) and plotted a graph for the false positive rate and true positive rate for each model.
10. Ensembled the four best models using Stacking technique to further increase the accuracy of the model and achieved an accuracy score of 96.17%
11. Concluded that Ensembling all the four most important models, with Random Forest Classification leading the way, has resulted in a very high accuracy score.



