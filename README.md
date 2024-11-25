<h1>Coronary Heart Disease Prediction</h1>
This is the Jupyter Notebook and the Dataset for the mentioned Classification Predictive Modeling


## About the dataset:

<p align="justify">
Data yang digunakan dalam penelitian ini adalah Starter: Cardiac data NHANES yang diambil dari Kaggle, yang berisi 52 atribut dan 37.079 catatan. NHANES (National Health and Nutrition Examination Survey) adalah survei yang dilakukan oleh National Center Health Statistics (NCHS). NCHS adalah bagian dari Centers for Disease Control and Prevention (CDC) di Amerika Serikat untuk memahami kondisi kesehatan dan gizi penduduk Amerika Serikat. Pada penelitian ini menggunakan data NHANES yang diperoleh dari tahun 1999-2000 hingga 2017-2018. Data diperoleh dalam tiga kategori data demografi, data pemeriksaan dan data laboratorium. Variabel demografi mencakup usia dan jenis kelamin partisipan survei pada saat pemeriksaan. Berat badan partisipan, tekanan darah dan indeks massa tubuh (BMI) dari data pemeriksaan juga dianggap sebagai seperangkat variabel faktor risiko untuk mempelajari pengaruhnya terhadap penyakit kardiovaskular.
</p>

## Objective: To build a classification model that classify Coronary Heart Disease in a subject.


## I have performed the following steps: 
1. Read the file and displayed its columns.
2. Handled missing values and Outliers.
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

## Outlier Detection with IQR
<p align="justify">
Dengan penggunaan Boxplot untuk memvisualisasikan outlier pada data. Berikut adalah boxplot dari variabel numerik yang ada pada dataframe, ditampilkan pada Gambar dibawah.
</p>
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/Outlier.png" alt="Outlier" width="800">


## Handling Outlier with Median
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/Handling%20Outlier%20with%20median.png" alt="Handling Outlier with Median" width="800">
<p align="justify">
Gambar diatas menampilkan boxplot dengan Q1, Q3, IQR, dan whisker yang menunjukkan lower bound dan upper bound. Data yang berada di luar whisker akan dianggap sebagai outlier. Sebagian besar variabel memiliki sejumlah outlier yang signifikan, beberapa di antaranya sangat jauh dari whisker. Ini menunjukkan bahwa ada nilai yang sangat tinggi atau sangat rendah yang tidak sesuai dengan distribusi utama data. Cara yang digunakan penulis untuk mengatasi data outlier adalah dengan melakukan imputasi, yaitu proses penggantian nilai outlier dengan nilai lain dan bukan menghapus nilai outlier karena jumlah outlier tidaklah sedikit. Jika outlier sangat ekstrim dan mempengaruhi rata-rata secara signifikan, maka mengganti outlier median adalah pilihan yang lebih baik. Langkah ini membantu mengurangi pengaruh outlier, namun dampak dari penggunaanya adalah terjadinya perubahan pada distribusi data.
Setelah mengganti outlier ekstrem dengan median, jumlah outlier yang signifikan telah berkurang drastis. Beberapa variabel masih menunjukkan sedikit outlier, tetapi mereka tidak lagi se-ekstrim sebelumnya. Data tersebut sekarang lebih sedikit dan lebih dekat ke distribusi utama data seperti yang terlihat pada Gambar diatas.
</p>

## SMOTE untuk Resampling
<p align="justify">
SMOTE bekerja dengan cara mengambil secara acak tetangga terdekat sebanyak k dari setiap instance dalam kelas minoritas kemudian membuat instance baru (sintetis) antara instance tersebut dengan tetangga terdekat k yang dipilih secara acak seperti yang ditampilkan pada Gambar dibawah. Dengan pendekatan SMOTE maka dapat dipastikan tidak terjadi masalah duplikasi data sehingga lebih kebal terhadap masalah overfitting. algoritma memilih sampel acak dari kelas minoritas dan memilih tetangga acak menggunakan K-Nearest Neighbors.
</p>
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/Resampling.png" alt="Resampling" width="800">
<p align="justify">
Sintetis pada SMOTE mengindikasikan bahwa tidak adanya duplikasi data pada data yang diresample. SMOTE meningkatkan sampel kelas minoritas sambil menghindari overfitting. Hal ini dilakukan dengan menghasilkan contoh sintetik baru yang dekat dengan titik lain (yang termasuk dalam kelas minoritas) dalam ruang fitur.
</p>

## Matrix Korelasi
<p align="justify">
Adapun cara untuk mengetahui data mana yang penting adalah dengan menganalisis nilai korelasinya. Untuk mempermudah proses analisis, dapat digunakan Heatmap Visualization. Berikut adalah Heatmap Visualization, seperti yang ditunjukkan pada Gambar dibawah. Metode ini menghitung koefisien korelasi antara kolom dalam DataFrame, yang membantu dalam mengidentifikasi hubungan antara variabel. Nilai korelasi berkisar angtara -1 sampai 1. Jika bernilai negatif maka data memiliki korelasi negatif, dan sebaliknya. Namun, jika nilai korelasi mendekati nol, artinya data tersebut hampir tidak memiliki korelasi atau korelasinya rendah, sehingga data tersebut bukan merupakan data yang penting. Variabel dengan korelasi rendah dapat mengganggu akurasi model.
</p>
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/Heatmap%20Visualization.png" alt="heatmap visualization" width="1000">

## Eksperimen secara empirik dan eksperimental
### Eksperimen Awal (Tanpa Hyperparameter Tuning)
<p align="justify">
Pada tahap awal eksperimen, model Gradient Boosting digunakan tanpa melakukan tuning hyperparameter. Hal ini bertujuan untuk memperoleh baseline performa dari model dengan menggunakan nilai default yang disediakan oleh library scikit-learn versi 1.02. Model ini diterapkan tanpa pendekatan sistematis dalam pengaturan hyperparameter, sehingga hasil yang diperoleh akan menjadi acuan untuk perbandingan dengan eksperimen selanjutnya yang melibatkan proses tuning hyperparameter.
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/model%20dengan%20hyperparameter%20default.png" alt="hyperparameter default" width="800">
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/confussion%20matrix.png" alt="confussion matrix" width="400">
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/AUC-ROC%20Curve.png" alt="auc roc curve" width="400"> 
</p>

### Setelah Menggunakan TPE
<p align="justify">
Setelah melakukan tuning hyperparameter menggunakan metode Tree-Structured Parzen Estimator (TPE), hyperparameter model dioptimalkan untuk meningkatkan performa secara signifikan. TPE memungkinkan pencarian hyperparameter yang lebih efisien dengan mengarahkan pencarian ke area yang lebih menjanjikan dalam ruang pencarian, berdasarkan hasil dari trial sebelumnya.
Setelah melakukan tuning hyperparameter menggunakan metode Tree-structured Parzen Estimator (TPE), hyperparameter dioptimalkan untuk meningkatkan performa model. Hasil eksperimen menunjukkan peningkatan akurasi model secara signifikan dengan waktu komputasi yang lebih rendah, serta pengurangan risiko overfitting karena optimisasi yang lebih presisi dalam pemilihan jumlah pohon keputusan (estimasi), kedalaman pohon, dan parameter lainnya.
</p>
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/model%20setelah%20dituning.png" alt="model setelah tuning" width="800">


### Hasil Eksperimen Pengoptimalan Hyperparameter
<p align="justify">
Dari eksperimen yang telah dilakukan, penulis membuat evaluasi dengan menggunakan metode TPE. Pemilihan hyperparameter dilakukan dengan menjalankan search trials untuk TPE. Penulis memilih jumlah trials sebanyak 100, parameter ini yang akan menentukan berapa kali algoritma optimasi akan mencoba kombinasi hyperparameter yang berbeda. Setiap evaluasi berarti satu set hyperparameter dipilih dari search space, kemudian model dilatih dan dievaluasi menggunakan set hyperparameter tersebut. Hasil eksperimen yang dilakukan pada model Gradient Boosting dengan framework hyperopt menggunakan 27 fitur terpilih, data train sebanyak 56.914 data, data test sebanyak 14.229 data.
</p>
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/TPE%20Search%20Trials.png" alt="model setelah tuning" width="800">
<p align="justify">
Berdasarkan kinerja model dari Gambar diatas, menujukkan bahwa sebagian besar trials menunjukkan accuracy yang sangat tinggi, ada beberapa trials yang menghasilkan accuracy yang jauh lebih rendah. Hal ini terjadi karena hyperparameter yang dipilih pada trial tersebut kurang optimal. Fluktuasi yang terlihat pada accuracy menunjukkan bahwa TPE sedang menjelajahi berbagai bagian dari search space hyperparameter, beberapa di antaranya menghasilkan accuracy yang lebih rendah. Namun, sebagian besar accuracy cukup stabil dan tinggi, menunjukkan bahwa model bekerja dengan baik pada banyak kombinasi hyperparameter. Trial dengan accuracy rendah disebabkan oleh kombinasi hyperparameter yang sangat tidak cocok untuk dataset ini. Penting untuk memeriksa hyperparameter yang dihasilkan oleh trial ini untuk menghindari konfigurasi serupa di masa depan. Gambar 21 memberikan gambaran yang baik tentang bagaimana TPE melakukan eksplorasi dalam ruang hyperparameter dan bagaimana performa model berfluktuasi selama proses tersebut. Berdasarkan trials diatas diketahui bahwa trials ke 77 adalah yang terbaik dengan test accuracy sebesar 0.9829 dan time computation 14310.94 seconds.
</p>

### Gradient Boosting Tree Visualization
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/gb_tree_visualization.png" alt="gb_tree_visualization">

1. **Root Node**:
    - Root node memulai proses pembagian dataset.
    - Pohon dibangun dengan memisahkan dataset berdasarkan fitur dengan pengurangan impuritas tertinggi (Gini Impurity).

2. **Intermediate Nodes** (Simpul Tengah/Cabang):
    - Memecah subset data dari root node ke subset yang lebih kecil berdasarkan fitur lain.
    - Mungkin memiliki dua cabang keluar:
        - **Cabang kiri**: Untuk nilai fitur yang memenuhi kondisi (<= threshold).
        - **Cabang kanan**: Untuk nilai fitur yang tidak memenuhi kondisi (> threshold).
          
3. **Leaf Nodes** (Simpul Daun):
    - Posisi: Node paling bawah tanpa cabang keluar.
    - Fungsi: Simpul daun adalah tempat di mana keputusan akhir dibuat.
    - Karakteristik:
        - Menampilkan jumlah sampel yang ada dalam node tersebut.
        - Memberikan prediksi akhir (label kelas atau nilai prediksi untuk regresi).
     
4. **Branch** (Cabang):
    - Pohon terlihat simetris karena setiap node diperiksa untuk kondisi tertentu (misalnya, feature_x <= threshold), dan kedua cabang (kiri dan kanan) terus dipecah hingga memenuhi kriteria penghentian (stopping criteria).
    - Parameter seperti max_depth (kedalaman maksimum pohon) membatasi seberapa dalam pohon dapat berkembang. Dalam pohon ini, kedalaman maksimum ditentukan oleh hyperparameter (max_depth=9), sehingga pohon tidak akan berkembang lebih dari 9 level.
    - Karena setiap simpul memecah data berdasarkan fitur tertentu, pohon memiliki banyak node intermediate sebelum mencapai leaf nodes. Ini mencerminkan banyaknya aturan logis yang diterapkan untuk membuat prediksi.

5. **Gradient Boosting Membatasi Overfitting**
    - Pohon ini relatif dangkal dibandingkan pohon keputusan konvensional karena Gradient Boosting menggunakan beberapa pohon kecil (weak learners) untuk membentuk model akhir.
    - Kedalaman yang terbatas mengurangi risiko overfitting sambil tetap menangkap pola-pola penting.

6. **Node Tampak Padat**
    - Karena Gradient Boosting membangun model secara iteratif, pohon ini mungkin tidak sepenuhnya mengurangi impuritas pada level terdalamnya.
    - Beberapa cabang mungkin tampak lebih pendek karena data pada cabang tersebut tidak lagi memenuhi kriteria pemisahan lebih lanjut.


### Faktor Utama yang Membentuk Struktur Pohon
1. **Parameter Model**:
   - max_depth: Mengontrol kedalaman maksimum pohon.
   - min_samples_split dan min_samples_leaf: Mengontrol jumlah sampel minimum untuk pemisahan dan leaf nodes.
   - max_features: Membatasi jumlah fitur yang diperiksa pada setiap pemisahan.
2. **Data**:
   - Distribusi dataset memengaruhi bagaimana pohon dibentuk.
   - Data dengan variasi tinggi cenderung menghasilkan pohon yang lebih bercabang.
3. **Loss Function**:
   - Untuk klasifikasi, Gradient Boosting sering menggunakan Log Loss, yang mengarahkan pohon untuk memfokuskan pada sampel yang sulit diklasifikasikan.
4. **Impuritas Node**:
   - Pemisahan pada setiap node ditentukan berdasarkan pengurangan impuritas terbesar (Gini).


### Evaluasi Kinerja Model yang diusulkan
Confussion matrix merinci hasil prediksi yang dilakukan oleh model terhadap data uji, dengan memperlihatkan jumlah prediksi yang benar dan salah yang dibuat oleh model. Sedangkan, untuk kurva AUC-ROC adalah grafik yang digunakan untuk mengevaluasi kinerja model klasifikasi biner pada berbagai threshold keputusan. Evaluasi dengan beberapa metrik dapat memberikan pandangan yang lebih menyeluruh tentang performa model dan menggunakan satu metrik saja bisa menyebabkan kesimpulan yang bias atau tidak akurat.
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/confussion%20matrix.png" alt="confussion matrix" width="400">
<img src="https://github.com/emilhrdnsy/emil-thesis-repo/blob/master/AUC-ROC%20Curve.png" alt="auc roc curve" width="400"> 







