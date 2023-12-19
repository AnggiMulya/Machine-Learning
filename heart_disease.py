#Library
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image

#Configurate the page
st.set_page_config(layout='wide',
                   page_title="Capstone Project DQlab",
                   initial_sidebar_state='expanded'
)

#Title


st.sidebar.title("Navigation")
add_sidebar_selectbox = st.sidebar.selectbox("Go to", ('Home', 'Dataset', 'Exploratory Data Analysis', 'Machine Learning', 'Prediction', 'Contact'))

def home():
    st.write('''
        # Heart Disease Machine Learning

        Halo, saya Ahmad Alfian Faisal. Ini adalah Deploying my Machine Learning Model untuk melengkapi Capstone Project Machine Learning and AI Beginner DQlab. Jika ada pertanyaan atau masukan, silakan hubungi saya di [Linkedin](www.linkedin.com/in/ahmadalfianfaisal)
        ''')
    st.write('''
        ## Cardiovascular Disease (CVDs)

        Cardiovascular disease (CVDs) atau penyakit jantung merupakan penyebab kematian nomor satu secara global dengan 17,9 juta kasus kematian setiap tahunnya. Penyakit jantung disebabkan oleh hipertensi, obesitas, dan gaya hidup yang tidak sehat. Deteksi dini penyakit jantung perlu dilakukan pada kelompok risiko tinggi agar dapat segera mendapatkan penanganan dan pencegahan. Sehingga tujuan bisnis yang ingin dicapai yaitu membentuk model prediksi penyakit jantung pada pasien berdasarkan feature-feature yang ada untuk membantu para dokter melakukan diagnosa secara tepat dan akurat. Harapannya agar penyakit jantung dapat ditangani lebih awal. Dengan demikian, diharapkan juga angka kematian akibat penyakit jantung dapat turun.
        ''')
    st.image("Cardiovascular disease (CVDs).png")


def dataset():
    st.header('Dataset')
    st.write('''
        ###### About Dataset

        Dataset yang digunakan dapat diunduh di UCI ML: https://archive.ics.uci.edu/dataset/45/heart+disease
        ''')
    st.write('''
    Dataset yang digunakan ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hungaria, Swiss, dan Long Beach V. Bidang "target" mengacu pada adanya penyakit jantung pada pasien. Ini adalah bilangan bulat bernilai 0 = tidak ada penyakit dan 1 = penyakit.

    Dataset heart disease terdiri dari 1025 baris data dan 13 atribut + 1 target. Dataset ini memiliki 14 kolom yaitu:
    1. `age`: variabel ini merepresentasikan usia pasien yang diukur dalam tahun.
    2. `sex`: variabel ini merepresentasikan jenis kelamin pasien dengan nilai 1 untuk laki-laki dan nilai 0 untuk perempuan.
    3. `cp` (Chest pain type): variabel ini merepresentasikan jenis nyeri dada yang dirasakan oleh pasien dengan 4 nilai kategori yang mungkin: nilai 1 mengindikasikan nyeri dada tipe angina, nilai 2 mengindikasikan nyeri dada tipe nyeri tidak stabil, nilai 3 mengindikasikan nyeri dada tipe nyeri tidak stabil yang parah, dan nilai 4 mengindikasikan nyeri dada yang tidak terkait dengan masalah jantung.
    4. `trestbps` (Resting blood pressure): variabel ini merepresentasikan tekanan darah pasien pada saat istirahat, diukur dalam mmHg (milimeter air raksa (merkuri)).
    5. `chol` (Serum cholestoral): variabel ini merepresentasikan kadar kolesterol serum dalam darah pasien, diukur dalam mg/dl (miligram per desiliter).
    6. `fbs` (Fasting blood sugar): variabel ini merepresentasikan kadar gula darah pasien saat puasa (belum makan) dengan nilai 1 jika kadar gula darah > 120 mg/dl dan nilai 0 jika tidak.
    7. `restecg` (Resting electrocardiographic results): variabel ini merepresentasikan hasil elektrokardiogram pasien saat istirahat dengan 3 nilai kategori yang mungkin: nilai 0 mengindikasikan hasil normal, nilai 1 mengindikasikan adanya kelainan gelombang ST-T, dan nilai 2 mengindikasikan hipertrofi ventrikel kiri.
    8. `thalach` (Maximum heart rate achieved): variabel ini merepresentasikan detak jantung maksimum yang dicapai oleh pasien selama tes olahraga, diukur dalam bpm (denyut per menit).
    9. `exang` (Exercise induced angina): variabel ini merepresentasikan apakah pasien mengalami angina (nyeri dada) yang dipicu oleh aktivitas olahraga, dengan nilai 1 jika ya dan nilai 0 jika tidak.
    10. `oldpeak`: variabel ini merepresentasikan seberapa banyak ST segmen menurun atau depresi saat melakukan aktivitas fisik dibandingkan saat istirahat.
    11. `slope`: variabel ini merepresentasikan kemiringan segmen ST pada elektrokardiogram (EKG) selama latihan fisik maksimal dengan 3 nilai kategori.
    12. `ca` (Number of major vessels): variabel ini merepresentasikan jumlah pembuluh darah utama (0-3) yang terlihat pada pemeriksaan flourosopi.
    13. `thal`: variabel ini merepresentasikan hasil tes thalium scan dengan 3 nilai kategori yang mungkin. Thal 1 menunjukkan kondisi normal, thal 2 menunjukkan adanya defek tetap pada thalassemia, thal 3 menunjukkan adanya defek yang dapat dipulihkan pada thalassemia
    14. `target`: 0 = tidak ada penyakit dan 1 = penyakit.
    ''')

    st.write('''
        ###### Import Dataset
        ''')
    st.code('''
        url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    data = pd.read_csv(url, names=column_names, skiprows=[0])

    # Menampilkan lima baris teratas
    data.head()
        ''')
    url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                    "ca", "thal", "target"]
    data = pd.read_csv(url, names=column_names, skiprows=[0])
    st.write('''
    **Dataset**
    ''')
    st.dataframe(data.head(10))

def exploratory_data_analysis():
    st.header('Exploratory Data Analysis')
    st.write('''
    ###### Import Library
    ''')
    st.code('''
import numpy as np
import pandas as pd
import math
import random
import seaborn as sns
from scipy.stats import pearsonr, jarque_bera
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
    ''')
    import numpy as np
    import pandas as pd
    import math
    import random
    import seaborn as sns
    from scipy.stats import pearsonr, jarque_bera
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings("ignore")

    st.write('''
    ###### Memuat dataset Heart Disease UCI ML
    ''')
    st.code('''
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names, skiprows=[0])
    ''')
    url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                    "ca", "thal", "target"]
    data = pd.read_csv(url, names=column_names, skiprows=[0])

    st.write('''
    ###### Menampilkan lima baris teratas
    ''')
    st.code('''
    data.head()
    ''')
    st.dataframe(data.head())

    st.write('''
    ###### Menampilkan lima baris terbawah
    ''')
    st.code('''
    data.tail()
    ''')
    st.dataframe(data.tail())

    st.write('''
    ###### Melihat jumlah baris dan kolom dataset
    ''')
    st.code('''data.shape''')
    data.shape


    st.write('''
    ###### Melihat Kolom dataset
    ''')
    st.code('''data.columns''')
    data.columns

    st.write('''
    ###### Melihat summary data
    ''')
    st.code('''
    data.describe()
    ''')
    st.dataframe(data.describe())

    st.write('''
    Melihat Informasi Data
    ''')
    tab1, tab2 = st.tabs(['Null_count', 'Dtypes'])

    with tab1:
        st.dataframe(data.isnull().sum())

    with tab2:
        st.dataframe(data.dtypes)
    st.write('''
    **Menghapus Data Duplikat**
    ''')
    st.write('''
    Jumlah Data Duplikat
    ''')
    st.write(data.duplicated().sum())
    st.write('''
    Shape sebelum menghapus data duplikat
    ''')
    data.shape
    data.drop_duplicates(inplace=True, keep='first')
    st.write('''
        Shape setelah menghapus data duplikat
        ''')
    data.shape
    st.write('''
    **Dataset Count Visualization**
    ''')
    st.write('''
    ###### Target
    ''')
    st.bar_chart(data.target.value_counts())
    st.write('''
    Kolom `target` merupakan kolom yang menunjukkan apakah seseorang terkena penyakit jantung atau tidak. Jika kolom 
    `target` bernilai 1 maka memiliki penyakit jantung dan jika kolom `target` bernilai 0 maka tidak memiliki penyakit jantung.
    Berdasarkan data di atas, jumlah orang yang memiliki penyakit jantung sebanyak 164 orang, sedangkan yang tidak memiliki penyakit jantung
    sebanyak 138 orang.
    ''')
    st.write('''
    ###### Sex
    ''')
    st.bar_chart(data.sex.value_counts())
    st.write('''
    Kolom `sex` merupakan kolom yang menunjukkan jenis kelamin dari tiap baris dataset. Jika kolom `sex` bernilai 0 maka perempuan dan jika kolom `sex` bernilai 1
    maka laki-laki.
    ''')
    st.write('''
    ###### Komposisi Terkena Penyakit Jantung Berdasarkan Sex
    ''')
    st.bar_chart(data.sex[data['target']==1].value_counts())
    st.write('''
    Dari visualisasi di atas, didapatkan kesimpulan bahwa dari 206 laki-laki, terdapat 92 yang terindikasi terkena penyakit jantung. 
    Sedangkan dari 96 perempuan, terdapat 72 yang terindikasi terkena penyakit jantung.
    ''')
    st.write('''
    ###### Age
    ''')
    st.bar_chart(data.age.value_counts())
    st.write('''
    Berdasarkan visualisasi di atas, yang paling banyak terkena penyakit jantung adalah di rentang umur 57-60 tahun.
    Sedangkan yang paling sedikit berada pada rentang 29-34 tahun dan 74-77 tahun.
    ''')




    tab3, tab4, tab5, tab6, tab7 = st.tabs(['Pengecekan Karakter','Handling Missing Value','Outlier', 'Distribusi Data', 'Korelasi'])

    with tab3:
        st.write('''
        ###### Melakukan handling kolom menjadi kategorikal
        ''')
        st.code('''
lst=['sex','cp','fbs','restecg','exang','slope','ca','thal','target']
data[lst] = data[lst].astype(object)
        ''')
        # Melakukan handling kolom menjadi kategorikal
        lst = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
        data[lst] = data[lst].astype(object)
        st.dataframe(data.dtypes)
        st.write('''
        ###### Pelabelan data categorical
        ''')
        data['sex'] = data['sex'].replace({1: 'Male',
                                           0: 'Female'})
        data['cp'] = data['cp'].replace({0: 'typical angina',
                                         1: 'atypical angina',
                                         2: 'non-anginal pain',
                                         3: 'asymtomatic'})
        data['fbs'] = data['fbs'].replace({0: 'No',
                                           1: 'Yes'})
        data['restecg'] = data['restecg'].replace({0: 'probable or definite left ventricular hypertrophy',
                                                   1: 'normal',
                                                   2: 'ST-T Wave abnormal'})
        data['exang'] = data['exang'].replace({0: 'No',
                                               1: 'Yes'})
        data['slope'] = data['slope'].replace({0: 'downsloping',
                                               1: 'flat',
                                               2: 'upsloping'})
        data['thal'] = data['thal'].replace({1: 'normal',
                                             2: 'fixed defect',
                                             3: 'reversable defect'})
        data['ca'] = data['ca'].replace({0: 'Number of major vessels: 0',
                                         1: 'Number of major vessels: 1',
                                         2: 'Number of major vessels: 2',
                                         3: 'Number of major vessels: 3'})
        data['target'] = data['target'].replace({0: 'No disease',
                                                 1: 'Disease'})
        st.code('''
        data['sex'] = data['sex'].replace({1: 'Male',
                                   0: 'Female'})
data['cp'] = data['cp'].replace({0: 'typical angina',
                                 1: 'atypical angina',
                                 2: 'non-anginal pain',
                                 3: 'asymtomatic'})
data['fbs'] = data['fbs'].replace({0: 'No',
                                   1: 'Yes'})
data['restecg'] = data['restecg'].replace({0: 'probable or definite left ventricular hypertrophy',
                                           1:'normal',
                                           2: 'ST-T Wave abnormal'})
data['exang'] = data['exang'].replace({0: 'No',
                                       1: 'Yes'})
data['slope'] = data['slope'].replace({0: 'downsloping',
                                       1: 'flat',
                                       2: 'upsloping'})
data['thal'] = data['thal'].replace({1: 'normal',
                                     2: 'fixed defect',
                                     3: 'reversable defect'})
data['ca'] = data['ca'].replace({0: 'Number of major vessels: 0',
                                 1: 'Number of major vessels: 1',
                                 2: 'Number of major vessels: 2',
                                 3: 'Number of major vessels: 3'})
data['target'] = data['target'].replace({0: 'No disease',
                                         1: 'Disease'})
        ''')
        st.write('''
        ###### Menampilkan data setelah pelabelan data kategorikal
        ''')
        st.dataframe(data)
        st.write('''
        Memisahkan data numerical dengan categorical
        ''')
        st.code('''
        numerical_col = data.select_dtypes(exclude='object_')
        categorical_col = data.select_dtypes(exclude='number')
        ''')
        numerical_col = data.select_dtypes(exclude='object_')
        categorical_col = data.select_dtypes(exclude='number')
        st.write('''
        ###### Pengecekan karakter dari data kategorikal
        ''')
        col1, col2 = st.columns(2)

        with col1:
            st.code('''
            data['ca'].value_counts()
            ''')
            st.dataframe(data['ca'].value_counts())
        with col2:
            st.code('''
            data['thal'].value_counts()
            ''')
            st.dataframe(data['thal'].value_counts())
        st.text('''
        Terdapat dua feature yang mengalami kesalahan penulisan:

Feature 'ca': Memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
Feature 'thal': Memiliki 4 nilai dari rentang 0-3, maka dari itu nilai 0 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')

        st.write('''
        ###### Melihat kolom 'ca' bernilai 4
        ''')
        st.code('''
        data[data['ca']==4]
        ''')
        data[data['ca'] == 4]
        st.write('''
        ###### Mengganti kolom 'ca' yang bernilai '4' menjadi NaN
        ''')
        st.code('''
        data.loc[data['ca']==4, 'ca'] = np.NaN
        ''')
        data.loc[data['ca'] == 4, 'ca'] = np.NaN
        st.write('''
        ###### Mengecek kembali kolom 'ca'
        ''')
        st.dataframe(data['ca'].value_counts())

        st.write('''
                ###### Melihat kolom 'thal' bernilai 0
                ''')
        st.code('''
                data[data['thal']==0]
                ''')
        data[data['thal'] == 0]
        st.write('''
                ###### Mengganti kolom 'thal' yang bernilai '0' menjadi NaN
                ''')
        st.code('''
                data.loc[data['thal']==0, 'thal'] = np.NaN
                ''')

        data.loc[data['thal'] == 0, 'thal'] = np.NaN

        st.write('''
                ###### Mengecek kembali kolom 'thal'
                ''')
        st.dataframe(data['thal'].value_counts())


        st.write('''
        ##### Mengecek kembali missing value
        ''')
        st.code('''
        data.isnull().sum()
        ''')
        st.dataframe(data.isnull().sum())


    with tab4:
        st.write('''
        ###### Check missing values
        ''')
        st.text('''
        Berdasarkan hasil pengecekan karakter, terdapat missing value yang terdapat pada kolom ca dan kolom thal. 
        Oleh sebab itu, akan dilakukan handling missing value.
        ''')
        st.dataframe(data.isnull().sum())
        st.text('''
        Kolom 'ca' dan 'thal' merupakan kolom kategorical, sehingga dilakukan pengisian missing value dengan modus.
        ''')

        st.write('''
        ###### Fillna pada kolom 'ca' dengan modus
        ''')
        st.code('''
        modus_ca = data['ca'].mode()[0]
        data['ca'] = data['ca'].fillna(modus_ca)
        ''')
        modus_ca = data['ca'].mode()[0]
        data['ca'] = data['ca'].fillna(modus_ca)

        st.write('''
        ###### Fillna pada kolom 'thal' dengan modus
        ''')
        st.code('''
        modus_thal = data['thal'].mode()[0]
        data['thal'] = data['thal'].fillna(modus_thal)
        ''')
        modus_thal = data['thal'].mode()[0]
        data['thal'] = data['thal'].fillna(modus_thal)

        st.write('''
        ###### Check Missing Value Kembali
        ''')
        st.code('''
        data.isnull().sum()
        ''')
        st.dataframe(data.isnull().sum())

    # with tab5:
    #     st.write('''
    #     ###### Jumlah Data Duplikat
    #     ''')
    #     st.text(data.duplicated().sum())
    #     st.code('''
    #     data.duplicated().sum()
    #     ''')
    #     st.write('''
    #     ###### Menghapus Data Duplikat
    #     ''')
    #     st.code('''
    #     data.drop_duplicates(inplace = True, keep = 'first')
    #     ''')
    #     data_duplicates = data.copy()
    #     data.drop_duplicates(inplace=True, keep='first')
    #     st.write('''
    #     ###### Jumlah kolom dan baris data setelah menghapus data duplikat
    #     ''')
    #     data.shape



    with tab5:
        st.write('''
        ###### Menampilkan dan Menghapus Data Outliers
        ''')
        st.code('''
                sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        sns.histplot(data=data, x='trestbps', kde=True, ax=axs[0, 0])
        sns.histplot(data=data, x='chol', kde=True, ax=axs[0, 1])
        sns.histplot(data=data, x='thalach', kde=True, ax=axs[1, 0])
        sns.histplot(data=data, x='oldpeak', kde=True, ax=axs[1, 1])

        plt.show()
        st.pyplot()
        ''')
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        sns.histplot(data=data, x='trestbps', kde=True, ax=axs[0, 0])
        sns.histplot(data=data, x='chol', kde=True, ax=axs[0, 1])
        sns.histplot(data=data, x='thalach', kde=True, ax=axs[1, 0])
        sns.histplot(data=data, x='oldpeak', kde=True, ax=axs[1, 1])

        plt.show()
        st.pyplot()

        st.text('''
        Dari visualisasi di atas, dapat dilihat bahwa terdapat kolom yang memiliki outliers,
        di antaranya 'trestbps', 'chol', 'thalach', dan 'oldpeak'
        ''')
        st.write('''
        ###### #Melakukan Transformasi data yang memiliki outliers
        ''')

        st.code(''' 
        cols_to_transform = ['trestbps', 'chol', 'thalach', 'oldpeak']
        for col in cols_to_transform:
            data[col] = np.log1p(data[col])''')
        cols_to_transform = ['trestbps', 'chol', 'thalach', 'oldpeak']
        for col in cols_to_transform:
            data[col] = np.log1p(data[col])

        st.code('''
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        sns.histplot(data=data, x='trestbps', kde=True, ax=axs[0, 0])
        sns.histplot(data=data, x='chol', kde=True, ax=axs[0, 1])
        sns.histplot(data=data, x='thalach', kde=True, ax=axs[1, 0])
        sns.histplot(data=data, x='oldpeak', kde=True, ax=axs[1, 1])
        ''')

        st.write('''
                ###### Menampilkan Data Setelah Transformasi
                ''')
        sns.set(style='whitegrid')
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        sns.histplot(data=data, x='trestbps', kde=True, ax=axs[0, 0])
        sns.histplot(data=data, x='chol', kde=True, ax=axs[0, 1])
        sns.histplot(data=data, x='thalach', kde=True, ax=axs[1, 0])
        sns.histplot(data=data, x='oldpeak', kde=True, ax=axs[1, 1])

        plt.show()
        st.pyplot()

    with tab6:
        st.write('''
        ###### Memvisualisasikan distribusi variabel kategorikal
        ''')
        plt.figure(figsize=(12, 12))
        for index, column in enumerate(categorical_col):
            plt.subplot(4, 3, index + 1)
            sns.countplot(data=categorical_col, x=column, hue='target', palette='magma')
            plt.xlabel(column.upper(), fontsize=14)
            plt.ylabel("count", fontsize=14)

        plt.tight_layout(pad=1.0)
        plt.show()
        st.pyplot()

        st.write('''
        ###### Memvisualisasikan distribusi variabel numerical
        ''')
        plt.figure(figsize=(16, 8))
        for index, column in enumerate(numerical_col):
            plt.subplot(2, 3, index + 1)
            sns.histplot(data=numerical_col, x=column, kde=True)
            plt.xticks(rotation=90)
        plt.tight_layout(pad=1.0)
        plt.show()
        st.pyplot()

    with tab7:
        st.write('''
        ###### melihat korelasi antar variable untuk mencari feature yang penting
        ''')
        data['sex'] = data['sex'].replace({'Male':1,
                                           'Female':0})
        data['cp'] = data['cp'].replace({'typical angina':0,
                                        'atypical angina':1,
                                        'non-anginal pain':2,
                                        'asymtomatic':3})
        data['fbs'] = data['fbs'].replace({'No':0,
                                           'Yes':1})
        data['restecg'] = data['restecg'].replace({'probable or definite left ventricular hypertrophy':0,
                                                   'normal':1,
                                                   'ST-T Wave abnormal':2})
        data['exang'] = data['exang'].replace({'No':0,
                                               'Yes':1})
        data['slope'] = data['slope'].replace({'downsloping':0,
                                               'flat':1,
                                               'upsloping':2})
        data['thal'] = data['thal'].replace({'normal':1,
                                             'fixed defect':2,
                                             'reversable defect':3})
        data['ca'] = data['ca'].replace({'Number of major vessels: 0':0,
                                         'Number of major vessels: 1':1,
                                         'Number of major vessels: 2':2,
                                         'Number of major vessels: 3':3})
        data['target'] = data['target'].replace({'No disease':0,
                                                 'Disease':1})
        data = data.astype('float64')

        plt.figure(figsize=(20, 20))
        cor = data.corr()
        sns.heatmap(cor, annot=True, linewidth=.5, cmap="magma")
        plt.title('Korelasi Antar Variable', fontsize=30)
        plt.show()
        st.pyplot()

        st.write('''
        ###### Menampilkan Korelasi Target terhadap Fitur yang Lain
        ''')
        cor_matrix = data.corr()
        st.dataframe(cor_matrix['target'].sort_values())

        st.write('''
    Korelasi target (penyakit jantung) dengan variabel lainnya. Korelasi positif dengan variabel tertentu berarti semakin tinggi variabel tersebut maka akan semakin tinggi juga kemungkinan terkena penyakit jantung, sedangkan korelasi negatif ialah semakin rendah nilai variabel tersebut maka kemungkinan terkena penyakit jantung lebih tinggi.

1. `ca` -0.456989 (Korelasi Negatif Kuat)
2. `oldpeak` -0.434108 (Korelasi Negatif Kuat)
3. `exang` -0.431599 (Korelasi Negatif Kuat)
4. `thal` -0.370759 (Korelasi Negatif Kuat)
5. `sex` -0.318896 (Korelasi Negatif Kuat)
6. `age` -0.222416 (Korelasi Negatif)
7. `trestbps` -0.115614 (Korelasi Negatif Lemah)
8. `chol` -0.0105627 (Korelasi Negatif Lemah)
9. `fbs` 0.027210 (Korelasi Positif Lemah)
10. `restecg` 0.171453 (Korelasi Positif Lemah)
11. `slope` 0.326473 (korelasi Positif Kuat)
12. `cp` 0.422559 (korelasi Positif Kuat)
13. `thalach` 0.432211 (korelasi Positif Kuat)
        ''')

        st.write('''
    Jadi, bisa disimpulkan faktor yang paling berpengaruh terhadap penyakit jantung ialah, sebagai berikut:

1. `ca` (semakin banyak major vessels ,maka akan semakin tinggi resiko terkena penyakit jantung)
2. `oldpeak` (Semakin rendah depresi ST yang disebabkan oleh latihan relatif terhadap istirahat, maka resiko terkena penyakit jantung akan semakin tinggi)
3. `exang` (Apibila exercise induced angina rendah, maka resiko terkena penyakit jantung akan semakin tinggi)
4. `thal` (semakin rendah tipe jenis defek jantung, maka resiko terkena penyakit jantung semakin tinggi)
5. `sex` (Perempuan memiliki resiko terkena penyakit jantung lebih tinggi dibandingkan laki-laki)
6. `age` (semakin muda umur, ternyata semakin tinggi terkena penyakit jantung)
7. `slope` (Semakin tinggi kemiringan segmen latihan ST maka, resiko terkena penyakit jantung semakin tinggi)
8. `cp` (Semakin tinggi tipe Jenis rasa sakit pada dada, maka resiko terkena penyakit jantung semakin tinggi)
9. `thalach` (semakin tinggi detak jantung maksimum yang dicapai pasien selama tes latihan, maka resiko terkena penyakit jantung semakin tinggi)
    ''')

        st.write('''
    ###### Kesimpulan.

1. `cp`, `thalach`, dan `slope` berkorelasi positif cukup kuat dengan `target`.
2. `oldpeak`, `exang`, `ca`, `thal`, `sex`, dan `age` berkorelasi cukup kuat dengan `target`.
3. `fbs`, `chol`, `trestbps`, dan `restecg` memiliki korelasi yang lemah dengan `target`.

Feature yang dipilih yaitu :`cp`, `thalach`, `slope`, `oldpeak`, `exang`, `ca`, `thal`, `sex`, dan `age` untuk dianalisa lebih lanjut.
    ''')

def machine_learning():
    st.header('Machine Learning Model')
    var = st.selectbox('Go To',('Before Tunning','After Tunning','ROC-AUC','Threshold','Kesimpulan'))
    if var == 'Before Tunning':
        accuracy_score = {
            'Logistic Regression': 0.80,
            'Decision Tree' : 0.70,
            'Random Forest' : 0.77,
            'MLP Classifier' : 0.78
        }
        st.write('''
        **Model Before Tunning**
        
        Berikut adalah hasil akurasi model sebelum melakukan tunning
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns = ['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi model sebelum dilakukan tunning, model dengan akurasi tertinggi adalah Logistic Regression  dengan tingkat akurasi 0.80
        ''')

    elif var == 'After Tunning':
        accuracy_score = {
            'Logistic Regression': 0.80,
            'Decision Tree' : 0.75,
            'Random Forest' : 0.84,
            'MLP Classifier' : 0.84
        }
        st.write('''
        **Model After Tunning**
        
        Berikut adalah hasil akurasi model setelah melakukan tunning
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns = ['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi model setelah dilakukan tunning, model dengan akurasi tertinggi adalah 
        Random Forest dan MLP Classifier dengan tingkat akurasi 0.84
        ''')

    elif var == 'ROC-AUC':
        ROC_AUC = {
            'Logistic Regression': 0.88,
            'Decision Tree': 0.89,
            'Random Forest': 0.80,
            'MLP Classifier': 0.88
        }
        st.write('''
        **ROC-AUC**

        Berikut adalah hasil ROC-AUC dari tiap model.
        ''')
        st.dataframe(pd.DataFrame(ROC_AUC.items(), columns=['Model', 'AUC Score']))
        st.write('''
        Jika dilihat dari nilai AUC-ROC, model Decision Tree memiliki nilai yang paling tinggi serta model Random Forest memiliki nilai yang paling rendah.

ROC adalah kurva probabilitas dan AUC mewakili tingkat atau ukuran pemisahan. Ini menunjukkan seberapa baik model mampu membedakan antara kelas. Semakin tinggi AUC, semakin baik modelnya dalam memprediksi kelas 0 sebagai 0 dan kelas 1 sebagai 1.

Kurva ROC digambarkan dengan TPR (True Positive Rate) melawan FPR (False Positive Rate) di mana TPR berada di sumbu y dan FPR berada di sumbu x.

Model yang sangat baik memiliki AUC mendekati 1, yang berarti memiliki ukuran pemisahan yang baik. Model yang buruk memiliki AUC mendekati 0, yang berarti memiliki ukuran pemisahan yang terburuk.

Ketika AUC adalah 0.7, artinya ada peluang sebesar 70% bahwa model akan mampu membedakan antara kelas positif dan kelas negatif. Ketika AUC mendekati 0.5, model tidak memiliki kemampuan diskriminasi untuk membedakan antara kelas positif dan kelas negatif. Ketika AUC mendekati 0, model memprediksi kelas negatif sebagai kelas positif dan sebaliknya.
        ''')
        st.image('ROC-AUC.png')
        st.write('''
        Berdasarkan hasil ROC-AUC dari tiap model, model dengan ROC-AUC tertinggi adalah Random Forest, yaitu 0.90
        ''')

    elif var == 'Threshold':
        Threshold = {
            'Logistic Regression': 0.373,
            'Decision Tree': 0.375,
            'Random Forest': 0.574,
            'MLP Classifier': 0.432
        }
        st.write('''
        **Threshold**

        Berikut adalah nilai threshold dari tiap model.
        ''')
        st.dataframe(pd.DataFrame(Threshold.items(), columns=['Model', 'Threshold']))
        st.write('''
        1. Jika kita menginginkan model yang memiliki sensitivitas yang tinggi, yaitu kemampuan untuk mendeteksi sebanyak mungkin kasus positif (True Positive), maka lebih baik menggunakan treshold yang lebih rendah. Namun, ini mungkin juga akan menyebabkan peningkatan False Positive Rate (kasus negatif yang salah diprediksi positif).
2. Sebaliknya, jika kita ingin mengurangi kesalahan dalam memprediksi kasus negatif sebagai positif (False Positive), maka kita akan memilih treshold yang lebih tinggi. Namun, ini dapat mengurangi sensitivitas model (menyebabkan lebih banyak True Negative yang salah diprediksi negatif).
        ''')
        st.write('''
        Berdasarkan nilai threshold tiap model, model dengan threshold tertinggi adalah Random Forest, yaitu 0.574. Sedangkan model dengan threshold terendah adalah Logistic Regression, yaitu 0.373
        ''')

    elif var == 'Kesimpulan':
        st.write('''
        Model dengan tingkat accuracy tertinggi adalah Multi-layer Perceptron dan Random Forest, yaitu sebesar 84%. Sedangkan Decision Tree memiliki tingkat accuracy sebessar 75%. 
Model yang memiliki score AUC-ROC tertinggi adalah Decision Tree, yaitu sebesar 89%. Sedangkan Random Forest sebesar 80% serta Multi-layer Perceptron dan Logistic Regression sebesar 88%. 
Model dengan tingkat threshold tertinggi adalah Random Forest, yaitu sebesar 0.54, sedangkan yang terendah adalah Logistic Regression sebesar 0.373. Sedangkan Decision Tree sebesar 0.375 dan Multi-layer Perceptron sebesar 0.43.

Berdasarkan data tersebut, dapat disimpulkan bahwa model yang performanya lebih bagus adalah Multi-layer Perceptron. Hal tersebut disebabkan oleh beberapa argumentasi sebagai berikut:
1. Multi-layer Perceptron merupakan salah satu model yang memiliki nilai akurasi yang tertinggi, yaitu sebesar 84%.
2. Multi-layer Perceptron memiliki nilai ROC-AUC yang tinggi. Walaupun ROC-AUC Decision Tree yang tertinggi, yaitu 89%, tetapi nilai tersebut tidak terpaut jauh dengan nilai ROC-AUC Multi-layer Perceptron, yaitu 88%.
3. Multi-layer Perceptron memiliki tingkat threshold yang lebih rendah, yaitu 0.43, dibandingkan dengan Random Forest, yaitu 0.57 sehingga dengan tingkat threshold yang rendah maka akan dapat mendeteksi kasus yang positif (True Positive) sebanyak mungkin. Walapun akan menyebabkan kasus negatif yang salah diprediksi positif. Namun, dalahm kasus ini, kasus True Positif lebih dipentingkan dibandingkan dengan kasus False Positif.
        ''')

def prediction():
    st.header('Prediction')
    st.write("""
    This app predicts the **Heart Disease**

    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1, 4, 2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1
            age = st.sidebar.slider("Usia", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features

    input_df = user_input_features()
    img = Image.open("Cardiovascular disease (CVDs).png")
    st.image(img, width=600)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("model_mlpt.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")


def contact():
    st.header('About Me')
    st.image('foto_profil.jpeg', width=500)
    st.write('''
    Saya adalah Mahasiswa yang sedang berkuliah di Universitas Hasanuddin Jurusan Teknik Pertambangan. Semasa kuliah, saya sangat tertarik di bidang data, lebih tepatnya Data Scientist/ AI Engineer.
    Saya mengikuti program Machine Learning dan AI Beginner di DQlab sebagai bentuk untuk memenuhi keingintauan saya terhadap Data Scientist/ AI Engineer serta saya berharap dapat berkarir di bidang data. Ini adalah Capstone Project saya sebagai tugas akhir untuk menyelesaikan program kelas Machine Learning dan AI Beginner di DQlab
    ''')
    st.write('''
    **Contact Me**
    
    - [LinkedIn]("www.linkedin.com/in/ahmadalfianfaisal")
    - [Instagram]("https://instagram.com/ahmad.alfian.faisal?igshid=NzZlODBkYWE4Ng==")
    ''')




#Home
if add_sidebar_selectbox == 'Home':
    home()

#Dataset
elif add_sidebar_selectbox == 'Dataset':
    dataset()

#Exploratory Data Analysis
elif add_sidebar_selectbox == 'Exploratory Data Analysis':
    exploratory_data_analysis()

#Machine Learning
elif add_sidebar_selectbox == 'Machine Learning':
    machine_learning()

#Prediction
elif add_sidebar_selectbox == 'Prediction':
    prediction()

#Contact
elif add_sidebar_selectbox == 'Contact':
    contact()
