# Laporan Proyek Machine Learning - Toni Andreas Susanto

## Domain Proyek

Peningkatan penggunaan internet di seluruh dunia menghasilkan data dalam jumlah yang sangat besar. Berbagai perangkat cerdas dan internet membuat data menjadi berlimpah. Perusahaan dan organisasi mengumpulkan data berjumlah besar untuk berbagai kepentingan, salah satunya proses pengambilan keputusan.

Supaya memberikan manfaat dan dapat digunakan untuk membuat keputusan, data harus melalui proses analisis dan ekstraksi informasi (*insight*). Mengekstrak informasi dari data adalah inti dari pekerjaan *data analytics*. *Predictive analytics*, pokok bahasan dalam modul ini merupakan sub-bidang *data analytics*.

Dalam artikel berjudul “Competing on Analytics” yang diterbitkan oleh [Harvard Business Review](https://hbr.org/2006/01/competing-on-analytics), Davenport, seorang ahli *business analytics* berpendapat bahwa senjata strategis di bidang bisnis saat ini adalah pengambilan keputusan analitik. Ia merupakan teknik pengambilan keputusan berdasarkan berbagai informasi yang diekstrak dari data.

> Masalah ini harus diselasaikan agar dapat meningkatkan efisiensi dan efektivitas dalam rangka mengantisipasi peningkatan permintaan jumlah sepeda yang ingin disewa. Masalah ini dapat diselesaikan dengan cara membangun model *Machine Learning* berkaitan kasus regresi, dengan target nya adalah jumlah sepeda yang disewa dan fitur/prediktornya seperti cuaca, musim, kelembaban dsb.

Referensi yang di gunakan :
- [Dicoding, Machine Learning Terapan ](https://www.dicoding.com/academies/319/corridor)
- [Predictive analytics](https://www.ibm.com/analytics/predictive-analytics)

## Business Understanding
Sistem berbagi sepeda adalah generasi baru persewaan sepeda tradisional di mana seluruh proses mulai dari keanggotaan, persewaan, dan pengembalian menjadi otomatis. Melalui sistem ini, pengguna dapat dengan mudah menyewa sepeda dari posisi tertentu dan kembali lagi di posisi lain. Saat ini, ada lebih dari 500 program berbagi sepeda di seluruh dunia yang terdiri dari lebih dari 500 ribu sepeda. Saat ini, ada minat besar dalam sistem ini karena peran penting mereka dalam masalah lalu lintas, lingkungan dan kesehatan. Oleh karena itu, kita sebagai data scientist diminta membangun model yang dapat memprediksi jumlah sepeda yang akan dipinjam dalam suatu hari berdasarkan berbagai informasi yang tersedia. Harapannya dengan mengetahui berapa banyak permintaan sewa sepeda dalam suatu hari, para pemangku kepentingan dapat mengatur strategi untuk menambah atau mengantisipasi peningkatan permintaan sehingga dapat memaksimalkan keuntungan.

### 1. Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi jumlah sepeda yang disewa untuk menjawab permasalahan berikut.
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap jumlah total sewa sepeda ?
- Berapa jumlah total sewa sepeda dengan karakteristik atau fitur tertentu?  

### 2. Goals
Untuk  menjawab pertanyaan tersebut, Anda akan membuat *predictive modelling* dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang paling berkorelasi dengan jumlah total sewa sepeda.
- Membuat model *Machine Learning* yang dapat memprediksi jumlah total sewa sepeda seakurat mungkin berdasarkan fitur-fitur (prediktor) yang ada.

    ### Solution statements
    Kita akan mengajukan 3 solution statement. Pertama, kita akan mencoba membangun model **K-Nearest Neighbor (KNN)** dengan mengatur parameter yang ada. Kedua, kita akan mencoba membangun model **Random Forest (Bagging Algorithm)** dengan mengatur berbagai parameter yang ada. Ketiga, kita akan membangun model **Boosting** dengan mengatur berbagai parameter yang ada. Ketiga solusi ini akan diukur deng metrik yang sama yaitu *Mean Squared Error*.

### 3. Metodologi
Prediksi jumlah sewa sepeda adalah tujuan yang ingin dicapai. Seperti yang kita tahu, jumlah sewa sepeda merupakan variabel kontinu. Dalam predictive analytics, saat membuat prediksi variabel kontinu artinya Anda sedang menyelesaikan permasalahan regresi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model regresi dengan jumlah sewa sepeda  sebagai target.

### 4. Metrik
Metrik digunakan untuk mengevaluasi seberapa baik model Anda dalam memprediksi harga. Untuk kasus regresi, beberapa metrik yang biasanya digunakan adalah Mean Squared Error (MSE). Secara umum, metrik ini mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya. Kita akan bahas lebih detail mengenai metrik ini di modul Evaluasi. Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm. Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terkecil. Dengan kata lain, kita akan membuat model seakurat mungkin, yaitu model dengan nilai kesalahan sekecil mungkin.

Membuat model prediktif dengan machine learning tentu memerlukan data. Berita baiknya adalah, perusahaan memiliki data yang dibutuhkan untuk membuat model prediksi. Dataset yang akan kita gunakan pada praktik kali ini adalah Bike Rents for the Day (Regression)amond dataset.


## Data Understanding
Dataset yang kita gunakan awalnya terdiri dari 731 baris dan 16 kolom dengan judul [Bike Rents for the Day (Regression)](https://www.kaggle.com/datasets/ayessa/bike-sharing-dataset-regression).

Pada projek kita terdapat beberapa tahapan untuk memahami data : 

####  1. Deskripsi Variabel

- `instant`: indeks record
- `dteday` : tanggal
- `season`: musim (1:musim dingin, 2:musim semi, 3:musim panas, 4:musim gugur)
- `yr`: tahun (0: 2011, 1:2012)
- `mnth`: bulan (1 sampai 12)
- `holiday`: hari cuaca adalah hari libur atau tidak (disarikan dari [Web Link])
- `weekday`: hari dalam seminggu
- `workingday`: jika hari bukan akhir pekan atau hari libur adalah 1, jika tidak adalah 0.
- `weathersit`:
  * 1: Cerah, Sedikit awan, Sebagian berawan, Sebagian berawan
  * 2: Kabut + Mendung, Kabut + Awan Rusak, Kabut + Sedikit Awan, Kabut
  * 3: Salju Ringan, Hujan Ringan + Badai Petir + Awan Tersebar, Hujan Ringan + Awan Tersebar
- `temp`: Suhu normal dalam Celcius. Nilai diturunkan melalui (t-tmin)/(tmax-tmin), tmin=-8, t_max=+39 (hanya dalam skala per jam)
- `atemp`: Suhu perasaan yang dinormalisasi dalam Celcius. Nilai diturunkan melalui (t-tmin)/(tmax-tmin), tmin=-16, t_max=+50 (hanya dalam skala per jam)
- `hum`: Kelembaban normal. Nilai dibagi menjadi 100 (maks)
kecepatan angin: Kecepatan angin yang dinormalisasi. Nilai dibagi menjadi 67 (maks)
- `casual`: jumlah pengguna biasa
- `registered`: jumlah pengguna terdaftar
- `cnt`: jumlah total sewa sepeda termasuk sepeda santai dan terdaftar

#### 2. Menyesuaikan Tipe Data

Pada tahapan ini kita mengubah tipe data yang lebih sesuai dan untuk kolom yang memiliki nilai berulang diubah menjadi tipe `object` untuk mempermudah pada tahap *one-hot-encoding*. Kemudian kita menghapus kolom `dteday` karena sudah terdapat informasi waktu dikolom lainnya. Kolom `casual` dan `registered` dihapus juga karena merupakan bagian yang menjadi target kita, yang mana nilai pada kolom `casual` dan `registered` ketika dijumlahkan adalah nilai `cnt`. Terlihat kita memiliki 731 baris dan 14 kolom. Yang mana terdiri 4 bertipe `float64(4)`, 3 `int64` dan 7 `object`.

#### 3. Mengecek *Missing Value*
Tahapan ini dilakukan agar setiap data yang akan diolah sudah dalam kondisi lengkap. Tidak terdapat nilai yang kosong (*missing value*) sehingga kita dapat ke tahap selanjutnya.

#### 4. Mengecek Ringkasan Data
Tahapan ini dilakukan agar kita dapat mendapatkan gambaran data secara umum.
Terlihat data kita memiliki rentang nilai yang beragam dan sepertinya terdapat *outlier* yang akan kita periksa pada tahap selanjutnya.

#### 5. Mengecek dan Menangani *Outliers*
Tahapan ini dilakukan agar meningkatkan performa model karena nilai yang sangat berbeda dari kumpulan data (*Outliers*) cenderung berpengaruh dalam pelatihan model. Terlihat pada beberapa kolom yaitu `hum` dan `windspeed` terdapat nilai outlier sehingga kita perlu membuang nilai tersebut. Kita menggunakan metode **IQR** sehingga nilai yang lebih kecil dari `Q1 - 1.5*IQR` dan lebih besar dari `Q3+1.5*IQR` merupakan nilai *Outlier*. Setelah membuang nilai *outlier* data kita berubah dari 731 baris dan 14 kolom menjadi 717 baris dan 14 kolom.

#### 6. Exploratory Data Analysis - Univariate Analysis
Pada proses ini kita bagi menjadi *Categorical Features* dan *Numerical Features*. Cenderung pada *Categorical Features* terdapat nilai fitur yang seimbangan antar kategori dan juga ada nilai fitur yang tidak seimbang antar kategori. Pada *Numerical Features*, khususnya histogram untuk variabel "cnt" yang merupakan fitur target (label) pada data kita. Dari histogram "cnt", kita bisa memperoleh beberapa informasi, antara lain:
- Peningkatan jumlah sewa sepeda cenderung menyebar cukup merata tinggi di tengah (mean) atau mediannya. Awalnya sedikit kemudian naik hingga titik tertentu dan menurun.
- Rentang jumlah sepeda yang disewa cenderung cukup tinggi yaitu dari skala 22 hingga 8714 unit.

#### 7. Exploratory Data Analysis - Multivariate Analysis
Dengan mengamati rata-rata harga relatif terhadap fitur kategori pada plot di `.ipynb`, kita memperoleh insight sebagai berikut:

- Pada fitur `season`, rata-rata jumlah bike yang disewa cenderung berbeda alias setiap kategori pada `season` cukup mempengaruhi `cnt`.
- Pada fitur `yr`, rata-rata jumlah bike yang disewa cenderung berbeda alias setiap kategori pada `yr` cukup mempengaruhi `cnt`.
- Pada fitur `month`, rata-rata jumlah bike yang disewa cenderung terdapat berbeda dan mirip alias ada beberapa bulan yang lebih berpengaruh dibandingkan bulan lainnya dan ada juga beberapa bulan yang tidak berpengaruh dibandingkan bulan lainnya.
- Pada fitur `holiday`, rata-rata jumlah bike yang disewa berbeda alias setiap kategori pada `holiday` cukup mempengaruhi `cnt`.
- Pada fitur `weekday`, rata-rata jumlah bike yang disewa cenderung mirip alias setiap kategori pada `weekday` kurang mempengaruhi `cnt`.
- Pada fitur `workingday`, rata-rata jumlah bike yang disewa cenderung mirip alias setiap kategori pada `workingday` kurang mempengaruhi `cnt`.
- Pada fitur `weathersit`, rata-rata jumlah bike yang disewa berbeda alias setiap kategori pada `weathersit` cukup mempengaruhi `cnt`.

Melalui Pairplot dan Correlation Matrix di bawah dapat disimpulkan :

**Pairplot**
![Screenshot (503)](https://user-images.githubusercontent.com/83503249/195485908-66f7de2e-8a71-4561-95d2-6df1c824c25e.png)


**Correlation Matrix**

![Screenshot (500)](https://user-images.githubusercontent.com/83503249/195484135-279ef857-0090-42ba-9f3a-6a99d7c793bc.png)

- Cenderung kolom `instant` berkorelasi kuat dengan kelas target (`cnt`) secara positif.
- Cenderung kolom `temp` berkorelasi kuat dengan kelas target (`cnt`) secara positif.
- Cenderung kolom `atemp` berkorelasi kuat dengan kelas target (`cnt`) secara positif.
- Cenderung kolom `hum` berkorelasi lemah dengan kelas target (`cnt`) secara negatif.
- Cenderung kolom `wind` berkorelasi sedang dengan kelas target (`cnt`) secara negatif.
- Cenderung kolom `day` berkorelasi lemah dengan kelas target (`cnt`) secara negatif.

## Data Preparation

### 1. Encoding Fitur Kategori
*One Hot Encoding* dilakukan dengan alasan cenderung model machine learning dapat lebih banyak untuk data bertipe numerik. Melalui tahap ini berguna untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Kita melakukan proses ini dengan menggunakan fungsi `pd.get_dummies()` dari library [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

### 2. Reduksi Dimensi dengan PCA
Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang paling populer adalah Principal Component Analysis atau disingkat menjadi PCA. Alasan dilakukan tahap ini untuk meringkan beban komputasi dan menyederhanakan informasi yang dipelajari model. Kita akan menerapkan reduksi dimensi pada kolom `temp` dan `atemp` karena pada proses EDA terlihat kedua kolom ini memiliki pola linear yang mirip,
dengan menggunakan fungsi `PCA()` dari library scikit-learn. Parameter yang digunakan adalah `n_components=1` untuk mengambil 1 dimensi saja karena sudah mewakili 100% informasi data. Selain itu, terdapat parameter `random_state=123` untuk mengunci kerandoman komputasi. Hasilnya kita memperoleh kolom baru `temperature` yang merangkum informasi dari kolom `temp` dan `atemp` sehingga kedua kolom ini dibuang.

### 3. Train-Test-Split
Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Proses ini menggunakan fungsi `train_test_split()` dari sklearn, dengan rasio pembagian dataset adalah 80 per 20, 80% (573 baris data) untuk proses pelatihan model dan 20% (144 baris data) untuk proses testing model. 

### 4. Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. 

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

> Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan fitur standarisasi pada data latih. 


## Modeling

### 1. Model Development dengan K-Nearest Neighbor (KNN)
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. Kelebihan model ini yaitu relatif mudah dinterpretasikanm dapat digunakan untuk kasus klasifikasi dan regresi serta relatif ringan komputas. Sedangkan kekurangannya adalah pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai *curse of dimensionality* (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data. Jadi, jika Anda menggunakan model KNN, pastikan data yang digunakan memiliki fitur yang relatif sedikit.
Parameter yang digunakan pada proses pemodelan adalah menentukan parameter k (jumlah data terdekat) `n_neighbors=20`. 


### 2. Model Development dengan Random Forest (Bagging Algorithm)
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Kelebihan model ini adalah cukup sederhana penggunaannya, memiliki stabilitas yang mumpuni dan performa relatif bagus. Sedangkan kekurangannya adalah sulit diinterpretasikan karena berbagai model yang digunakan bersifat acak, relatif berat secara komputasi karena gabungan beberapa model. 

Parameter yang digunakan pada proses pemodelan :
- `n_estimator`: jumlah trees (pohon) di forest. Di sini kita set `n_estimator=40`.
- `max_depth`: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Di sini kita set `max_depth=8`.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan. Di sini kita set `random_state=55`.
- `n_jobs`: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. `n_jobs=-1` artinya semua proses berjalan secara paralel.

## 3. Model Development dengan Boosting Algorithm
Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Kelebihan model ini adalah sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest. Sedangkan kekurangannya relatif sulit diinterpretasikan dan relatif berat secara komputasi karena menggabungkan beberapa model sederhana dan dianggap lemah (*weak learners*) sehingga membentuk suatu model yang kuat (*strong ensemble learner*). 

Parameter yang digunakan pada proses pemodelan :
- `learning_rate`: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting. Diisi dengan `learning_rate = 0.05`.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan. Diisi dengan `random_state = 55`.

### Model Yang Dipilih
Kita memilih model **Random Forest** karena secara performa lebih baik dibandingkan kedua model lainnya pada kasus kita ini. Performa yang digunakan adalah *Mean Squared Error*. Cenderung performa model **Random Forest** memberikan solusi terbaik bagi kasus kita karena baik train dan testing memiliki performa lebih baik.

## Evaluation
Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau *Mean Squared Error* yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. 

Formula MSE : 
$$MSE = {\sum{(Y_t - Y_p)^2} \over n}$$

Ket
$$Y_t = Y true (Sebenarnya)$$
$$Y_p = Y predict (Prediksi)$$
$$n = jumlah data$$

Metrics ini dipilih karena relevan dengan konteks data kita yang ingin memprediksi nilai numerical bukan categorical. Sesuai juga dengan *problem statment* kita yaitu menenentukan nilai jumlah total sewa sepeda (nilai kontinue) berdasarkan prediktor yang ada. Selaras juga dengan solusi yang diinginkan yaitu memprediksi jumlah total sewa sepeda seakurat mungkin berdasarkan fitur-fitur (prediktor) yang ada sehingga harus mencari model yang memiliki MSE kecil alias lebih akurat. Formula MSE secara konsep adalah menghitung selisih nilai sebenarnya dengan nilai prediksi lalu dikuadratkan agar positif dan dibagi jumlah data. Melalui formula ini meghasilkan mekanisme yang memberikan hukuman lebih besar terhadap setiap error yang terjadi karena dikuadratkan sehingga kita dapat dengan lebih mudah memilih model yang lebih baik. 

Berikut hasil peroleh MSE setiap model kita (dalam ribu) :
**Model**|**Train**|**Testing**
:-----:|:-----:|:-----:
**KNN**|585.704388 | 645.181687 |
**Random Forest**| 116.289618 | 441.565122 |
**Boosting**| 705.481594 | 763.99904 |

Berikut gambar bar chart perbandingan nilai MSE :

![Screenshot (498)](https://user-images.githubusercontent.com/83503249/195475730-8bb771e5-048e-4203-8ca9-efe87dc9a3bd.png)


Berikut lampiran hasil prediksi model kita dalam bentuk tabel :
**Index** | Y_True | Prediksi_KNN | Prediksi RF | Prediksi Boosting
|:------:|:----:|:------:|:------:|:------:|
|**226** | 4338 | 4456.8 | 4491.9 | 4390.1 |
|**430** | 3956 | 4388.6 | 3387.0 | 3252.2 |
|**649** | 7570 | 6532.6 | 7019.5 | 6813.6 |
|**653** | 5875 | 5570.9 | 7074.7 | 6942.2 |
|**342** | 3620 | 3811.6 |  3418.2 | 3213.1 |

Berdasarkan tabel dan chart nilai MSE cenderung model **Random Forest** jauh lebih baik dari kedua algoritma lainnya. Meskipun pada lampiran hasil prediksi **Random Forest** di tabel atas masih terdapat error yang cukup signifikan dengan nilai sebenarnya tetapi kalau dihitung atau dipertimbangkan secara keseluruhan relatif lebih kecil dibandingkan kedua model lainnya. 


## Conclusion
Pada projek ini kita telah melalui berbagai tahapan mulai dari memahami Domain proyek, kemudian memahami kasus bisnis yang ingin diselesaikan, lalu mencoba mengerti data yang ada, menyiapkan dataset, modeling hingga evaluasi. Kita juga telah berhasil menyelesaikan masalah bisnis dan mencapai tujuan yang ada melalui pembangunan 3 model kemudian memilih satu model terbaik sebagai solusi masalah kita yaitu model **Random Forest**. Harapannya melalui projek ini dapat memberikan manfaat bagi penulis dan pembaca secara luas. Sekian terima kasih telah membaca projek ini.
