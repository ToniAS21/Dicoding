# Laporan Proyek Machine Learning - Toni Andreas Susanto

## Project Overview

Mitchell Kapor, seorang entrepreneur berkewarganegaraan Amerika pernah menyampaikan pernyataan seperti berikut.

*“Getting information off the internet is like taking a drink from a fire hydrant”.*
        

Dunia telah berubah. Informasi seakan menjadi tidak terbatas dan arusnya begitu deras. Hal ini membuat banyak orang kewalahan dengan informasi yang tersedia di internet. Seperti yang digambarkan Kapor, bagaikan mengambil air minum dari hidran. Air yang kita butuhkan mungkin hanya sedikit --segelas misalnya, tapi sumber airnya begitu besar dan mengalir deras. 

Kewalahan atau overwhelmed adalah kata yang tepat untuk menggambarkan situasi tersebut. Dalam menghadapi derasnya arus data di era digital, kita perlu memiliki keahlian untuk memilah dan mengekstrak mana informasi penting dan mana yang tidak. Tanpa keahlian ini, kita bisa “tenggelam” dalam derasnya arus informasi dan menghabiskan banyak waktu di internet. 

Di sinilah peran sistem rekomendasi yaitu untuk memfilter data yang melimpah menjadi informasi penting dan bermanfaat bagi perusahaan atau organisasi. 

Bagi perusahaan atau organisasi, era digital membuat data menjadi melimpah, mudah, dan cepat diperoleh. Di satu sisi, kemudahan ini adalah hal positif. Namun, di sisi lain, terkadang data yang melimpah tidak disertai dengan kualitas yang memadai. 

Data mungkin mudah dikumpulkan, tetapi mendapatkan data berkualitas baik, bisa jadi sulit. Kita perlu menginvestasikan sebagian besar waktu untuk membersihkan dan memahami bagaimana karakteristik data sebelum menggunakannya. 


## Business Understanding

Selama beberapa dekade terakhir, dengan munculnya Youtube, Amazon, Netflix, dan banyak layanan web sejenis lainnya, sistem pemberi rekomendasi semakin banyak mengambil tempat dalam kehidupan kita. Dari e-commerce (menyarankan kepada pembeli artikel yang dapat menarik minat mereka) hingga iklan online (menyarankan kepada pengguna konten yang tepat, sesuai dengan preferensi mereka), sistem pemberi rekomendasi saat ini tidak dapat dihindari dalam perjalanan online kita sehari-hari.

Secara umum, sistem pemberi rekomendasi adalah algoritme yang ditujukan untuk menyarankan item yang relevan kepada pengguna (item seperti film untuk ditonton, teks untuk dibaca, produk untuk dibeli, atau apa pun tergantung pada industri).

Sistem rekomendasi sangat penting di beberapa industri karena dapat menghasilkan pendapatan dalam jumlah besar ketika efisien atau juga menjadi cara untuk menonjol secara signifikan dari pesaing. Sebagai bukti pentingnya sistem pemberi rekomendasi, kami dapat menyebutkan bahwa, beberapa tahun yang lalu, Netflix mengadakan tantangan ("hadiah Netflix") di mana tujuannya adalah untuk menghasilkan sistem pemberi rekomendasi yang berkinerja lebih baik daripada algoritmanya sendiri dengan hadiah dari 1 juta dolar untuk menang.

### 1. Problem Statements
Berdasarkan uraian yang telah disampaikan sebelumnya, kita menyadari masalah yang umumnya terjadi diberbagai perusahaan digital saat ini adalah bagaimana memberikan rekomendasi yang efektif dan cepat kepada pelanggan atau konsumennya?


### 2. Goals
Kita menyadari diera saat ini (*Big Data*) cenderung sulit
bahkan mustahil memberikan rekomendasi yang jumlahnya sangat banyak dengan efektif dan efisien, apabila dilakukan secara manual. Oleh sebab itu, melalui projek ini bertujuan untuk mencoba masalah tersebut yaitu menghasilkan sistem rekomendasi yang memberikan beberapa rekomendasi yang memudahkan pelanggan atau konsumen karena dapat dengan efektif dan efisien.


### 3. Solution Approach
Dalam rangka mencapai tujuan sebelumnya yaitu menghasilkan model yang efektif dan efisien. Saya mengajukan dua pendekatan solusi yaitu **Content Based Filtering** dan **Collaborative Filtering**. Kedua pendekatan ini dipilih karena relatif cukup populer dan dapat menjadi solusi yang saling melengkapi. Ide dari sistem rekomendasi berbasis konten (*content-based filtering*) adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. *Collaborative filtering* bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Collaborative filtering dibagi lagi menjadi dua kategori, yaitu: model based (metode berbasis model *machine learning*) dan memory based (metode berbasis memori). Yang pada projek kali ini saya menerapkan metode berbasis model *machine learning* lebih spesifiknya *Deep learning atau Neural Network*. *Deep Learning* bekerja dengan membangun 3 tipe yaitu *input layer*, beberapa *hidden layer* dan *output layer* kemudian melakukan pembaruan bobot dsb agar menemukan pola yang optimal dalam mengeneralisasi data.



## Data Understanding
Dataset yang kita gunakan awalnya terdiri dari 3 file yaitu Users, Books dan Ratings dengan judul [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Ratings.csv). 

### 1. Informasi dan Uraian Data

- Pengguna (*Users*)
Berisi pengguna. Perhatikan bahwa ID pengguna (User-ID) telah dianonimkan dan dipetakan ke bilangan bulat. Data demografis disediakan (Lokasi, Usia) jika tersedia. Jika tidak, bidang ini berisi nilai NULL. Pada file ini terdiri dari 278.858 baris data. 
   * `User-ID` : Nomor identitas pengguna.
   * `Location` : Lokasi pengguna
   * `Age` : Umur Pengguna

- Buku (*Books*)
Buku diidentifikasi oleh ISBN masing-masing. ISBN yang tidak valid telah dihapus dari set data. Selain itu, beberapa informasi berbasis konten diberikan (Judul Buku, Penulis Buku, Tahun Penerbitan, Penerbit), diperoleh dari Amazon Web Services. Perhatikan bahwa dalam kasus beberapa penulis, hanya yang pertama disediakan. URL yang tertaut ke gambar sampul juga diberikan, muncul dalam tiga rasa berbeda (Image-URL-S, Image-URL-M, Image-URL-L), yaitu kecil, sedang, besar. URL ini mengarah ke situs web Amazon. Pada file ini terdiri dari 271.360 baris data.

   * `ISBN` : (*International Standard Book Number*) adalah kode pengidentifikasian buku yang bersifat unik (identitas buku).
   * `Book-Title` : Judul buku 
   * `Book-Author` : Penulis buku
   * `Year-Of-Publication` : Tahun publikasi buku
   * `Image-URL-S` : Link untuk melihat gambar kecil dari buku 
   * `Image-URL-M` : Link untuk melihat gambar sedang dari buku
   * `Image-URL-L` : Link untuk melihat gambar besar dari buku

- Peringkat (*Ratings*)
Berisi informasi peringkat buku. Rating (Book-Rating) bersifat eksplisit, dinyatakan dalam skala 1-10 (nilai yang lebih tinggi menunjukkan apresiasi yang lebih tinggi), atau implisit, yang dinyatakan dengan 0. Pada file ini terdiri dari 1.149.780 baris data.
   * `User-ID` : Nomor identitas pemberi rating
   * `ISBN` : kode pengidentifikasian buku yang bersifat unik (identitas buku)
   * `Book-Rating` : Rating yang diberikan konsumen (rentang 0 - 10)


### 2. Exploratory Data Analysis (EDA)

  - Book-Title
    * Jumlah Judul Buku :  242135
    * Jumlah Nomor ISBN Buku :  271360
    > Kita menemukan terdapat perbedaan jumlah antara Judul buku dan nomor ISBN. Seperti yang kita ketahui nomor ISBN cenderung bersifat unik bagi setiap buku sehingga perbedaan jumlah ini disebabkan terdapat judul buku yang sama persis namun berbeda secara isi atau/dan penulis.
  
  - Book-Author
    * Jumlah Penulis Buku :  102024
    * Jumlah Nomor ISBN Buku :  271360
    > Ternyata terdapat 102.024 jumlah penulis dalam data books dari 271.360 jenis buku yang berbeda. Jadi terdapat beberapa buku yang ditulis oleh penulis yang sama.
  
  - Jumlah rating yang diberikan :  1149780

  - Jumlah Buku Yang Diberikan Rating : 340556

  - Jumlah pengguna yang memberikan rating :  105283

Informasi Tambahan : 

 - Jumlah Judul Buku :  242135
 - Jumlah Penulis Buku :  102024
 - Jumlah pengguna yang memberikan rating :  105283
 - Jumlah rating yang diberikan :  1149780
 - Jumlah ISBN pada data books :  271360
 - Jumlah ISBN pada data rating :  340556


## Data Preparation - Umum :

### 1. Menggabungkan Data
Kita mencoba menggabungkan data tentang rating dan books berdasarkan kolom `ISBN` dengan fungsi `merge()`. Alasan dari penggabungan ini agar pada satu data memiliki informasi lengkap berupa buku dan rating.

### 2. Mengecek dan Mengatasi Missing Value
Melalui proses penggabungan Data menimbulkan banyak baris yang menjadi missing value. Hal ini disebabkan adanya perbedaan identitas data buku sehingga menimbulkan diantara salah satu hilang (entah pada books atau rating) sehinga teridentifikasi missing value. Terdapat banyak missing value pada sebagian besar fitur. Hanya fitur `User-ID`, `ISBN`, dan `Book-Rating` atau `Book_Rating` saja yang memiliki 0 missing value. Selain itu, `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L`. Selanjutnya, kita membuang baris-baris yang meiliki missing value dengan fungsi `dropna()`. Alasan dilakukan proses ini agar ketika proses pelatihan model tidak terdapat informasi yang hilang sehingga menghasilkan model yang lebih optimal.


## Data Preparation - Content Based Filtering :

### 1.Memotong Dataset 
Kita membuang data-data yang memiliki nilai ISBN lebih dari 1 maka a
Melalui proses penggabungan data dan menghapus missing value menghasilkan 1.031.129 baris dan 11 kolom data. Data ini relatif sangat besar sehingga pada projek kali ini kita hanya menggunakan 20.000 baris saja dikarenakan keterbatasan komputasi dan hanya untuk keperluan belajar.

### 2. Konversi Data Series-List
Proses ini dilakukan agar data yang awalnya berbentuk data frame berbentuk list. Proses ini dilakukan karena persyaratan input TF-IDF Vectorizer membutuhkan list. Proses ini menggunakan fungsi `tolist()`.

### 3. Membuat Dictionary
untuk menentukan pasangan key-value pada data ISBN, Author, dan book_title yang telah kita siapkan sebelumnya. Proses ini menggunakan fungsi `DataFrame({})`.


## Model Development dengan Content Based Filtering

Model ini bekerja dengan melihat kemiripan suatu konten, dalam kasus ini konten yang dimaksud adalah judul buku. Model berusaha menghitung tingkat kemiripan antar judul buku dan memberikan kepada user tingkat kemiripan yang paling tinggi. Model ini menjadi solusi dalam menghasilkan sistem rekomendasi yang efektif dan efisien karena mempertimbangkan konten user dan hanya dalam hitungan detik dalam memberikan rekomendasi.


### 1. TF-IDF Vectorizer
Teknik penilaian ini disebut TF-IDF, kepanjangan dari *Term Frequency-Inverse Document Frequency*. Ia bertujuan untuk mengukur seberapa penting suatu kata terhadap kata-kata lain dalam dokumen. Secara matematis, TF-IDF didefinisikan dengan dua besaran, yaitu TF dan IDF. TF (*Term Frequency*) mengukur frekuensi atau seberapa sering suatu kata atau term muncul dalam teks tertentu. Teks yang berbeda dalam dokumen mungkin memiliki panjang yang berbeda, tergantung dari panjang dokumen. Oleh karena itu, kita melakukan normalisasi dengan membagi jumlah kemunculan terhadap panjang dokumen. Proses ini menggunakan fungsi ` TfidfVectorizer()`. Kita menggunakan fitur judul buku (`book_title`) karena dirasa lebih relevan dalam memberikan rekomendasi (input `.fit()`)berdasarkan kemiripan, cenderung judul yang mirip relatif memiliki isi buku yang serupa pula. Berbeda dengan yang lainnya seperti nama penulis, misal nama penulis Joni dengan Jonis tentu lebih sering berbeda dari segi konten, judul dsb. Selanjutnya dilakukan fit baru ditransformasikan ke bentuk matrix dengan fungsi `fit_transform()` sehingga menghasilkan ukuran matrix sebesar 20000 adalah ukuran data dan 15924 adalah jenis Judul Buku. Setelah itu, untuk menghasilkan vektor tf-idf dalam bentuk matriks, kita menggunakan fungsi `todense()`. Proses ini diperlukan untuk dapat diproses ketahapan selanjutnya yaitu menghitung tingkat kemiripan.

### 2. Cosine Similarity
Cosine similarity mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks. Sebagai contoh, dalam studi kasus ini, cosine similarity digunakan untuk mengukur kesamaan nama judul buku.
Proses ini menggunakan fungsi `cosine_similarity()` dari pustaka *sklearn*. Pada tahapan ini, kita menghitung *cosine similarity* dataframe `tfidf_matrix` yang kita peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi cosine similarity dari library sklearn, kita telah berhasil menghitung kesamaan (*similarity*) antar restoran. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array. 

### 3. Menyajikan Top-N Recommendation
Kita membuat fungsi yang dapat memberikan hasil rekomendasi. Fungai ini memiliki parameter yang terdiri dari wajib di isi dan tidak. Parameter yang wajib di isi adalah ISBN dan selain itu sudah terdapat nilai *default*. Parameter yang memiliki nilai *default* seperti similiraity_data yang menggunakan variabel yang telah dihitung pada tahap sebelumnya dengan nama `cosine_sim_df`, lalu parameter items yang akan memberikan hasil berupa informasi `ISBN`, `book_author`, dan `book_title`, kemudian parameter k yang merujuk pada berapa banyak rekomendasi yang ingin diberikan, secara *default* adalah 5. Cara kerja fungsi ini adalah Mengambil data dengan menggunakan `argpartition()` untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan, kemudian mengubah dataframe menjadi numpy dan membuat range yang terdiri dari `range(start, stop, step)`. Setelah itu, mengambil data dengan similarity terbesar dari index yang ada dan menghapus nama buku yang dimasukan oleh pengguna sehingga tidak merekomendasikan nama buku yang sama dengan yang dimasukan pengguna melalui fungsi `drop()`. Setelah itu akan mengembalikan dalam bentuk dataframe.

Berikut hasil rekomendasi dari buku dengan ISBN `0155061224` :

- **Data Buku `0155061224`** :

| No | ISBN | Book Author | Book Title |
|--|---|:---:|---|
| 1. | 0155061224 | Judith Rae	| Rites of Passage |


- **5 Rekomendasi Buku, Berdasarkan Konten Buku `0155061224`**

| No | ISBN | Book Author | Book Title |
|--|---|:---:|---|
| 1. | 0553580515 | Connie Willis | Passage | 
| 2. | 0679435506 | Marianne Williamson | Illuminata: Thoughts, Prayers, Rites of Passage | 
| 3. | 0380715325 | Alison McLeay | Passage Home |
| 4. | 0812510488 | Christopher Pike	 | The Season of Passage |
| 5. | 0373031203 | Rebecca Winters | Rites Of Love (Harlequin Romance, No 3120) |



Dari gambar di atas terlihat kita ingin mencari rekomendasi dari buku yang berjudul **Rites of Passage** dan sistem kita sudah dapat merekomendasikan judul buku yang serupa, memberikan judul buku yang memiliki keyword *Rites* atau/dan *of Passage*.



## Data Preparation - Collaborative Filtering :

### 1. Menyandikan (Encode)
Mengubah `userID` menjadi list tanpa nilai yang sama dengan fungsi `unique()` dan `tolist()` serta menyimpannya dalam variabel `user_ids`. Setelah itu,  Melakukan encoding `user_ids` ke dalam indeks integer dan Melakukan proses encoding angka ke ke `user_ids`. Begitu juga dengan `ISBN`, Mengubah `ISBN` menjadi list tanpa nilai yang sama dengan fungsi `unique()` dan `tolist()` serta menyimpannya dalam variabel `books_ids`. Setelah itu,  Melakukan encoding `books_ids` ke dalam indeks integer dan Melakukan proses encoding angka ke ke `book_ids`. 

### 2. Memetakan Fitur
Selanjutnya, memetakan `userID` dan `ISBN` ke dataframe yang berkaitan.
Mengecek beberapa hal dalam data seperti jumlah user sebesar 92106, jumlah buku sebesar 2701145, kemudian mengubah nilai rating menjadi float.

Tahap persiapan ini penting dilakukan agar data siap digunakan untuk pemodelan. Namun sebelumnya, kita perlu membagi data untuk training dan validasi terlebih dahulu yang akan kita pelajari di materi berikutnya. 

### 3. Membagi Data untuk Training dan Validasi
Pertama kita mengacak dataset ini sebelum membagi data tersebut `sample(frac=1, random_state=42)`. Kemudian Membuat variabel x untuk mencocokkan data user dan books menjadi satu value. Lalu Membuat variabel y untuk membuat rating dari hasil Membagi menjadi 80% data train dan 20% data validasi.



## Model Development dengan Collaborative Filtering

Model ini bekerja dengan mengidentifikasi buku-buku yang mirip dan tidak pernah dibeli konsumen dengan mempertimbangkan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Model ini menjadi solusi dalam menghasilkan sistem rekomendasi yang efektif dan efisien karena mempertimbangkan preferensi pengguna dan hanya hitungan detik dalam memberikan rekomendasi.


### 1. Membuat class RecommenderNet
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan resto dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan resto. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan resto. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan resto. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Kita membuat class RecommenderNet dengan [keras Model class](https://keras.io/api/models/model/). Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs [Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan.


### 2. Compile Model 
Kita melakukan insiasai model menggunakan class RecommenderNet yang telah dibuat sebelumnya dengan variabel yang berisi jumlah user dan buku serta ukuran *embedding* yaitu 50. Setelah itu, melakukan *compile* model. Model ini menggunakan *Binary Crossentropy* untuk menghitung *loss function*, Adam (*Adaptive Moment Estimation*) sebagai *optimizer*, dan *root mean squared error* (RMSE) sebagai *metrics evaluation*. 


### 3. Menerapkan Callbacks Dinamis
Kita menggunakan fungsi `ReduceLROnPlateau()` dan `EarlyStopping()` untuk meningkatkan efektivitas dan efisiensi proses pelatihan model. Melalui *callbacks* dinamis ini relatif model dapat mengecilkan nilai *learning rate* (kecepatan belajar) seiring pertambahan *epochs* sehingga memudahkan model menemukan titik optimal (konvergen) dalam mengeneralisasi data. 


### 4. Pelatihan Model
Setelah model telah dibuat, compile dan inisiasi callbacks akan dilakukan pelatihan model dengan fungsi `fit()`. Yang mana mengisi parameter `x`, `y`, `batch_size = 64`, `epochs = 100`, `validation_data = (x_val, y_val)` dan `callbacks = [reduce_lr, early_stop]`.
 

### 5. Menyajikan Top-N Recommendation
Kita mulai dengan megambil sampel data kita untuk sebagai contoh. Setelah mendapatkan sampelnya kita membuat variabel terkait buku yang sudah pernah dibeli user. Kemudian kita juga membuat variabel terkait buku yang belum pernah dibeli user. Kemudian mempersiapkan datanya menjadi bentuk array sehingga dapat diprediksi model. Kemudian kita menggunakan model yang telah dilatih untuk memberikan prediksi lalu mengumpulkan hasil prediksi model. Kemudian kita menampilkan hasil kepada user dalam bentuk 2 hal. Pertama terkait 5 buku yang memiliki rating tertinggi yang diberikan user. Kedua terkait 10 buku yang belum pernah dibeli user dan diperkirakan akan digemari user berdasarkan pertimbangan preferensi user (rating), preferensi ini menggunakan nilai median agar dapat melihat nilai rating tengah atau nilai rating yang dapat mewakilkan buku tersebut.

Berikut hasil rekomendasi untuk user id `185233` :

- **Books with High Ratings from User**

| No | ISBN | Book Author | Book Title | Rating |
|--|---|:---:|---|---|
| 1. | 043935806X | J. K. Rowling | Harry Potter and the Order of the Phoenix (Book 5) | 10.0 | 
| 2. | 0307010368 | Little Golden Staff | Snow White and the Seven Dwarfs | 9.0 |
| 3. | 0812550544 | Michael Norman | Haunted America (Haunted America) | 10.0 |
| 4. | 0451169530 | Michael Norman | The Stand: Complete and Uncut | 10.0 |
| 5. | 078686382X | Deanna F. Cook | Disneys Family Cookbk-OS | 10.0 |


- **Top 10 Books Recommendation**

| No | ISBN | Book Author | Book Title | Rating |
|--|---|:---:|---|---|
| 1. | 0091842050 | Bradley Trevor Greive | The Blue Day Book: A Lesson in Cheering Yourself Up | 10.0 |
| 2. | 0394800389 | Dr. Seuss | Fox in Socks (I Can Read It All by Myself Beginner Books) | 10.0 |
| 3. | 0823401898 | Florence Parry Heide | The Shrinking of Treehorn | 10.0 |
| 4. | 1563891336 | Neil Gaiman | Death: The High Cost of Living | 10.0 |
| 5. | 3522128001 | Michael Ende | Die unendliche Geschichte: Von A bis Z | 10.0 |
| 6. | 0920668364 | Robert Munsch | Love You Forever | 9.5 |
| 7. | 0316779059 | Martha Sears | The Baby Book: Everything You Need to Know About Your Baby from Birth to Age Two | 9.0 |
| 8. | 1844262553 | Paul Vincent | Free | 9.0 |
| 9. | 2205054252 | Larcenet | Le Combat ordinaire, tome 1 | 9.0 |
| 10. | 0064440508 | Else Holmelund Minarik | A Kiss for Little Bear | 8.5 |


Dari gambar di atas terlihat sistem menampilkan 5 buku dengan rating tertinggi dari pemberian user. Kemudian menampilkan 10 rekomendasi buku yan belum pernah dibeli user dan cenderung memiliki rating nilai sangat tinggi yaitu 8,5-10 dari skala 0-10.


## Kelebihan dan Kekurangan Setiap Pendekatan


### Content Based Filtering

- **Kelebihan** : Model dapat memberikan rekomendasi yang serupa dengan buku yang telah kita beli sehingga relatif dapat membeli buku yang tepat dan telah terbukti diminati kita karena berdasarkan kemiripan judul buku di masa lalu.

- **Kekurangan** : Cenderung model hanya memberikan rekomendasi buku yang mirip atau relatif bukan buku yang unik.


### Collaborative Filtering

- **Kelebihan** : Model dapat memberikan rekomendasi yang lebih unik karena mempertimbangkan segi preferensi (rating) bukan konten yang pernah dibeli pengguna dan relatif masih disukai oleh konsumen karena memiilki rating serupa dengan buku yang pernah dibeli. 

- **Kekurangan** : Cenderung model hanya memberikan rekomendasi buku unik dan kemungkinan tidak disukai konsumen karena belum terbukti diminati sebab tidak berdasarkan kemiripan judul buku di masa lalu.


## Evaluation

### 1. Model Content Based Filtering

Pertama pada model *Content Based Filtering* dapat dikatakan sudah sangat baik karena relatif 5 dari 5 rekomendasi yang diberikan sistem pada data sampel cenderung mirip karena memberikan rekomenedasi dengan keyword *Rites* dan *of Passage*. Kita menggunakan metrik *recomender system precision*. Metrik ini sesuai dengan konteks data kita yaitu teks karena berdasarkan judul buku sehingga perlu melihat apakah kata-kata antar judul mirip. Kemudian terkait konteks masalah dan solusi yaitu ingin menghasilkan sistem rekomendasi secara efektif karena judul yang diberikan relatif relevan dan efisien karena berbasis sistem sehingga cepat, jadi metrik ini sesuai dengan masalah dan solusi karena ingin meningkatkan ketepatan memberikan rekomendasi yang sesuai/relevan. 

Formula metrik Recomender System Precision (RSP) ini adalah sebagai berikut :

$$RSP = R_R/R_A$$

Ket : 

$R_R$ = Jumlah rekomendasi yang relevan

$R_A$ = Jumlah keseluruhan rekomendasi yang prediksi model


Cara kerja metrik ini adalah dengan membandingkan seberapa banyak prediksi model yang relevan atau sesuai dengan keseluruhan rekomendasi yang telah diberikan. 

Berikut hasil rekomendasi dari buku dengan ISBN `0155061224` :

- **Data Buku `0155061224`** :

| No | ISBN | Book Author | Book Title |
|--|---|:---:|---|
| 1. | 0155061224 | Judith Rae	| Rites of Passage |


- **5 Rekomendasi Buku, Berdasarkan Konten Buku `0155061224`**

| No | ISBN | Book Author | Book Title |
|--|---|:---:|---|
| 1. | 0553580515 | Connie Willis | Passage | 
| 2. | 0679435506 | Marianne Williamson | Illuminata: Thoughts, Prayers, Rites of Passage | 
| 3. | 0380715325 | Alison McLeay | Passage Home |
| 4. | 0812510488 | Christopher Pike	 | The Season of Passage |
| 5. | 0373031203 | Rebecca Winters | Rites Of Love (Harlequin Romance, No 3120) |


Sehingga presisi sistem rekomendasi *Content Based Filtering* pada sampel ini adalah 5/5 = 100%.

### 2. Model Collaborative Filtering

Kedua pada model *Collaborative Filtering*, menggunakan metrik *Root Mean Squared Error (RMSE)* untuk mengevaluasi seberapa baik model dalam memberikan rekomendasi. Kita memilih metrik RMSE karena sesuai dengan konteks data kita yaitu angka karena berdasarkan ratings sehingga perlu melihat apakah model dapat memprediksi nilai rating dengan selisih kesalahan terkecil. Kemudian terkait konteks masalah dan solusi yaitu ingin menghasilkan sistem rekomendasi secara efektif karena berbasis rating pengguna dan efisien karena berbasis sistem sehingga cepat, jadi metrik ini sesuai dengan masalah dan solusi karena ingin mengecilkan tingkat error sehingga sistem lebih efektif. Selain itu, dengan metrik RMSE relatif dapat diinterpretasikan langsung karena merupakan nilai rata-rata tingkat kesalahan dan sudah diakarkan.

Formula metrik Root Mean Squared Error (RMSE) adalah sebagai berikut :

$$RMSE = \sqrt{\sum{(Y_t - Y_p)^2} \over n}$$

Ket :

$Y_t$ = Y true (Aktual)

$Y_p$ = Y predict (Prediksi)

n = jumlah data

Cara kerja metrik ini adalah dengan menyelisihkan nilai aktual dengan nilai prediksi lalu dikuadratkan kemudian ditotalkan dengan seluruh data dan selanjutnya dibagi dengan jumlah data, terakhir diakarkan. 

![Hasil RMSE](https://user-images.githubusercontent.com/83503249/200128246-1ee97eb6-3d89-49e6-a551-9febbc4375d8.png)

Kalau kita lihat performa model *Collaborative Filtering* sudah bagus karena cenderung memiliki error yang menurun dan relatif sudah stabil seiring pertambahan epochs. Cenderung tidak *overfitting* karena selisih error antara training dan validasi masih wajar sekitar 0.06 atau 6% an. Selain itu, model dapat dikatakan lumayan *good fit* karena relatif sudah mencapai titik error optimal di angka 0.3-an, kemudian relatif hasil RMSE *training* dan validasi sudah baik untuk kasus sistem rekomendasi, merujuk pada modul "Machine Learning Terapan".


## Conclusion
Pada projek ini kita telah melalui berbagai tahapan mulai dari memahami ulasan proyek, memahami kasus bisnis yang ingin diselesaikan, lalu mencoba mengerti data yang ada, menyiapkan dataset, modeling hingga evaluasi. Kita juga telah berhasil menyelesaikan masalah bisnis dan mencapai tujuan yang ada melalui pembangunan model dengan 2 pendekatan yaitu **Content Based Filtering** dan **Collaborative Filtering**. Yang mana kedua pendekatan ini relatif dapat saling melengkapi dalam rangka mengoptimalkan performa sistem rekomendasi dalam memberikan rekomendasi yang efektif dan efisien. Harapannya melalui projek ini dapat memberikan manfaat bagi penulis dan pembaca secara luas. Sekian terima kasih telah membaca projek ini.


## Reference

[1] Setiani, Tia Dwi. "Machine Learning Terapan". Dicoding. 2021. Tersedia: [tautan](https://www.dicoding.com/academies/319/corridor). Diakses pada 05 November 2022.

[2] Ricci, Francesco, et al. "Recommender Systems Handbook". Springer Media. 2011. Tersedia: [tautan](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf). Diakses pada 05 November 2022
