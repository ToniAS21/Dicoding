# Submission 1: Prediction Potensial Customer
Nama: Toni Andreas Susanto

Username dicoding: toni_andreas_s

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Bank Marketing](https://www.kaggle.com/datasets/dhirajnirne/bank-marketing) |
| Masalah | Dalam menjalankan sebuah bisnis, Marketing merupakan salah satu faktor kunci untuk mendorong kemajuan sebuah perusahaan. Hal ini disebabkan Marketinglah yang menjadi faktor langsung dalam rangka menjual sebuah produk. Pada kasus di industri perbankan memiliki tantangan  Marketing serupa pula yaitu bagaimana mengoptimalkan penjualan produk perbankan. Dalam industri perbankan memiliki data konsumen maupun calon konsumen yang melimpah tetapi mereka memiliki sumber daya (biaya marketing dan waktu) yang terbatas sehingga mesti menentukan konsumen mana yang mesti dihubungi terlebih dahulu alias menentukan manakah konsumen yang memiliki probabilitas lebih besar dalam membeli sebuah produk tersebut. Dengan demikian, masalah pada kasus kali ini adalah **Bagaimana Memprediksi Seseorang akan Membeli atau Tidak sebuah Produk ?**.|
| Solusi machine learning | Kita dapat menyelesaikan masalah menentukan seseorang akan membeli atau tidak dengan memanfaatkan Machine Learning. Melalui Machine Learning akan memprediksi seseorang lebih berpeluang membeli atau tidak berdasarkan beberapa informasi sebagai inputnya.|
| Metode pengolahan | Dimulai dengan pengolahan data yang memiliki label 0 (not purchase) dan 1 (purchase). Kemudian melakukan tahapan pengembangan dan validasi model, pengembangan disini dilakukan dengan bereksperimen untuk mencari parameter terbaik (*Tuning Parameters*) lalu menggunakannya dalam proses training model kemudian analisis validasi model untuk memastikan model sudah baik sehingga dapat dilanjutkan ke tahap berikutnya. Tahapan terakhir yaitu Penerapan model (*Deployment*) dengan mencoba menerapkan pada lingkungan **cloud*. |
| Arsitektur model | Arsitektur model ini menerapkan konsep Jaringan Saraf Tiruan yang terdiri dari 3 komponen layer yaitu input layer, hidden layer dan output. |
| Metrik evaluasi | Metrik yang disediakan sebagai bahan evaluasi model adalah Example Count, False Positives, True Positives, False Negatives, True Negatives, dan yang utama yaitu Binary Accuracy |
| Performa model | Performa yang diperoleh model yaitu melalui metrics Binary Accruacy untuk training dan validasi masing-masing adalah 89,52% dan 89,43%. Performa ini sudah sangat baik sebab sudah dapat memberikan performa mendekati 90% untuk training dan validasi (tidak *underfitting*). Selain itu, model kita tidak mengalami *overfitting* pula sebab perbedaan performa training dan validasi sangat kecil (hanya 0.09%). Harapannya ke depan dapat dikembangkan lebih lanjut. Selain itu, alur pengembangan machine learning lanjutannya akan lebih mudah karena telah menerapkan Tensorflow Extension (TFX).|
| Opsi deployment | Pada proyek Machine Learning ini telah mencoba di deployment menggunakan sebuah platform Railway |
| Web app |[Purchase-Model](https://pamlops-production.up.railway.app/v1/models/purchase-model/metadata)|
| Monitoring | Pada proyek Machine Learning ini telah mencoba dimonitoring menggunakan Prometheus yang disinkronkan dengan Grafana untuk menghasilkan sebuah dashboard monitoring yang menarik. Dengan Menerapkan Grafana kita dapat melihat banyak informasi terkait proses deployment, salah satunya adalah Fluktuasi Penggunaan Memory RSS dari waktu ke waktu. |