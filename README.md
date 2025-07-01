# Klasifikasi-Jenis-Tanaman
## Domain Proyek
Pertanian merupakan sektor yang sangat penting bagi kehidupan manusia, menyediakan pangan dan bahan baku untuk industri. Namun, dengan meningkatnya populasi dan perubahan iklim, tantangan dalam pertanian semakin kompleks. Salah satu tantangan utama adalah pemilihan jenis tanaman yang tepat untuk ditanam di lahan tertentu, yang sangat bergantung pada kondisi tanah dan iklim.

Di Indonesia, pemahaman mengenai status hara tanah seperti Nitrogen (N), Fosfor (P ), Kalium (K), dan pH merupakan kunci untuk meningkatkan produktivitas pertanian. Penelitian oleh [(Siswanto, 2019)](https://doi.org/10.33366/bs.v18i2.1184) menunjukkan bahwa ketersediaan unsur hara yang tepat akan memungkinkan petani untuk mengoptimalkan penggunaan pupuk dan mengurangi biaya, sekaligus menghindari risiko gagal panen.

Dengan kemajuan teknologi, penerapan machine learning dalam pertanian semakin relevan.  [(Liakos et al., 2018)](https://doi.org/10.3390/s18082674) menjelaskan bahwa machine learning dapat digunakan untuk menganalisis data besar dan memberikan rekomendasi yang berbasis pada kondisi lingkungan yang dinamis. Teknologi ini memungkinkan prediksi yang lebih baik tentang faktor-faktor yang mempengaruhi hasil pertanian, sehingga petani dapat membuat keputusan yang lebih cerdas dalam pemilihan jenis tanaman.

Dengan demikian, klasifikasi tanaman berdasarkan kondisi tanah dan iklim sangat penting dalam upaya meningkatkan produktivitas pertanian. Melalui penerapan teknologi yang tepat, petani dapat mengakses informasi akurat mengenai jenis tanaman yang paling sesuai untuk ditanam di lahan mereka. Pendekatan ini membantu petani dalam memilih varietas yang optimal, sehingga meminimalkan risiko kegagalan panen akibat ketidakcocokan. Dengan memanfaatkan data tentang status hara tanah dan iklim, diharapkan para petani dapat meningkatkan hasil pertanian mereka secara berkelanjutan, sekaligus berkontribusi pada ketahanan pangan nasional.

## Business Understanding
Dalam pertanian, pemilihan jenis tanaman yang tepat sangat penting untuk mencapai hasil panen yang optimal. Kondisi tanah dan iklim yang bervariasi seringkali membuat petani kesulitan dalam menentukan pilihan yang terbaik. Berikut ini adalah pernyataan masalah yang dihadapi serta tujuan dari analisis ini.

### Problem Statements
- Pernyataan Masalah
Banyak petani kesulitan dalam menentukan jenis tanaman yang cocok untuk ditanam berdasarkan kondisi tanah dan iklim yang ada. Hal ini dapat menyebabkan hasil panen yang tidak optimal dan kerugian ekonomi bagi petani.

### Goals
- Jawaban pernyataan masalah
Mengembangkan model macnhine learnig untuk memprediksi, yang dapat membantu petani memilih jenis tanaman yang optimal berdasarkan parameter tanah dan iklim, sehingga meningkatkan hasil panen dan mengurangi kerugian.

### Solution statements
- Membuat empat model dan membandingkan mana yang terbaik untuk melakukan prediksi. Model yang digunakan adalah Random Forest, Gradient Boosting, Support Vector Machine, dan K-Nearest Neighbor.
- Mengevaluasi hasil pelatihan dan pengujian menggunakan accuracy score untuk mengukur seberapa baik model tersebut dalam memprediksi label yang benar, serta menggunakan classification report untuk menilai kinerja model.

## Data Understanding
Dataset tersebut adalah data yang berisi informasi tentang parameter untuk memprediksi jenis tanaman mana yang cocok untuk ditanam dilahan pertanian, yang mana memililiki jumlah data 2200 baris dan 8 kolom.  Untuk Dataset dapat diunduh di Kaggle dengan judul [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

### Variabel-variabel pada Crop Recommendation Dataset adalah sebagai berikut:
* N : Merepresentasikan rasio kandungan Nitrogen dalam tanah
* P : Merepresentasikan rasio kandungan Fosfor dalam tanah
* K : Merepresentasikan rasio kandungan Kalium dalam tanah
* temperature : Merepresentasikan suhu dalam derajat Celcius
* humidity : Merepresentasikan kelembaban relatif dalam %
* ph : Merepresentasikan nilai tanah
* rainfall : Merepresentasikan curah hujan dalam mm
* label : Merupakan nama untuk jenis tanaman

### Exploratory Data Analysis
- Memeriksa Statistik pada data numeric pada dataset, yang mana diketahu pada variabel N memiliki nilai 0 untuk nilai minimum nya, yang mana kandungan Nitrogen dalam tanah tidak mungkin bernilai 0. kemudian, saat diperiksa memiliki nilai 0 yang berjumlah 27, yang mana perlu dihapus pada dataset tersebut.
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/Describe.png?raw=true)
- Pada kolom numeric pada setiap variabel memiliki outlier yang cukup banyak, namun hanya variabel N yang tidak memiliki outlier, yang mana perlu dihapus atau dibersihkan. 
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/Outliers.png?raw=true)

- Setelah data bersih, melakukan binning untuk variabel ph untuk mengelompok berdasarkan kadar ph menjadi 5 kategori, yaitu Sangat asam, Agak asam, Netral, Agak basa, Sangat basa. Yang mana diketahui untuk pada dataset tersebut, banyak tanaman dikadar ph agak asam dan netral, diketahui juga bahwa sedikit menaman dikadar sangat basa. 
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/binph.png?raw=true)
- Kemudian, Melihat distribsui suhu pada dataset dengan binning menjadi 3 kategori, yaitu Suhu Rendah, Suhu Ideal, Suhu Tinggi. Yang mana diketahui bahwa tanaman banyak ditanaman pada suhu ideal.
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/suhu.png?raw=true)

## Data Preparation
Dalam tahap Data Preparation, beberapa langkah penting dilakukan untuk memastikan bahwa dataset siap digunakan dalam model machine learning. Proses ini meliputi pemisahan fitur dan target, konversi kategori, standarisasi, pembagian data, serta penanganan ketidakseimbangan kelas. Setiap langkah memiliki peran krusial dalam meningkatkan kualitas data dan kinerja model yang akan dibangun.

- Pada tahap ini mengahpus outlier pada variable numeric, yang mana setiap variabel memiliki banyak outlier, kecuali variabel N yang tidak memiliki outlier. Setelah menghapus outlier dan diperiksa ulang untuk jumlah sampel nya, diketahui bahwa kelas rice memiliki banyak outlier yang membuat kelas memjadi tidak seimbang, hal ini dapat membuat model menjadi bias dan mengalami overfitting, setelah dibersihkan jumlah data berjumlah 1740 baris.
```sh
numeric_cols = data.select_dtypes(include='number').columns

Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

filter_outliers = ~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

data = data[filter_outliers]
```
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/kelas.png?raw=true)
- Melakukan pemisahan fitur dan target, yang mana berfungsi untuk membedakan variabel independen dari variabel dependen yang ingin diprediksi. Yang mana, variabel X untuk fitur dan variabel y untuk target.
```sh
X = data.drop('label', axis=1)
y = data['label']
```
- Melakukan konversi menggunakan label encoder untuk variabel y, yang mana digunakan untuk mengubah kategori teks atau string menjadi angka, di mana setiap kategori unik dalam fitur diwakili oleh nilai numerik. Hal ini dilakukan karena mesin hanya dapat memahami numerik, sehingga data kategori perlu diubah menjadi format yang dapat diproses. Tanpa konversi ini, model tidak akan dapat mengenali pola atau hubungan dalam data, yang dapat mengakibatkan hasil analisis yang tidak akurat. 
```sh
le = LabelEncoder()
y = le.fit_transform(y)
```
- Melakukan standarisasi pada variabel X, yang mana nilai pada data tersebut diubah sehingga memiliki rata-rata 0 dan deviasi standar 1, Yang berfungsi untuk menghilangkan skala yang berbeda antar fitur, yang membantu meningkatkan performa model machine learning dengan memastikan bahwa semua fitur berkontribusi secara seimbang.
```sh
sc = StandardScaler()
X = sc.fit_transform(X)
```
- Melakukan split dari dataset menjadi data train dan data test, yang mana untuk data nya 80% untuk data train dan 20% untuk data test, menggunakan random state untuk mengontrol pengacakan saat membagi dataset.
```sh
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Melakukan SMOTE untuk menangani ketidakseimbang kelas, Karena pada saat cleaning data, pada dataset diketahui untuk kelas rice jumlah datanya menjadi sangat berkurang. Hal ini dapat mengakibatkan model menjadi bias dan overfitting pada kelas mayoritas. Oleh karena itu SMOTE dilakukan untuk meningkatkan jumlah data untuk kelas yang kurang terwakili.
```sh
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled =smote.fit_resample(X_train, y_train)
```
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/SMOTE.png?raw=true)

## Modeling
Dalam proses modeling, berbagai algoritma digunakan untuk memprediksi dan mengklasifikasikan data dengan cara yang berbeda. Setiap algoritma memiliki kelebihan dan kekurangan. Untuk algoritma yang digunakan sebagai berikut :
- Random Forest
Random Forest adalah algoritma ensemble yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi dan mengurangi risiko overfitting. Kelebihannya termasuk kemampuannya untuk menangani data yang hilang dan memberikan estimasi pentingnya fitur yang baik. Model ini juga robust terhadap outlier dan dapat menangani variabel kategorikal dan numerik secara efisien. Meskipun Random Forest sangat kuat, ia juga bisa menjadi lambat dalam melakukan prediksi jika jumlah pohon yang digunakan sangat banyak. Model ini juga cenderung menghasilkan model yang kurang interpretatif dibandingkan dengan pohon keputusan tunggal, sehingga sulit untuk memahami keputusan yang diambil.
```sh
RF_model = RandomForestClassifier()
```
- Gradient Boosting
Gradient Boosting adalah teknik yang sangat efektif untuk klasifikasi, dengan keunggulan menghasilkan model yang lebih akurat, terutama pada dataset yang lebih kecil. Pada model ini menggunakan 100 pohon keputusan, model dapat menangkap pola dalam data secara efektif tanpa overfitting. Learning rate 0.1 memastikan kontribusi moderat dari setiap pohon, sehingga memerlukan lebih banyak pohon untuk mencapai hasil optimal. Batasan kedalaman maksimum 3 menjaga model tetap sederhana dan interpretable, sementara penggunaan random_state=42 menjamin reproduksibilitas hasil, memastikan bahwa eksperimen dapat diulang dengan hasil yang sama. Memiliki kekurangan yaitu, Gradient Boosting lebih rentan terhadap overfitting, terutama jika tidak diatur dengan baik.
```sh
boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
```
- Support Vector Classification (SVC)
SVC sangat efektif dalam menangani data yang memiliki margin klasifikasi yang jelas antara kelas-kelas. Kelebihannya termasuk kemampuan untuk menggunakan berbagai kernel, yang memungkinkan model menangkap pola non-linear dalam data. Penggunaan kernel linear memungkinkan model untuk melakukan klasifikasi dengan cepat dan efisien, terutama pada dataset yang relatif sederhana. Dengan menetapkan random_state=42, hasil pelatihan menjadi konsisten dan dapat direproduksi, memastikan bahwa model dapat diujicobakan dengan cara yang sama setiap kali. SVC memiliki kekurangan yang dapat menjadi kurang efisien pada dataset yang besar, baik dalam hal waktu pelatihan maupun penggunaan memori.
```sh
svm_model = SVC(kernel='linear', random_state=42)
```
- K-Nearest Neighbors (KNN)
KNN adalah algoritma yang sederhana dan mudah dipahami, serta tidak memerlukan pelatihan eksplisit. Kelebihannya termasuk kemampuannya untuk beradaptasi dengan data baru dan tidak terpengaruh oleh asumsi distribusi data. KNN juga dapat memberikan hasil yang baik pada dataset kecil dengan distribusi yang jelas. Pada model KNN menggunakan jumlah tetangga terdekat yang ditetapkan sebanyak 5. Model ini bekerja dengan mengklasifikasikan data baru berdasarkan mayoritas kelas dari lima tetangga terdekatnya dalam ruang fitur. kekurangan  KNN adalah bisa menjadi sangat lambat saat melakukan prediksi pada dataset besar, karena memerlukan pencarian di seluruh dataset untuk setiap prediksi. Sensitivitas terhadap data yang tidak relevan atau noise juga dapat mengurangi akurasi model
```sh
knn_model = KNeighborsClassifier(n_neighbors=5)
```

### Solution statements
Dari empat model tersebut, saya rasa Random Forest adalah pilihan yang terbaik untuk dataset ini. Model ini cocok karena mampu menangani variabel numerik dengan baik dan dapat menangkap interaksi kompleks antar fitur. Kelebihan ini menjadikan Random Forest sangat efektif untuk aplikasi dalam rekomendasi tanaman berdasarkan kondisi tanah dan iklim yang beragam.


## Evaluation
Pada tahap evaluasi model, penting untuk menggunakan metrik yang tepat untuk menilai performa algoritma yang telah diterapkan. Berikut adalah empat metrik evaluasi yang digunakan akurasi, precision, recall, dan F1 score.
1. **Akurasi** 
Akurasi adalah proporsi dari prediksi yang benar di antara total prediksi. Ini menunjukkan seberapa baik model dalam mengklasifikasikan data. 
Formula :
![Akurasi](https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c426738658d748af1b1e5_644af6a24701d43aaecd8771_classification_guide_apc09.png)
Cara kerja:
hitung semua prediksi benar (baik positif maupun negatif), lalu bagi dengan total prediksi. Misalnya, jika dari 100 prediksi, 95 benar, maka akurasinya 95%.

2. **Precision**
Precision mengukur ketepatan dari prediksi positif yang dihasilkan oleh model. Ini menunjukkan seberapa banyak dari prediksi positif yang benar-benar merupakan positif.
![precision](https://miro.medium.com/max/700/1*pDx6oWDXDGBkjnkRoJS6JA.png)
Cara kerjanya: 
Dari semua yang diprediksi positif, berapa banyak yang benar-benar positif. Contoh, jika model memprediksi 20 sampel sebagai tanaman padi dan 18 di antaranya benar, maka precision untuk padi adalah 90% (18/20), artinya 90% prediksi padi adalah akurat.

3. **Recall**
Recall mengukur kemampuan model untuk menemukan semua instance positif dalam dataset. Ini menunjukkan seberapa banyak dari semua kasus positif yang berhasil diprediksi sebagai positif oleh model.
![recall](https://miro.medium.com/max/538/1*OV0hfgCStTI8hy6lAY1SdA.jpeg)
Cara kerjanya:
Dari semua yang seharusnya positif, berapa banyak yang berhasil ditemukan model. Misalnya, jika terdapat 50 sampel jagung asli dan model hanya mengenali 40, maka recall-nya 80% (40/50), menunjukkan 20% sampel jagung tidak teridentifikasi.

4. **F1 score**
F1-Score adalah rata-rata harmonis dari precision dan recall. Ini memberikan keseimbangan antara kedua metrik, terutama ketika ada ketidakseimbangan kelas.
![recall](https://miro.medium.com/v2/resize:fit:1400/1*c0rbp5CU51h5JL4mA4nVZQ.jpeg)
Cara kerjanya: 
Gabungkan presisi dan recall dalam satu angka. Jika presisi 70% dan recall 50%, maka F1-Score-nya 58%. Angka ini penting untuk data tidak seimbang karena tidak hanya melihat akurasi atau presisi saja. Contoh, jika suatu jenis tanaman memiliki precision 85% dan recall 75%, maka F1-Score-nya adalah 80%, menunjukkan performa model yang seimbang dalam mengidentifikasi jenis tanaman tersebut.

Pada saat setelah ditrain dengan menggunakan empat model tersebut, selanjutnya melihat evaluasi dari hasil data latih dengan data test, menggunakan accuracy score yang digunakan untuk mengukur seberapa baik model klasifikasi dalam memprediksi label yang benar. Untuk hasilnya sebagai berikut :
|  | train | test |
| ------ | ------ | ------ |
| RandomForest | 1.0 | 0.991379 |
| Boosting | 1.0 | 0.974138 |
| SVC | 0.986628 | 0.985632 |
| KNN | 0.988372 | 	0.982759 |

Setelah melihat accuracy, diketahui bahwa semua model memliki accuracy yang tinggi untuk data latih data test. Namun pada RandomForest dan Boosting memiliki skor (1.0) di data train, yang mungkin mengindikasikan overfitting. Namun, karena skor test-nya tetap sangat tinggi, overfitting tidak terlihat sebagai masalah serius dalam kasus ini. Selanjutnya adalah menguji ke empat model untuk melakukan prediksi, yang mana hasilnya sebagi berikut :

|   y_true   | prediksi_RandomForest | prediksi_Boosting | prediksi_SVC | prediksi_KNN |
|-----|--------|---------|--------------|--------|
| mungbean| mungbean| mungbean| mungbean| mungbean|
| cotton| cotton| cotton| cotton| cotton|
| mango | mango| mango| mango| mango|
| mango| mango | mango| mango | mango |
| lentil| lentil| lentil| lentil| lentil|

Dari hasil prediksi tersebut diketahui bahwa, semua model dapat meprediksi dengan benar dengan semua data yang diprediksi. yang mana semua model cocok untuk kalsifikasi pada dataset ini. Namun, saya akan memilih model Random Forest yang terbaik, karena memiliki accuracy yang paling tinggi dari semua model dari data latih dan data test.

Dari empat model tersebut, didapat hasil dari metrik evaluasi sebagai berikut :
1. Random Forest
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/RandomForest.png?raw=true)
Model Random Forest menunjukkan performa sangat kuat dengan akurasi 99%. Hampir semua kelas tanaman seperti banana, mango, dan coconut diprediksi dengan sempurna, dengan precision dan recall mencapai 1.00. Namun, ada kelemahan pada kelas tertentu. Untuk tanaman jute, recall-nya sempurna (1.00), tetapi precision hanya 0.88, artinya beberapa sampel bukan jute salah diklasifikasikan. Pada tanaman rice recall nya hanya 0.50, menunjukkan setengah dari sampel rice tidak terdeteksi, meskipun precision-nya sempurna (1.00).

2. Gradient Boosting
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/Boosting.png?raw=true)
Model Boosting mencapai akurasi 97%, sedikit lebih rendah dibandingkan Random Forest. Namun, pada model terdapat ketidakkonsistenan pada beberapa kelas. Blackgram memiliki precision 0.85, artinya sekitar 15% prediksi blackgram ternyata salah. Mothbeans menunjukkan masalah serius dengan recall hanya 0.68, yang berarti hampir sepertiga sampel mothbeans tidak terdeteksi. Di sisi positif, Boosting berhasil mendeteksi 75% sampel rice (recall 0.75), lebih baik dibandingkan Random Forest. Namun, gap antara akurasi training (100%) dan testing (97%) mengindikasikan adanya potensi overfitting, terutama pada kelas seperti mothbeans dan blackgram.

3. Support Vector Classification (SVC)
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/SVC.png?raw=true)
Model SVC menghasilkan akurasi setinggi 99%, sebanding dengan Random Forest. Namun, SVC mengalami kesulitan dengan tanaman jute, di mana precision dan recall sama-sama hanya 0.86, menunjukkan bahwa model sering keliru dalam mengidentifikasi jute. Untuk rice, precision turun drastis menjadi 0.60, meskipun recall-nya lebih baik (0.75). Artinya, ketika SVC memprediksi suatu sampel sebagai rice, hanya 60% yang benar, tetapi model mampu mendeteksi 75% sampel rice yang ada.

4. K-Nearest Neighbors (KNN)
![alt text](https://github.com/Donsss/IMAGE/blob/main/MLT/KNN.png?raw=true)
Model KNN mencapai akurasi 98%, lebih rendah dibandingkan Random Forest dan SVC. Seperti model lainnya, KNN sangat akurat untuk kelas mayoritas seperti banana dan mango. Namun, KNN menunjukkan kelemahan signifikan pada kelas minoritas. Untuk rice, recall hanya 0.50 dan precision 0.67, menghasilkan F1-score terendah (0.57) dibandingkan semua model lainnya. Lentil juga bermasalah dengan precision 0.88, meskipun recall-nya sempurna. Performa buruk KNN pada rice dan beberapa kelas lain menunjukkan bahwa model ini sangat sensitif terhadap ketidakseimbangan data.

Dari keempat model yang dievaluasi, Random Forest terbukti sebagai model terbaik dengan akurasi tertinggi (99%) dan performa paling konsisten. Model ini menghasilkan precision dan recall sempurna (1.00) untuk hampir semua kelas tanaman, kecuali pada kelas minoritas seperti rice yang memiliki recall rendah (0.50) karena jumlah sampelnya sangat sedikit. SVC juga mencapai akurasi 99%, tetapi mengalami masalah precision pada rice (0.60) dan jute (0.86). Boosting menunjukkan potensi overfitting dengan performa tidak stabil di beberapa kelas seperti mothbeans. Sementara itu, KNN menjadi model terlemah dengan F1-score terendah untuk rice (0.57). Dengan demikian, Random Forest merupakan pilihan optimal untuk klasifikasi tanaman ini.
