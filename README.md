# 🚀 Metin Tabanlı Soru-Cevap Sınıflandırma ve Optimizasyon Analizi

⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐ **Bu projede içerisinde bulunduğum Cosmos ekibimizin ytu-ce-cosmos/Turkish-Gemma-9b-T1 modeli kullanılmıştır.**⭐⭐⭐⭐⭐⭐⭐⭐⭐

Projenin temel amacı; yerli büyük dil modelleri (LLM) kullanarak sentetik bir veri kümesi oluşturmak ve bu veriler üzerinde farklı optimizasyon algoritmalarının (GD, SGD, Adam) performansını matematiksel olarak analiz etmektir .

## 📑 Proje Genel Bakışı

Sistem üç ana aşamadan oluşmaktadır:

**• Veri Üretimi:** Turkish-Gemma-9b-T1 modeli ile 100 eğitim ve 100 test örneği (Soru + İyi Cevap + Kötü Cevap) üretimi.

**• Vektörizasyon:** Metinlerin turkish-e5-large modeli ile yüksek boyutlu anlamsal temsil (embedding) vektörlerine dönüştürülmesi.

**• Regresyon ve Optimizasyon:** Lojistik regresyona benzer bir tanh modeli kullanılarak ağırlık parametrelerinin öğrenilmesi ve algoritmaların karşılaştırılması.

## 🏗️ Teknik Mimari

### 1. Veri Hazırlama (Data Generation)
Dil modeli kullanılarak Tarih, Coğrafya ve Teknoloji gibi sözel ağırlıklı konularda sorular üretilmiştir.

**• İyi Cevap (+1):** Bilimsel olarak doğru, kesin ifadeler içeren cevaplar.

**• Kötü Cevap (-1):** Çok kısa, saçma veya yanlış bilgi içeren cevaplar.

### 2. Model Yapısı
Model, giriş olarak soru ve cevap embedding vektörlerinin birleştirilmesini (concat) kullanır. Matematiksel formülasyonu şu şekildedir:

**c\c​ıkıs\c​=tanh(w⋅x)**

Burada x giriş vektörünü, w ise öğrenilecek parametreleri temsil eder.

### 3. Optimizasyon Stratejileri
Proje kapsamında 5 farklı başlangıç w değeri için üç algoritma karşılaştırılmıştır:

**• GD (Batch Gradient Descent):** Tüm veri kümesini kullanarak kararlı ama yavaş güncellemeler yapar.

**• SGD (Stochastic Gradient Descent):** Her adımda rastgele bir örnek seçerek gürültülü ama hızlı bir yol izler.

**• Adam:** Momentum ve adaptif öğrenme oranını birleştirerek en hızlı yakınsamayı sağlar.

## 📊 Analiz ve Görselleştirme

### Performans Grafikleri:
• Eğitim ve test başarısı; Süre vs. Loss ve Epoch vs. Accuracy kriterlerine göre analiz edilmiştir.

• Adam algoritması, doğası gereği en hızlı öğrenen ve en düşük kayıp (loss) değerine ulaşan yöntem olmuştur.

• Başlangıç ağırlıklarının (w) yakınsama hızı üzerindeki etkisi, özellikle sıçramalı artışlarda gözlemlenmiştir.

### t-SNE ile Yörünge Analizi
• Ağırlık parametrelerinin (w1:t) optimizasyon sürecinde izlediği yol, t-SNE algoritması ile 2 boyuta indirgenerek görselleştirilmiştir.

• Farklı başlangıç noktalarından başlanmasına rağmen, algoritmaların benzer minimum bölgelerine yöneldiği kanıtlanmıştır.

## 🛠️ Kullanılan Teknolojiler

**• Dil:** Python 

**• LLM & Embedding:** * ytu-ce-cosmos/Turkish-Gemma-9b-T1 & ytu-ce-cosmos/turkish-e5-large

**• Kütüphaneler:** llama-cpp-python, SentenceTransformer, NumPy, Matplotlib, scikit-learn, PyTorch

**• Platform:** Google Colab (GPU Destekli)
