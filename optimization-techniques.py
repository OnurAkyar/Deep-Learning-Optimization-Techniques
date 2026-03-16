from llama_cpp import Llama
import time
import random
import re
import csv

gemma = Llama.from_pretrained(
    repo_id="ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF",
    filename="*Q4_K.gguf",
    verbose=False,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_batch=1024,
)

# =============================================================================
# TEMİZLEME
# =============================================================================
def clean(x):
    if not x:
        return ""
    x = x.replace("<think>", "")
    if "</think>" in x:
        x = x.split("</think>")[-1]

    # LaTeX formatındaki matematiksel ifadeleri temizle
    x = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', x)  # \sqrt{x} -> √(x)
    x = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', x)  # \frac{a}{b} -> (a)/(b)
    x = re.sub(r'\\log_(\d+)', r'log_\1', x)  # \log_2 -> log_2
    x = re.sub(r'\\log', 'log', x)  # \log -> log
    x = re.sub(r'\^\{([^}]+)\}', r'^(\1)', x)  # ^{2} -> ^(2)
    x = re.sub(r'\\times', '×', x)
    x = re.sub(r'\\div', '÷', x)
    x = re.sub(r'\\cdot', '·', x)
    x = re.sub(r'\\', '', x)  # Kalan backslash'leri temizle

    return x.strip()


# =============================================================================
# VERİ ÜRETİCİ FONKSİYON
# =============================================================================
def generate_dataset(batch_size=8, total_count=50, dataset_name="train"):
    """
    Her seferinde batch_size kadar soru üretir, toplam total_count'a ulaşana kadar devam eder.
    """
    dataset = []
    generated_count = 0

    # Parse edici pattern'ler
    pattern_question = re.compile(r"^\s*Soru\s*:\s*(.*)", re.I)
    pattern_good = re.compile(r"^\s*İyi Cevap.*:\s*(.*)", re.I)
    pattern_bad = re.compile(r"^\s*Kötü Cevap.*:\s*(.*)", re.I)

    while generated_count < total_count:
        remaining = total_count - generated_count
        current_batch = min(batch_size, remaining)

        seed_value = random.randint(1, 9_999_999)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Rastgele tohum: {seed_value}\n\n"
                    f"{current_batch} tane kısa AMA biraz daha orta karmaşık düzeyde soru üret.\n"
                    "Her soru için 1 iyi cevap (+1) ve 1 kötü cevap (-1) ver.\n\n"
                    "KONU DAĞILIMI (ÖNEMLİ):\n"
                    "- Soruların çoğu (%70-80): Tarih, Coğrafya, Teknoloji, Sanat, Genel Kültür gibi SÖZEL/BİLGİ ağırlıklı olsun.\n"
                    "- Soruların azı (%20-30): Basit Matematik, Mantık veya Fizik problemleri olabilir.\n\n"
                    "FORMAT KESİNLİKLE ŞÖYLE OLACAK:\n"
                    "Soru: ...\n"
                    "İyi Cevap (+1):\n"
                    "   - Tahmin, belirsizlik, 'olabilir', 'sanırım' gibi ifadeler kullanma.\n"
                    "   - Bilimsel olarak/aritmetik olarak yanlış hiçbir bilgi içermesin.\n"
                    "Kötü Cevap (-1): kısa çok saçma ve yanlış bilgi, açıklama YOK\n\n"
                    "Sadece bu formatta üret. Numara ekleme. Ekstra yazı ekleme."
                )
            }
        ]

        print(f"\n {dataset_name.upper()} kümesi için {current_batch} veri üretiliyor... "
              f"({generated_count + current_batch}/{total_count})")

        response = gemma.create_chat_completion(
            messages=messages,
            max_tokens=1500,
            temperature=0.5,  # Daha dengeli: ne çok sıkı ne çok gevşek
            top_p=0.8,        # Makul çeşitlilik ama kontrollü
            repeat_penalty=1.15,
        )

        raw = response["choices"][0]["message"]["content"]
        cleaned = clean(raw)

        # Parse et
        lines = cleaned.split("\n")
        current = {"question": None, "good": None, "bad": None}

        for line in lines:
            q = pattern_question.match(line)
            if q:
                if current["question"]:
                    dataset.append(current)
                current = {"question": None, "good": None, "bad": None}
                current["question"] = q.group(1).strip()
                continue

            g = pattern_good.match(line)
            if g:
                current["good"] = g.group(1).strip()
                continue

            b = pattern_bad.match(line)
            if b:
                current["bad"] = b.group(1).strip()
                continue

        # Son kayıt
        if current["question"]:
            dataset.append(current)

        generated_count = len(dataset)

        if generated_count < total_count:
            time.sleep(1)

    return dataset[:total_count]


# =============================================================================
# CSV'YE KAYDETME FONKSİYONU 
# =============================================================================
def save_to_csv(dataset, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # başlık satırı - PDF formatına uygun
        writer.writerow(["soru", "cevap", "etiket"])
        for item in dataset:
            # İyi cevap: +1
            writer.writerow([item["question"], item["good"], "+1"])
            # Kötü cevap: -1
            writer.writerow([item["question"], item["bad"], "-1"])


# =============================================================================
# EĞİTİM KÜMESİ OLUŞTUR (50 adet)
# =============================================================================
print("EĞİTİM KÜMESİ OLUŞTURULUYOR...")

train_dataset = generate_dataset(batch_size=8, total_count=50, dataset_name="train")

train_output_file = "train_dataset-2.csv"
save_to_csv(train_dataset, train_output_file)

print(f"\n EĞİTİM KÜMESİ TAMAM! {len(train_dataset)*2} adet soru-cevap '{train_output_file}' dosyasına kaydedildi.")

# =============================================================================
# TEST KÜMESİ OLUŞTUR (50 adet)
# =============================================================================
print("TEST KÜMESİ OLUŞTURULUYOR...")

test_dataset = generate_dataset(batch_size=8, total_count=50, dataset_name="test")

test_output_file = "test_dataset-2.csv"
save_to_csv(test_dataset, test_output_file)

print(f"\n TEST KÜMESİ TAMAM! {len(test_dataset)*2} adet soru-cevap '{test_output_file}' dosyasına kaydedildi.")



import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')

# =============================================================================
# CSV DOSYALARINI YÜKLE
# =============================================================================
def load_csv(file_path):
    """CSV dosyasını yükle (Noktalı virgül ayırıcı ile)"""
    return pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip')


train_data = load_csv('train_dataset-2.csv')
test_data = load_csv('test_dataset-2.csv')

# =============================================================================
# EMBEDDİNG'E ÇEVİR
# =============================================================================
def create_embeddings(data, dataset_name="Dataset"):
    """
    CSV verilerini embedding matrisine çevir
    
    CSV Format: question, answer, label
    
    Returns:
        X: (N, 2*d) boyutunda - concat(soru_embedding, cevap_embedding)
        y: (N,) boyutunda - label'lar (+1 veya -1)
    """
    questions = data['question'].tolist()
    answers = data['answer'].tolist()
    labels = data['label'].tolist()
    
    print(f" {dataset_name}:")
    print(f"   - Toplam örnek: {len(questions)}")
    print(f"   - +1 (iyi): {labels.count(1)}")
    print(f"   - -1 (kötü): {labels.count(-1)}")
    
    # Embedding'leri oluştur
    question_embeddings = model.encode(
        questions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    answer_embeddings = model.encode(
        answers,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    # Concat(soru, cevap)
    X = torch.cat([question_embeddings, answer_embeddings], dim=1)
    
    # X'e bir sütun 1'ler ekleyeceğiz: [soru_emb, cevap_emb, 1]
    ones = torch.ones((X.shape[0], 1), device=X.device)
    X = torch.cat([X, ones], dim=1)
    
    # Numpy array'e çevir
    X = X.cpu().numpy()
    y = np.array(labels)
    
    print(f" Embedding boyutu: {X.shape}")
    print(f"   - Soru embedding: {question_embeddings.shape[1]} boyut")
    print(f"   - Cevap embedding: {answer_embeddings.shape[1]} boyut")
    print(f"   - Bias: 1 boyut")
    print(f"   - Toplam (concat+bias): {X.shape[1]} boyut (2d+1)\n")
    
    return X, y

# =============================================================================
# EĞİTİM SETİ
# =============================================================================
print(" EĞİTİM SETİ İŞLENİYOR")

X_train, y_train = create_embeddings(train_data, "Eğitim Seti")

# Kaydet
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
print(f" X_train.npy ve y_train.npy kaydedildi\n")

# =============================================================================
# TEST SETİ
# =============================================================================
print(" TEST SETİ İŞLENİYOR")

X_test, y_test = create_embeddings(test_data, "Test Seti")

# Kaydet
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print(f" X_test.npy ve y_test.npy kaydedildi\n")


# =============================================================================
# ÖZET
# =============================================================================
print(f" Eğitim Seti:")
print(f"   - X_train: {X_train.shape} (soru+cevap+bias concat)")
print(f"   - y_train: {y_train.shape} (label: +1 veya -1)")
print(f"\nTest Seti:")
print(f"   - X_test: {X_test.shape} (soru+cevap+bias concat)")
print(f"   - y_test: {y_test.shape} (label: +1 veya -1)")
print(f" Model: çıkış = tanh(w*x), w boyutu: ({X_train.shape[1]}, 1)")
print(f" x boyutu: ({X_train.shape[1]}, 1) = (2d+1, 1) = (2*1024+1, 1)")



import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # t-SNE için

# ============================================================
# 1) VERİYİ YÜKLE
# ============================================================
X_train = np.load("X_train.npy")  # shape: (N_train, D)
y_train = np.load("y_train.npy")  # shape: (N_train,)
X_test  = np.load("X_test.npy")   # shape: (N_test, D)
y_test  = np.load("y_test.npy")   # shape: (N_test,)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# Etiketler +1 / -1, yine de float'a çevirelim
unique_labels = np.unique(y_train)
print("Train label set:", unique_labels)

y_train = y_train.astype(float)
y_test  = y_test.astype(float)

D = X_train.shape[1]  # w boyutu (= 2d+1, bias dahil)

# ============================================================
# 2) MODEL, LOSS VE GRADİENT
# ============================================================
def forward(w, X):
    """
    w: (D,)
    X: (N, D)
    return: y_hat ∈ (-1, 1), shape: (N,)
    """
    return np.tanh(X @ w)

def mse_loss(y, y_hat):
    """Mean Squared Error"""
    return np.mean((y - y_hat) ** 2)

def accuracy(y, y_hat):
    """Başarı oranı: sign(y_hat) == y"""
    y_pred = np.sign(y_hat)
    return np.mean(y_pred == y)

def grad_mse_tanh(w, X, y):
    """
    ∂L/∂w, MSE + tanh model için
    L = mean( (y - tanh(Xw))^2 )
    dL/dw = mean_i[ -2 (y_i - y_hat_i) (1 - y_hat_i^2) x_i ]
    """
    y_hat = forward(w, X)          # (N,)
    dL_dyhat = -2 * (y - y_hat)    # (N,)
    dyhat_dz = (1 - y_hat**2)      # (N,)
    factor = dL_dyhat * dyhat_dz   # (N,)
    grad = (factor[:, None] * X).mean(axis=0)  # (D,)
    return grad

# ============================================================
# 3) 5 FARKLI BAŞLANGIÇ w OLUŞTUR
# ============================================================
np.random.seed(42)
initial_ws = [
    np.zeros(D),                           # 1) hepsi 0
    np.random.randn(D) * 0.1,              # 2) küçük normal
    np.random.randn(D) * 0.75,             # 3) büyük normal
    np.random.uniform(-0.1, 0.1, size=D),  # 4) küçük uniform
    np.random.uniform(-0.5, 0.5, size=D),  # 5) büyük uniform
]

print("\n5 farklı başlangıç w hazırlandı.")
for i, w0 in enumerate(initial_ws):
    print(f" w{i}: mean={w0.mean():.4f}, std={w0.std():.4f}")

# ============================================================
# 4) GD (Batch Gradient Descent)
# ============================================================
def train_GD(X_train, y_train, X_test, y_test, w0,
             lr=1e-3, epochs=200):

    w = w0.copy().astype(float)

    history = []
    w_path = []  # B kısmı için
    t0 = time.time()
    updates = 0

    for epoch in range(epochs):
        g = grad_mse_tanh(w, X_train, y_train)
        w -= lr * g
        updates += 1

        # w_t kaydet (epoch sonunda)
        w_path.append(w.copy())

        # metrikler
        y_tr_hat = forward(w, X_train)
        y_te_hat = forward(w, X_test)

        history.append({
            "epoch":    epoch,
            "time_sec": time.time() - t0,
            "updates":  updates,
            "loss_train": mse_loss(y_train, y_tr_hat),
            "loss_test":  mse_loss(y_test,  y_te_hat),
            "acc_train":  accuracy(y_train, y_tr_hat),
            "acc_test":   accuracy(y_test,  y_te_hat),
        })

    w_path = np.vstack(w_path)  # shape: (epochs, D)
    return w, history, w_path

# ============================================================
# 5) SGD (Stochastic Gradient Descent)
# ============================================================
def train_SGD(X_train, y_train, X_test, y_test, w0,
              lr=5e-3, epochs=50, shuffle=True):

    w = w0.copy().astype(float)

    history = []
    w_path = []
    t0 = time.time()
    updates = 0
    N = len(X_train)
    indices = np.arange(N)

    for epoch in range(epochs):
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            xi = X_train[idx:idx+1]
            yi = y_train[idx:idx+1]

            g = grad_mse_tanh(w, xi, yi)
            w -= lr * g
            updates += 1

        # epoch sonu: w_t kaydet ve metrikleri hesapla
        w_path.append(w.copy())

        y_tr_hat = forward(w, X_train)
        y_te_hat = forward(w, X_test)

        history.append({
            "epoch":    epoch,
            "time_sec": time.time() - t0,
            "updates":  updates,
            "loss_train": mse_loss(y_train, y_tr_hat),
            "loss_test":  mse_loss(y_test,  y_te_hat),
            "acc_train":  accuracy(y_train, y_tr_hat),
            "acc_test":   accuracy(y_test,  y_te_hat),
        })

    w_path = np.vstack(w_path)  # shape: (epochs, D)
    return w, history, w_path

# ============================================================
# 6) ADAM
# ============================================================
def train_Adam(X_train, y_train, X_test, y_test, w0,
               lr=1e-3, epochs=200,
               beta1=0.9, beta2=0.999, eps=1e-8):

    w = w0.copy().astype(float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    history = []
    w_path = []
    t0 = time.time()
    updates = 0

    for epoch in range(1, epochs + 1):
        g = grad_mse_tanh(w, X_train, y_train)
        updates += 1

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)

        w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # w_t kaydet
        w_path.append(w.copy())

        y_tr_hat = forward(w, X_train)
        y_te_hat = forward(w, X_test)

        history.append({
            "epoch":    epoch - 1,
            "time_sec": time.time() - t0,
            "updates":  updates,
            "loss_train": mse_loss(y_train, y_tr_hat),
            "loss_test":  mse_loss(y_test,  y_te_hat),
            "acc_train":  accuracy(y_train, y_tr_hat),
            "acc_test":   accuracy(y_test,  y_te_hat),
        })

    w_path = np.vstack(w_path)  # shape: (epochs, D)
    return w, history, w_path

# ============================================================
# 7) 5 FARKLI w × 3 OPTİMİZER = 15 DENEY 
# ============================================================
all_results = []  # her eleman: {"init_id", "opt", 
#                               "w_final", "history", "w_path"}

for init_id, w0 in enumerate(initial_ws):
    print(f"\n=== Başlangıç w #{init_id} ===")

    w_gd, hist_gd, path_gd = train_GD(
        X_train, y_train, X_test, y_test,
        w0, lr=0.075, epochs=400 
    )
    all_results.append({
        "init_id": init_id,
        "opt": "GD",
        "w_final": w_gd,
        "history": hist_gd,
        "w_path": path_gd,
    })
    print(
        f" [GD] Son epoch -> "
        f"train acc: {hist_gd[-1]['acc_train']:.3f}, "
        f"test acc: {hist_gd[-1]['acc_test']:.3f}"
    )

    w_sgd, hist_sgd, path_sgd = train_SGD(
        X_train, y_train, X_test, y_test,
        w0, lr=0.001, epochs=400 
    )
    all_results.append({
        "init_id": init_id,
        "opt": "SGD",
        "w_final": w_sgd,
        "history": hist_sgd,
        "w_path": path_sgd,
    })
    print(
        f" [SGD] Son epoch -> "
        f"train acc: {hist_sgd[-1]['acc_train']:.3f}, "
        f"test acc: {hist_sgd[-1]['acc_test']:.3f}"
    )

    w_adam, hist_adam, path_adam = train_Adam(
        X_train, y_train, X_test, y_test,
        w0, lr=0.001, epochs=400 
    )
    all_results.append({
        "init_id": init_id,
        "opt": "Adam",
        "w_final": w_adam,
        "history": hist_adam,
        "w_path": path_adam,
    })
    print(
        f" [Adam] Son epoch -> "
        f"train acc: {hist_adam[-1]['acc_train']:.3f}, "
        f"test acc: {hist_adam[-1]['acc_test']:.3f}"
    )


# ============================================================
# 8) METRİK GRAFİKLERİ 
# ============================================================
def h2np(history, key):
    """history list -> numpy array (seçilen key)"""
    return np.array([h[key] for h in history])

opt_order = ["GD", "SGD", "Adam"]

for init_id in range(len(initial_ws)):
    res_init = [r for r in all_results if r["init_id"] == init_id]
    histories = {r["opt"]: r["history"] for r in res_init}

    # ---------- 8.a) ZAMAN vs LOSS / ACCURACY ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
    fig.suptitle(f"Başlangıç w#{init_id} - Zaman ekseni", fontsize=14)

    for opt in opt_order:
        hist = histories[opt]
        t   = h2np(hist, "time_sec")
        ltr = h2np(hist, "loss_train")
        lte = h2np(hist, "loss_test")
        atr = h2np(hist, "acc_train")
        ate = h2np(hist, "acc_test")

        axes[0, 0].plot(t, ltr, label=opt, linewidth=2)
        axes[1, 0].plot(t, lte, label=opt, linewidth=2)
        axes[0, 1].plot(t, atr, label=opt, linewidth=2)
        axes[1, 1].plot(t, ate, label=opt, linewidth=2)

    axes[0, 0].set_ylabel("Train Loss (MSE)")
    axes[1, 0].set_ylabel("Test Loss (MSE)")
    axes[1, 0].set_xlabel("Zaman (saniye)")

    axes[0, 1].set_ylabel("Train Accuracy")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_xlabel("Zaman (saniye)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(title="Optimizer")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"w{init_id}_time_vs_metrics.png", dpi=150)
    plt.close(fig)

    # ---------- 8.b) EPOCH vs LOSS / ACCURACY ----------                                                                                                                                       
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
    fig.suptitle(f"Başlangıç w#{init_id} - Epoch ekseni", fontsize=14)

    for opt in opt_order:
        hist = histories[opt]
        e   = h2np(hist, "epoch")
        ltr = h2np(hist, "loss_train")
        lte = h2np(hist, "loss_test")
        atr = h2np(hist, "acc_train")
        ate = h2np(hist, "acc_test")

        axes[0, 0].plot(e, ltr, label=opt, linewidth=2)
        axes[1, 0].plot(e, lte, label=opt, linewidth=2)
        axes[0, 1].plot(e, atr, label=opt, linewidth=2)
        axes[1, 1].plot(e, ate, label=opt, linewidth=2)

    axes[0, 0].set_ylabel("Train Loss (MSE)")
    axes[1, 0].set_ylabel("Test Loss (MSE)")
    axes[1, 0].set_xlabel("Epoch")

    axes[0, 1].set_ylabel("Train Accuracy")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_xlabel("Epoch")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(title="Optimizer")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"w{init_id}_epoch_vs_metrics.png", dpi=150)
    plt.close(fig)

print("\n Metrik grafikleri kaydedildi (time & epoch).")

# ============================================================
# 9) T-SNE GRAFİKLERİ
# ============================================================

for opt in opt_order:
    # Bu optimizer'a ait tüm w_path'leri topla
    res_opt = [r for r in all_results if r["opt"] == opt]
    paths = [r["w_path"] for r in res_opt]  # liste: 5 adet (T_i, D)

    # Hepsini birleştir (t-SNE giriş datası)
    lengths = [p.shape[0] for p in paths]
    W_all = np.vstack(paths)  # shape: (sum(lengths), D)
    
    print(f"\n {opt} için t-SNE hesaplanıyor...")
    print(f"   Toplam {len(W_all)} nokta, {len(paths)} yörünge")
    print(f"   w_path boyutları: {[p.shape for p in paths]}")
    
    # t-SNE parametrelerini optimize et
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(W_all) - 1),
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        init='random'
    )
    
    Z_all = tsne.fit_transform(W_all)  # shape: (sum(lengths), 2)

    # Tekrar 5 yörüngeye böl
    trajs_2d = []
    start = 0
    for L in lengths:
        trajs_2d.append(Z_all[start:start+L])
        start += L

    # Çizim
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"{opt} için t-SNE ile w_t yörüngeleri (5 farklı başlangıç w)", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, traj in enumerate(trajs_2d):
        xs = traj[:, 0]
        ys = traj[:, 1]
        ax.plot(xs, ys, marker=markers[i], linewidth=2.0, markersize=4, 
                label=f"init w#{i}", color=colors[i], alpha=0.7, markevery=5)
        # Başlangıç ve bitiş noktalarını özel işaretle
        ax.scatter(xs[0], ys[0], marker='x', s=150, color=colors[i], linewidth=3, zorder=5)
        ax.scatter(xs[-1], ys[-1], marker='*', s=200, color=colors[i], edgecolors='black', zorder=5)

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"tsne_{opt}_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f" {opt} t-SNE tamamlandı ve kaydedildi")


# ============================================================
# 10) EK: OPTİMİZASYON KARŞILAŞTIRMASI
# ============================================================

# Tüm optimizasyonlar için karşılaştırmalı grafik
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Optimizasyon Algoritmaları Karşılaştırması', fontsize=16)

metrics = ['loss_train', 'acc_train', 'loss_test', 'acc_test']
metric_names = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']

for init_id in range(len(initial_ws)):
    res_init = [r for r in all_results if r["init_id"] == init_id]
    
    for idx, opt in enumerate(opt_order):
        hist = [r["history"] for r in res_init if r["opt"] == opt][0]
        epochs = h2np(hist, "epoch")
        
        for metric_idx, metric in enumerate(metrics):
            row = metric_idx // 2
            col = metric_idx % 2
            values = h2np(hist, metric)
            axes[row, col].plot(epochs, values, label=f'{opt} w#{init_id}', 
                               alpha=0.7, linewidth=1.5)

# Eksen etiketlerini ayarla
for i, name in enumerate(metric_names):
    row = i // 2
    col = i % 2
    axes[row, col].set_ylabel(name)
    axes[row, col].set_xlabel('Epoch')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].legend(fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print("  Optimizasyon karşılaştırma grafiği kaydedildi")

