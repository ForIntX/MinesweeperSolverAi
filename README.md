# Minesweeper Solver AI — DQN Tabanlı Otonom Mayın Temizleme Botu

> **Yapay Sinir Ağları (YSA) Dersi Final Projesi**  
> Derin Q-Öğrenmesi (DQN) ve Kural Tabanlı (Heuristic) algoritmaları harmanlayarak Minesweeper oynamayı öğrenen otonom bir ajan.

---

## 📋 İçindekiler

1. [Proje Özeti](#proje-özeti)
2. [Mimari ve Çalışma Mantığı](#mimari-ve-çalışma-mantığı)
3. [Kurulum](#kurulum)
4. [Nasıl Eğitilir? (Train)](#nasıl-eğitilir-train)
5. [Nasıl Oynatılır? (Demo)](#nasıl-oynatılır-demo)
6. [Canlı Site vs Lokal Mod (IPBAN Ayarı)](#canlı-site-vs-lokal-mod-ipban-ayarı)
7. [Elde Edilen Sonuçlar](#elde-edilen-sonuçlar)
8. [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
9. [Proje Yapısı](#proje-yapısı)

---

## Proje Özeti

Bu proje, klasik Minesweeper oyununu bir insan gibi mantıklı adımlarla ve tıkandığı yerlerde derin öğrenme modelleri ile oynayan hibrit bir Derin Q-Ağı (Deep Q-Network, DQN) ajanı geliştirmektedir. Ajan:

- Özel olarak yazılmış, hızlı bir **Python tabanlı Minesweeper simülatöründe** eğitilir.
- **Double DQN** ve **Deneyim Tekrarı (Experience Replay)** tekniklerini kullanır.
- Görsel sunum aşamasında Selenium kullanarak **ya doğrudan orijinal `minesweeper.online` web sitesinde** canlı olarak, **ya da** IP engellemesine takılmamak için **lokal HTML simülasyonunda (`minesweeper_local.html`)** oynayabilme esnekliğine sahiptir.
- Gerçek bir oyuncu gibi **deterministik kurallarla (emin olunan mayınları bayraklama ve etrafındaki güvenli blokları otomatik açma)** oynar, kural tabanlı gidişat tıkandığında (tahmin/pattern algısı gerektiğinde) DQN yapay sinir ağına başvurarak anlık karar alır.

---

## Mimari ve Çalışma Mantığı

Proje iki ana beynin birleşiminden oluşur:

1. **Deterministik (İnsan) Mantığı:** Ajan ekrandaki sayılara bakar, "buradaki 1 sayısının tek bir kapalı komşusu var, o zaman kesin mayındır" kuralıyla tüm kesin mayınları bayraklar ve kesin güvenli hücreleri tıklar.
2. **Yapay Sinir Ağı (DQN) Mantığı:** Yüzde 100 kesin karar verilemeyen durumlarda, devreye Eğitilmiş DQN modeli girer ve tüm tahtanın anlık durumuna bakarak hangi bloğun açılmasının "en mantıklı risk" olduğuna veya hangi pattern'in mayın sakladığına karar verir.

```text
┌─────────────────────────────────────────────────────────────────┐
│                        DQN Ajanı                                │
│                                                                 │
│  State (81,) ──► [256] ──► [256] ──► [128] ──► Q-Values (81,) │
│                    ReLU     ReLU      ReLU                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Kurulum

### 1. Python Ortamı

```bash
# Python 3.9 veya daha güncel bir sürüm önerilir
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. GPU Desteği (İsteğe Bağlı)
Eğer sisteminizde bir NVIDIA ekran kartı varsa PyTorch otomatik olarak GPU (CUDA) hızlandırmasını kullanacaktır. CPU ile de sorunsuz çalışır ancak GPU eğitim aşamasını hızlandırır.

### 3. Chrome Tarayıcısı
Demo aşaması Selenium kullanmaktadır. Sisteminizde standart bir Google Chrome yüklü olması yeterlidir (gerekli Webdriver otomatik olarak indirilir).

---

## Nasıl Eğitilir? (Train)

DQN ajanını baştan eğitmek veya eğitimine kaldığı yerden devam etmek isterseniz `train.py` dosyasını kullanmalısınız. Ajan eğitim esnasında arka planda çalışan matematiksel `MinesweeperEnv` simülatörünü kullanır (görsel tarayıcı açılmaz).

```bash
# Standart eğitimi başlat (varsayılan ayarlarla)
python train.py

# Daha uzun süreli bir eğitim başlat (örn: 50.000 el oyun oynat)
python train.py --episodes 50000

# Eğitime en son kalınan yerden (checkpoint üzerinden) devam et
python train.py --resume
```

Eğitim sonunda `checkpoints/best.pth` (en iyi model) ve `checkpoints/last.pth` (son model) dosyaları güncellenir. Ayrıca `logs/` dizini altında grafikler oluşturulur.

---

## Canlı Site vs Lokal Mod (IPBAN Ayarı)

Ajan varsayılan olarak IP engellerine takılmamak ve hızlı denemeler yapabilmek için **Lokal Modda** çalışır. Ancak ajanın yeteneklerini orijinal `minesweeper.online` canlı web sitesi üzerinde görmek isterseniz, kodu kolayca canlı modda çalışacak şekilde değiştirebilirsiniz.

Ayar için `web/selenium_controller.py` dosyasını açın ve en üstteki `IPBAN` değişkenini değiştirin:

- **Lokalde Oynamak İçin:**  `IPBAN = True`  (Orijinal oyunun yerel kopyası olan `minesweeper_local.html` dosyasını açar)
- **Canlı Sitede Oynamak İçin:**  `IPBAN = False` (Doğrudan `https://minesweeper.online` sunucusuna bağlanır ve oyunu orada oynar)

---

## Nasıl Oynatılır? (Demo)

Eğitilmiş modelinizi gerçek bir tarayıcı ortamında mayın tarlası oynarken izlemek için `demo.py` scriptini kullanabilirsiniz. Betik, tarayıcıyı başlatacak ve (IPBAN ayarınıza göre) otomatik tıklamalara başlayacaktır.

```bash
# Ajanın 5 el arka arkaya oyun oynamasını izle
python demo.py --games 5

# Arka planda Chrome penceresi görünmeden (headless mode) daha hızlı 100 el test et
python demo.py --games 100 --headless

# Farklı bir model (.pth) ağırlık dosyasını test et
python demo.py --model checkpoints/last_model.pth --games 3
```

**Demo Ekranı Terminal Çıktısı Beklentisi:**
```text
[SeleniumController] Tarayıcı başlatıldı...
=======================================================
  OYUN 1
=======================================================
  Adım   1 | Hamle: (5, 6) | Açık: 0 | Toplam Bayrak: 0
  Adım   2 | Hamle: Mantıksal Güvenli (6 hücre) | Açık: 18 | Toplam Bayrak: 3
  Adım   3 | Hamle: (2, 8) | Açık: 21 | Toplam Bayrak: 3
  ...
  ★ OYUN KAZANILDI! (14 adımda)
```

---

## Elde Edilen Sonuçlar

| Metrik | Değer |
|--------|-------|
| Tahta boyutu | 9×9, 10 mayın (Beginner/Kolay mod) |
| Rastgele Ajan Başarısı | ~%0-1 |
| Eğitilmiş Ajan Win Rate | ~%20-30+ |
| Teorik Üst Sınır | ~%35-40 (Oyunun doğası gereği şans limiti) |

Ajan, DQN ve İnsan Mantıksal Kurallarının birleşimi sayesinde "gereksiz yere risk alan" saf sinir ağlarından çok daha stabil, deterministik ve "insansı" bir performans göstermektedir. 

---

## Bilinen Sınırlamalar

1. **Şansa Dayalı (50/50) Çıkmaz Durumlar:** Minesweeper oyununda saf mantıkla veya örüntü tanıma (pattern recognition) ile çözülmesi matematiksel olarak imkânsız olan; sadece şansla tahmin edilmesi gereken %50/%50 ihtimalli bloklar bulunur. Dünyanın en iyi mayın tarlası oyuncusu veya en gelişmiş yapay zekası dahi bu durumlarda mayına basabilir. Teorik sınır olan ~%35'in ötesine geçilememesinin sebebi budur.
2. **Canlı Site Engellemeleri:** Eğer `IPBAN = False` modunda çok fazla oyun oynatırsanız, `minesweeper.online` sitesi bot davranışlarını algılayıp IP adresinize engelleme uygulayabilir. Bu gibi durumlarda `IPBAN = True` yapılarak simülatöre geçiş yapılabilir.
3. **Eksplorasyon ve Ölçeklenebilirlik:** Daha büyük tahtalarda (örneğin 16x30 Expert seviyesinde) durum uzayı çok aşırı büyüyeceği için salt Flatten (1D Vektör) DQN performansı yeterli gelmeyebilir, bu durumda daha derin Evrişimli Sinir Ağlarına (CNN) geçiş yapmak faydalı olabilir.

---

## Proje Yapısı

```text
minesweeper-ai/
├── env/
│   └── minesweeper_env.py      # Arka plan hızlı oyun simülatörü (Gym API benzeri)
├── agent/
│   ├── q_network.py            # PyTorch YSA Modeli Mimarisi
│   ├── replay_memory.py        # Deneyim kaydı (Experience Replay Buffer) 
│   └── dqn_agent.py            # Double DQN Algoritması Ana Motoru
├── web/
│   └── selenium_controller.py  # Chrome üzerinden oyuna bağlanan Selenium modülü
├── checkpoints/                # Eğitilmiş AI model ağırlıkları (.pth dosyaları)
├── logs/                       # Eğitim performansı grafikleri
├── train.py                    # Sinir ağını simülatör üzerinden eğiten ana script
├── demo.py                     # Modeli görsel olarak oynatan sunum/demo betiği
├── minesweeper_local.html      # Orijinal oyunun lokal kopyası olan statik web sayfası
├── requirements.txt            # Python kütüphane bağımlılıkları
└── README.md                   # Proje dökümantasyonu
```

---

*Bu proje, Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning) tekniklerinin ve deterministik uzman sistem kurallarının klasik bilgisayar oyunlarında nasıl harmanlanabileceğini göstermek amacıyla geliştirilmiştir.*
