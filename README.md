# Minesweeper Solver AI

Bu proje, tarayici tabanli Minesweeper oyununu Selenium ile kontrol eden ve oyunun durumunu okuyarak Deep Q-Network (DQN) ile hamle secmeyi ogrenmeye calisan bir yapay zeka calismasidir.

Proje, Yapay Sinir Aglari dersi icin 4 sprintlik plana gore ilerlemektedir. Bu repo su anda Sprint 1 altyapisi uzerine Sprint 2 DQN ilk surumunu icerir.

## Proje Ozeti

- Problem: Minesweeper tahtasinda mayina basmadan guvenli hucreleri secen bir ajan gelistirmek.
- Baseline: Rastgele hucre secen ajan.
- Onerilen yontem: PyTorch ile MLP tabanli Deep Q-Network.
- Veri kaynagi: Hazir veri seti yoktur; oyun durumu Selenium ve board okuma akisi ile anlik uretilir.
- Basari olcutleri: kazanma orani, ortalama reward, mayina basmadan atilan ortalama adim sayisi.

## Sprint 1 Ozeti

Sprint 1'in amaci veri hazirligi ve baseline kurulumuydu.

Yapilan calismalar:

- Selenium ile `minesweeper.online` sayfasi acildi ve Beginner oyun modu baslatildi.
- `SeleniumController` ile oyun tahtasindan ekran goruntusu alma ve hucre tiklama altyapisi kuruldu.
- `TemplateMatcher` ile hucre template'lerini OpenCV `matchTemplate` kullanarak tanima denemeleri yapildi.
- `templates/` klasorune hucre gorselleri eklendi: `zero`-`eight`, `mine`, `flag`, `unsolved`, `reset`, `gg` vb.
- Rastgele hamle yapan baseline ajan ile ilk referans sonuc olusturuldu.

Sprint 1 raporundaki baseline bulgusu:

- Rastgele ajan kazanma orani: yaklasik `%0`
- Ortalama hayatta kalma: yaklasik `3.4` adim
- TemplateMatcher dogrulugu: raporda yaklasik `%98`

Sprint 1 sonunda proje, tarayiciyi acabilen, tahtayi okuyabilen ve rastgele hamlelerle baseline uretebilen bir hale geldi.

## Sprint 2'de Yapilanlar

Sprint 2'nin hedefi, onerilen DQN yonteminin ilk calisan surumunu kurmaktir.

Bu sprintte yapilan ana isler:

- `QNetwork` sinifi PyTorch MLP modeli olarak tamamlandi.
- `ReplayMemory` ile experience replay buffer eklendi.
- `DQNAgent` icinde policy network, target network, epsilon-greedy action secimi, action masking, Adam optimizer ve Huber loss akisi kuruldu.
- `MinesweeperEnv` icinde `reset`, `step`, reward ve `done` mantigi tamamlandi.
- Beginner tahta boyutu `9x9` olacak sekilde egitim akisi tutarli hale getirildi.
- `train.py` yeniden duzenlenerek seed sabitleme, kisa smoke training, metrik kaydi ve checkpoint kaydi eklendi.
- Egitim sonunda `logs/sprint2_metrics.csv` ve `checkpoints/sprint2_dqn.pt` uretilir hale getirildi.

Onemli teknik not:

Mevcut template dosyalari `34x34`, fakat site Selenium ile acildiginda hucreler `24px` zoom seviyesinde gorunebiliyor. Bu nedenle Sprint 2'de asil calisan state okuma yolu DOM uzerinden hucre class'larini okumaktir. `TemplateMatcher` ise 2D board matrisi donduren fallback olarak korunmustur.

## Proje Yapisi

```text
.
├── train.py
├── requirements.txt
├── current_board.png
├── templates/
│   ├── zero.png
│   ├── one.png
│   ├── ...
│   └── unsolved.png
├── src/
│   ├── agent/
│   │   ├── dqn_agent.py
│   │   ├── q_network.py
│   │   └── replay_memory.py
│   ├── env/
│   │   ├── minesweeper_env.py
│   │   ├── selenium_controller.py
│   │   └── template_matcher.py
│   └── utils/
│       ├── config.py
│       └── tensorboard_logger.py
├── logs/
└── checkpoints/
```

## Kurulum

Python sanal ortam kullanilmasi onerilir.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Firefox ve Selenium WebDriver ortaminin calisiyor olmasi gerekir. Proje Firefox uzerinden `minesweeper.online` sitesini acar.

## Projeyi Calistirma

Varsayilan Sprint 2 smoke egitimi:

```bash
python train.py
```

Daha kisa test icin:

```bash
python train.py --episodes 1 --max-steps 2 --cpu
```

Daha uzun egitim icin:

```bash
python train.py --episodes 50 --max-steps 40
```

Temel parametreler:

- `--episodes`: oynanacak episode sayisi
- `--max-steps`: bir episode icindeki maksimum hamle sayisi
- `--rows`: tahta satir sayisi, varsayilan `9`
- `--cols`: tahta sutun sayisi, varsayilan `9`
- `--batch-size`: replay memory batch boyutu
- `--learning-rate`: Adam optimizer learning rate degeri
- `--target-update-freq`: target network guncelleme sikligi
- `--cpu`: CUDA olsa bile CPU kullanir

## Uretilen Ciktilar

Egitimden sonra asagidaki dosyalar uretilir:

- `logs/sprint2_metrics.csv`: episode, step, reward, average loss, epsilon ve done bilgileri
- `checkpoints/sprint2_dqn.pt`: policy network, target network, optimizer state ve epsilon bilgisini iceren checkpoint
- `current_board.png`: Selenium tarafindan alinan son ekran goruntusu

## Dogrulama

Statik Python derleme kontrolu:

```bash
python -m py_compile train.py src/env/*.py src/agent/*.py
```

Hizli Selenium smoke testi:

```bash
python train.py --episodes 1 --max-steps 2 --cpu
```

Beklenen davranis:

- Tarayici acilir.
- Beginner Minesweeper secilir.
- Ajan 1-2 hamle yapar.
- Konsolda episode, adim, reward, epsilon ve loss bilgisi gorunur.
- `logs/` ve `checkpoints/` altinda cikti dosyalari olusur.

## Mevcut Sinirliliklar

- Egitim Selenium uzerinden yapildigi icin yavas calisir.
- DQN modeli MLP oldugu icin 2D komsuluk bilgisini dogrudan CNN kadar iyi kullanamaz.
- Minesweeper eksik bilgi iceren bir oyun oldugu icin bazi durumlarda hamleler olasiliksal risk tasir.
- Template matching, site zoom seviyesi degistiginde dogrudan calismayabilir; bu nedenle Sprint 2'de DOM tabanli okuma tercih edilmistir.
