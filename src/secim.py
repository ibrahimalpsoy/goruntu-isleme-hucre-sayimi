"""Seed ile geliştirme, doğrulama ve test karelerinin seçilmesi.

Kurallar:
- Sequence 01: rastgele 5 doğrulama karesi, kalanlardan rastgele 10 geliştirme karesi.
  İki grup arasında en az 5 kare boşluk olmalı (|dev - val| >= 6).
- Sequence 02: rastgele 3 başlangıç noktası, her birinden ardışık 5 kare.
  Bloklar çakışmamalı ve aralarında en az 5 kare boşluk olmalı.
- Koşul sağlanmazsa aynı rastgele üreteçle yeniden çekiliyor (deterministik tekrar).
Kullanım:  python -m src.secim
"""
import json

import numpy as np
import pandas as pd

from . import ayarlar as A
from .veri import kare_sayisi, veriyi_hazirla


def dev_val_sec(rng, n_kare):
    deneme = 0
    while True:
        deneme += 1
        val = np.sort(rng.choice(n_kare, A.VAL_SAYISI, replace=False))
        uygun = [k for k in range(n_kare) if all(abs(k - v) > A.MIN_BOSLUK for v in val)]
        if len(uygun) < A.DEV_SAYISI:
            continue
        dev = np.sort(rng.choice(uygun, A.DEV_SAYISI, replace=False))
        return [int(x) for x in dev], [int(x) for x in val], deneme


def test_bloklari_sec(rng, n_kare):
    L, deneme = A.TEST_BLOK_UZUNLUGU, 0
    while True:
        deneme += 1
        baslangic = np.sort(rng.choice(n_kare - L + 1, A.TEST_BLOK_SAYISI, replace=False))
        # bir bloğun son karesi ile sonraki bloğun ilk karesi arasında en az 5 kare olmalı
        if all(baslangic[i + 1] - (baslangic[i] + L - 1) - 1 >= A.MIN_BOSLUK
               for i in range(len(baslangic) - 1)):
            return [int(b) for b in baslangic], deneme


def secimi_yap():
    veriyi_hazirla()
    rng = np.random.default_rng(A.SEED)
    n01, n02 = kare_sayisi("01"), kare_sayisi("02")
    dev, val, deneme1 = dev_val_sec(rng, n01)
    baslangic, deneme2 = test_bloklari_sec(rng, n02)

    satirlar = []
    for k in dev:
        satirlar.append(("development", "01", k, "-"))
    for k in val:
        satirlar.append(("validation", "01", k, "-"))
    for b, s in enumerate(baslangic, 1):
        for k in range(s, s + A.TEST_BLOK_UZUNLUGU):
            satirlar.append(("test", "02", k, f"B{b}"))
    df = pd.DataFrame(satirlar, columns=["grup", "sekans", "kare", "test_blogu"])
    df["dosya_adi"] = [f"{s}/t{k:03d}.tif" for s, k in zip(df.sekans, df.kare)]
    df["referans_dosyasi"] = [f"{s}_GT/TRA/man_track{k:03d}.tif" for s, k in zip(df.sekans, df.kare)]
    df["seed"] = A.SEED

    # kontroller
    assert len(set(dev) & set(val)) == 0
    assert min(abs(d - v) for d in dev for v in val) > A.MIN_BOSLUK
    A.SONUC_KLASORU.mkdir(exist_ok=True)
    df.to_csv(A.SONUC_KLASORU.parent / "selected_frames.csv", index=False)
    kayit = {"seed": A.SEED, "seed_hesabi": "29040 + 2026", "sequence01_kare_sayisi": n01,
             "sequence02_kare_sayisi": n02, "dev_val_deneme_sayisi": deneme1,
             "test_deneme_sayisi": deneme2, "development": dev, "validation": val,
             "test_blok_baslangiclari": baslangic,
             "dev_val_en_kucuk_fark": int(min(abs(d - v) for d in dev for v in val))}
    (A.SONUC_KLASORU / "secim_kaydi.json").write_text(json.dumps(kayit, indent=2, ensure_ascii=False))
    return df, kayit


if __name__ == "__main__":
    df, kayit = secimi_yap()
    print(json.dumps(kayit, indent=2, ensure_ascii=False))
    print(df.to_string(index=False))
