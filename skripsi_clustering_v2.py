"""
PROFILING GAYA BERMAIN PEMAIN LCK - K-MEANS CLUSTERING MULTI-REGIONAL
Versi skrip dari skripsi_clustering_v2.ipynb

CARA PAKAI
  Setiap sel dipisahkan oleh penanda '# %%' dan banner bernomor.
  Salin isi antar-banner ke dalam satu sel notebook, berurutan dari SEL 01.
  Sel bertanda MARKDOWN disalin ke sel Markdown (buang tanda '# ' di awal baris).
  Sel bertanda KODE disalin ke sel Code apa adanya.

  Alternatif: file ini bisa langsung dibuka sebagai notebook di VS Code
  (Python Interactive) atau JupyterLab dengan ekstensi Jupytext.

  Letakkan oracles_elixir_2025.csv di folder yang sama sebelum menjalankan.
  Total 70 sel: 28 markdown, 42 kode.
"""

# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 01 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# # Profiling Gaya Bermain Pemain LCK — K-Means Clustering Berbasis Data Multi-Regional
#
# **Versi 2.** Perbaikan atas versi sebelumnya:
#
# | Aspek | Versi lama | Versi ini | Alasan |
# |---|---|---|---|
# | Jumlah fitur | 5 | 12 | 5 fitur lama saling berkorelasi (r sampai 0.75) sehingga PC1 menyerap 42–56% variansi → yang terukur kualitas, bukan gaya |
# | Normalisasi | Z per liga+role | **Z global per role** | Z per liga menghapus perbedaan antar-liga *by construction*, membuat RM#4 tidak terjawab |
# | Ruang clustering | 5 fitur mentah | PCA (retensi 80% variansi) | Denoising, rasio n:p lebih sehat |
# | Pemilihan K | Elbow saja | Elbow + Silhouette + DB + CH + Gap Statistic + **stabilitas bootstrap ARI** | Elbow & silhouette saling bertentangan di versi lama tanpa aturan tie-break |
# | Pembanding | — | Ward hierarchical + GMM | Menjawab "kenapa K-Means?" |
# | Validasi domain | — | **Champion pool lift** per cluster | RM#3 menjanjikan validasi domain tapi belum ada mekanismenya |
# | Analisis region | — | Uji chi-square + permutasi liga × cluster | Jawaban langsung RM#4 |
#
# **Catatan data penting:** seluruh 6.190 baris LPL berstatus `datacompleteness = "partial"` sehingga tidak memiliki kolom `golddiffat15`, `xpdiffat15`, `csdiffat15`, `damagemitigatedperminute`, `damagetotowers`, dan `monsterkillsownjungle`/`monsterkillsenemyjungle`. Karena LPL termasuk dalam ruang lingkup penelitian, 12 fitur utama disusun **hanya dari kolom yang lengkap di keenam liga**. Fitur tempo (`at15`) diuji terpisah pada Lampiran B.


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 02 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 0. Setup


# %%
# ==============================================================================
# SEL 03 dari 71  |  KODE
# ==============================================================================
import warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42


# %%
# ==============================================================================
# SEL 04 dari 71  |  KODE
# ==============================================================================
DATA_PATH      = "oracles_elixir_2025.csv"
TIER1_LEAGUES  = ["LCK", "LPL", "LEC", "LCP", "LTA N", "LTA S"]
TARGET_YEAR    = 2025
FOCUS_LEAGUE   = "LCK"
ROLES          = ["top", "jng", "mid", "bot", "sup"]
MIN_GAMES      = 10
PCA_VAR_TARGET = 0.80    # proporsi variansi yang dipertahankan untuk ruang clustering
K_RANGE        = range(2, 9)
K_MIN_MEANING  = 3       # K=2 dianggap terlalu kasar untuk profiling
ARI_THRESHOLD  = 0.60    # ambang stabilitas bootstrap
N_BOOTSTRAP    = 100
BOOT_FRAC      = 0.80

# Override manual bila hasil aturan otomatis dinilai kurang interpretable secara domain.
# Kosongkan ({}) untuk memakai hasil aturan sepenuhnya. Setiap override WAJIB dijustifikasi
# di naskah -- misalnya: "K=4 dipilih meskipun aturan menunjuk K=3 karena cluster keempat
# memiliki champion pool yang jelas berbeda (assassin) dan bermakna secara domain."
K_OVERRIDE = {}   # contoh: {"sup": 4}

# ---------------------------------------------------------------------------
# Nama profil gaya bermain, disusun mengikuti prosedur lima langkah pada
# subbab 3.5.4 naskah: sumbu dominan -> pernyataan fungsi -> padanan definisi
# kelas champion -> uji champion lift -> nama.
#
# Diletakkan di sel konfigurasi karena dipakai oleh sel radar chart dan scatter
# plot yang dijalankan SEBELUM bagian penamaan. Kamus ini berlaku untuk
# k = 4, 3, 5, 3, 4 (top, jng, mid, bot, sup). Bila aturan keputusan
# menghasilkan k yang berbeda, kamus ini harus disesuaikan.
# ---------------------------------------------------------------------------
CLUSTER_NAMES = {
    "top_0": "Penahan Utilitas",
    "top_1": "Pengumpul Sumber Daya Mandiri",
    "top_2": "Penuntas Bersumber Daya Tinggi",
    "top_3": "Garda Depan Ofensif",

    "jng_0": "Fasilitator Bersumber Daya Rendah",
    "jng_1": "Penyerbu Agresif",
    "jng_2": "Pengumpul Sumber Daya Terkendali",

    "mid_0": "Fasilitator Utilitas",
    "mid_1": "Penuntas Berisiko Tinggi",
    "mid_2": "Penyeimbang Terkendali",
    "mid_3": "Penghasil Kerusakan Mandiri",
    "mid_4": "Poros Serangan Dominan",

    "bot_0": "Penopang Bersumber Daya Rendah",
    "bot_1": "Penyerang Seimbang",
    "bot_2": "Poros Serangan Utama",

    "sup_0": "Inisiator Garda Depan",
    "sup_1": "Penyergap Mandiri",
    "sup_2": "Fasilitator Terkendali",
    "sup_3": "Pengendali Penglihatan",
}


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 05 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 1. Muat Data


# %%
# ==============================================================================
# SEL 06 dari 71  |  KODE
# ==============================================================================
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print("Baris mentah:", f"{df_raw.shape[0]:,}", "| Kolom:", df_raw.shape[1])
print("Liga tersedia:", len(df_raw["league"].dropna().unique()), "liga")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 07 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 2. Penyaringan Data
#
# Tahapan penyaringan mengikuti Batasan Masalah:
# 1. Hanya 6 liga Tier 1, tahun 2025
# 2. Hanya Regular Season — buang playoffs dan LCK Cup (turnamen pramusim)
# 3. Hanya baris pemain individu — buang baris ringkasan tim (`position == "team"`)


# %%
# ==============================================================================
# SEL 08 dari 71  |  KODE
# ==============================================================================
d = df_raw[df_raw["league"].isin(TIER1_LEAGUES) & (df_raw["year"] == TARGET_YEAR)].copy()
print(f"Setelah filter liga Tier 1 + tahun {TARGET_YEAR}: {d.shape[0]:,}")

d = d[d["playoffs"] == 0]
d = d[~((d["league"] == "LCK") & (d["split"] == "Cup"))]
print(f"Setelah buang playoffs & LCK Cup           : {d.shape[0]:,}")

d = d[d["position"] != "team"]
print(f"Setelah buang baris ringkasan tim          : {d.shape[0]:,}")

print()
print("Split yang tercakup per liga:")
print(d.groupby("league")["split"].unique().to_string())


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 09 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ### 2.1 Audit kelengkapan data (`datacompleteness`)
#
# Langkah ini tidak ada di versi sebelumnya, padahal menentukan fitur mana yang boleh dipakai.


# %%
# ==============================================================================
# SEL 10 dari 71  |  KODE
# ==============================================================================
print(pd.crosstab(d["league"], d["datacompleteness"]).to_string())

partial_only_cols = ["golddiffat15", "xpdiffat15", "csdiffat15",
                     "damagemitigatedperminute", "damagetotowers",
                     "monsterkillsownjungle", "monsterkillsenemyjungle"]
print()
print("Persentase missing per liga untuk kolom yang tidak universal:")
print((d.groupby("league")[partial_only_cols].apply(lambda x: x.isna().mean() * 100)
        .round(1)).to_string())
print()
print(">> Kolom di atas TIDAK dipakai sebagai fitur utama karena hilang total di salah satu liga.")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 11 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 3. Rekayasa Fitur (Feature Engineering)
#
# ### Prinsip
# 1. **Agregasi rasio-dari-jumlah, bukan rata-rata rasio.** Untuk fitur berbasis rasio, nilai per
#    pemain dihitung sebagai `sum(pembilang) / sum(penyebut)` sepanjang musim, bukan rata-rata
#    rasio per pertandingan. Cara ini membobot pertandingan secara proporsional terhadap durasi /
#    volume, dan menghindari bias dari pertandingan sangat pendek.
# 2. **Semua fitur dinormalisasi terhadap waktu atau terhadap tim**, sehingga sebanding antar
#    pertandingan dengan durasi berbeda.
# 3. **Fitur harus menangkap *gaya*, bukan hanya *jumlah sumber daya*.** Ini perbaikan utama:
#    versi lama hanya memuat variabel keluarga sumber daya sehingga hasilnya menjadi sumbu
#    "banyak resource" vs "sedikit resource".
#
# ### Dua belas fitur
#
# | # | Fitur | Rumus | Dimensi gaya yang diukur |
# |---|---|---|---|
# | 1 | `dpm` | Σ damage / Σ menit | Laju output damage |
# | 2 | `damageshare` | Σ damage pemain / Σ damage tim | Porsi ofensif dalam tim |
# | 3 | `earnedgoldshare` | Σ earned gold pemain / Σ earned gold tim | Porsi sumber daya ekonomi |
# | 4 | `cspm` | Σ total CS / Σ menit | Efisiensi farming |
# | 5 | `dmg_taken_pm` | Σ damage diterima / Σ menit | Posisi garis depan (tank/frontline vs squishy) |
# | 6 | `deaths_pm` | Σ deaths / Σ menit | Frekuensi mengambil risiko |
# | 7 | `death_share` | Σ deaths pemain / Σ deaths tim | Porsi risiko yang ditanggung dalam tim |
# | 8 | `kp` | Σ(kills+assists) / Σ team kills | Partisipasi teamfight |
# | 9 | `kill_ratio` | Σ kills / Σ(kills+assists) | Finisher vs enabler |
# | 10 | `wpm` | Σ wards placed / Σ menit | Pemasangan vision |
# | 11 | `wcpm` | Σ wards killed / Σ menit | Perebutan vision (denial) |
# | 12 | `jungle_cs_share` | Σ monster kills / Σ total CS | Sumber farm: jalur vs hutan (proxy roam/resource sharing) |
#
# **Fitur yang sengaja tidak dipakai** (diuji lalu dibuang karena multikolinearitas, lihat §5):
# `vspm` (komposit dari `wpm`+`wcpm`+control ward, r=0.66–0.83 terhadap `wcpm`),
# `cwpm` (r=0.66–0.92 terhadap `wpm`),
# `dmg_per_gold` (r=0.65–0.76 terhadap `dpm`).


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 12 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ### Cara membaca dua sel berikut
#
# Kedua belas variabel dibentuk melalui **dua jalur yang berbeda**, dan perbedaannya disengaja.
#
# **Tujuh variabel sudah tersedia** sebagai kolom pada dataset Oracle's Elixir: `dpm`, `damageshare`,
# `earnedgoldshare`, `cspm`, `damagetakenperminute`, `wpm`, dan `wcpm`. Ketujuhnya dipakai apa adanya
# lalu **dirata-ratakan antar-pertandingan**, sehingga nilainya dapat ditelusuri langsung ke kolom asli
# tanpa perhitungan perantara.
#
# **Lima variabel tidak tersedia** dan harus diturunkan sendiri: `jungle_cs_share`, `deaths_pm`,
# `death_share`, `kp`, dan `kill_ratio`. Karena memang dihitung sendiri, argumen keterlacakan tidak
# berlaku, sehingga bentuk perhitungan dipilih atas dasar statistik: kelimanya memakai **rasio dari
# jumlah**, yaitu menjumlahkan pembilang dan penyebut sepanjang musim lebih dulu, baru membaginya.
#
# > **Alasannya.** Sebagian penyebut bernilai kecil pada tingkat pertandingan — median `teamkills`
# > adalah 15 tetapi persentil ke-10 hanya 5, sementara hampir 15% pertandingan memiliki
# > `kills + assists` ≤ 3. Rasio dengan penyebut sekecil itu hanya bisa mengambil sedikit nilai
# > diskrit, sehingga merata-ratakannya antar-pertandingan menyerap banyak derau pembulatan.
# > Menjumlahkan lebih dulu membuat penyebutnya besar. Uji reliabilitas pada bagian 6 menunjukkan
# > pendekatan ini tidak pernah lebih buruk, dan jauh lebih baik untuk `kp` dan `kill_ratio`.


# %%
# ==============================================================================
# SEL 13 dari 71  |  KODE
# ==============================================================================
# =============================================================
# LANGKAH 1 - Tentukan sumber setiap variabel
# =============================================================
# Pada tahap ini satu baris masih berarti "satu pemain pada satu game".

d["mins"]          = d["gamelength"] / 60.0     # durasi disimpan dalam detik
d["kills_assists"] = d["kills"] + d["assists"]  # dipakai sbg pembilang & penyebut

# JALUR 1 - sudah tersedia di dataset, dipakai apa adanya lalu dirata-ratakan.
#           kunci = nama variabel, nilai = nama kolom pada dataset.
KOLOM_SIAP_PAKAI = {
    "dpm":             "dpm",
    "damageshare":     "damageshare",
    "earnedgoldshare": "earnedgoldshare",
    "cspm":            "cspm",
    "dmg_taken_pm":    "damagetakenperminute",
    "wpm":             "wpm",
    "wcpm":            "wcpm",
}

# JALUR 2 - tidak tersedia di dataset, dihitung sbg rasio dari jumlah.
#           kunci = nama variabel, nilai = (kolom pembilang, kolom penyebut).
VARIABEL_TURUNAN = {
    "jungle_cs_share": ("monsterkills",  "total cs"),
    "deaths_pm":       ("deaths",        "mins"),
    "death_share":     ("deaths",        "teamdeaths"),
    "kp":              ("kills_assists", "teamkills"),
    "kill_ratio":      ("kills",         "kills_assists"),
}

# urutan mengikuti Tabel 2.1 pada naskah (dikelompokkan menurut dimensi gaya)
FEATURES = ["dpm", "damageshare",                      # keluaran serangan
            "earnedgoldshare", "cspm", "jungle_cs_share",   # sumber daya ekonomi
            "deaths_pm", "death_share",                # pengambilan risiko
            "dmg_taken_pm", "kp", "kill_ratio",        # posisi dalam benturan tim
            "wpm", "wcpm"]                             # penguasaan penglihatan

assert set(FEATURES) == set(KOLOM_SIAP_PAKAI) | set(VARIABEL_TURUNAN)
print(f"{len(KOLOM_SIAP_PAKAI)} variabel diambil langsung dari dataset")
print(f"{len(VARIABEL_TURUNAN)} variabel diturunkan sendiri")
print()
print("Bukti penyebut kecil yang mendasari pilihan rasio-dari-jumlah:")
for kol in ["teamkills", "kills_assists"]:
    q = d[kol]
    print(f"  {kol:<14} median={q.median():>5.0f}  persentil-10={q.quantile(.10):>4.0f}  "
          f"proporsi <= 3: {(q <= 3).mean() * 100:>5.1f}%")


# %%
# ==============================================================================
# SEL 14 dari 71  |  KODE
# ==============================================================================
# =============================================================
# LANGKAH 2 - Agregasi dari tingkat PERTANDINGAN ke tingkat PEMAIN
# =============================================================
KEY = ["playerid", "league", "position"]      # unit analisis
g = d.groupby(KEY)

agg = pd.DataFrame(index=g.size().index)
agg["n_games"] = g.size()

# JALUR 1: rata-rata nilai kolom antar-pertandingan
for nama, kolom in KOLOM_SIAP_PAKAI.items():
    agg[nama] = g[kolom].mean()

# JALUR 2: jumlahkan pembilang & penyebut dulu, baru bagi
for nama, (pembilang, penyebut) in VARIABEL_TURUNAN.items():
    agg[nama] = g[pembilang].sum() / g[penyebut].sum().replace(0, np.nan)

# label identitas, diambil dari game terakhir (tim bisa berubah di tengah musim)
agg["playername"] = g["playername"].last()
agg["teamname"]   = g["teamname"].last()
agg = agg.reset_index()

print("Satu baris = satu pemain, pada satu liga, pada satu peran.")
print("Jumlah unit analisis   :", len(agg))
print("Nilai kosong pada fitur:", int(agg[FEATURES].isna().sum().sum()))
agg[["playername", "league", "position", "n_games"] + FEATURES].head(3).round(3)


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 15 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ### 3.1 Pemain yang berpindah role atau liga
#
# Unit analisis adalah kombinasi **pemain × liga × role**, bukan pemain saja. Konsekuensinya,
# pemain yang berganti role atau pindah liga di tengah musim akan muncul sebagai lebih dari satu
# unit — dan itu memang perilaku yang diinginkan, karena gaya bermainnya di role berbeda memang
# berbeda. Bagian ini mendokumentasikan kasus tersebut agar bisa dicantumkan di naskah.


# %%
# ==============================================================================
# SEL 16 dari 71  |  KODE
# ==============================================================================
multi = agg.groupby("playerid")[["league", "position"]].nunique()
multi = multi[(multi["league"] > 1) | (multi["position"] > 1)]
print(f"Pemain yang tercatat di >1 liga atau >1 role: {len(multi)}")
if len(multi):
    print(agg[agg["playerid"].isin(multi.index)]
          [["playername", "league", "position", "teamname", "n_games"]]
          .sort_values(["playername", "n_games"], ascending=[True, False]).to_string(index=False))


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 17 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 4. Ambang Batas Minimum Pertandingan
#
# Pemain dengan sedikit pertandingan menghasilkan estimasi rata-rata yang tidak stabil.
# Ambang dipilih dengan mempertimbangkan trade-off antara reliabilitas estimasi dan retensi pemain,
# khususnya pada liga yang jumlah pertandingannya memang lebih sedikit (LCP, LTA N, LTA S).


# %%
# ==============================================================================
# SEL 18 dari 71  |  KODE
# ==============================================================================
print("Statistik jumlah game per unit analisis:")
print(agg["n_games"].describe().round(1).to_string())
print()
print(f"{'Ambang':>8} | {'Pemain tersisa':>15} | {'Retensi':>8} | Retensi per liga")
print("-" * 95)
for t in [1, 3, 5, 8, 10, 12, 15, 20, 25]:
    sub = agg[agg["n_games"] >= t]
    per_liga = (sub.groupby("league").size() / agg.groupby("league").size() * 100).round(0)
    s = " ".join(f"{k}:{int(v)}%" for k, v in per_liga.items())
    print(f"{t:>8} | {len(sub):>7} / {len(agg):<5} | {len(sub)/len(agg)*100:>6.1f}% | {s}")


# %%
# ==============================================================================
# SEL 19 dari 71  |  KODE
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(agg["n_games"], bins=30, color="#4C72B0", edgecolor="white")
ax.axvline(MIN_GAMES, color="crimson", ls="--", lw=2, label=f"ambang terpilih = {MIN_GAMES}")
ax.set(title="Distribusi jumlah pertandingan per pemain",
       xlabel="Jumlah pertandingan", ylabel="Jumlah pemain")
ax.legend(); plt.tight_layout(); plt.savefig("fig_01_min_games.png", bbox_inches="tight"); plt.show()

P = agg[agg["n_games"] >= MIN_GAMES].reset_index(drop=True).copy()
print(f"Pemain final: {len(P)} (dari {len(agg)})")
print()
print(P.groupby(["league", "position"]).size().unstack(fill_value=0).to_string())


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 20 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 5. Uji Multikolinearitas Antar Fitur
#
# Langkah ini tidak ada di versi sebelumnya. K-Means berbasis jarak Euclidean memperlakukan setiap
# fitur dengan bobot sama; jika beberapa fitur mengukur hal yang sama, dimensi itu secara efektif
# mendapat bobot ganda dan mendominasi hasil clustering. Inilah penyebab hasil versi lama
# terkonsentrasi pada satu sumbu "banyak/sedikit sumber daya".


# %%
# ==============================================================================
# SEL 21 dari 71  |  KODE
# ==============================================================================
fig, ax = plt.subplots(2, 3, figsize=(9.5, 7.2))
for i, role in enumerate(ROLES):
    a = ax[i // 3][i % 3]
    Cm_ = P[P["position"] == role][FEATURES].corr()
    im = a.imshow(Cm_, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(FEATURES))); a.set_xticklabels(FEATURES, rotation=90, fontsize=5.5)
    a.set_yticks(range(len(FEATURES))); a.set_yticklabels(FEATURES, fontsize=5.5)
    a.set_title(role, fontsize=10, fontweight="bold")
    for x in range(len(FEATURES)):
        for y in range(len(FEATURES)):
            if x != y and abs(Cm_.iloc[x, y]) >= 0.60:
                a.text(y, x, f"{Cm_.iloc[x, y]:.2f}", ha="center", va="center", fontsize=4.6)
ax[1][2].axis("off")
fig.colorbar(im, ax=ax[1][2], fraction=0.5, aspect=14)
plt.tight_layout()
plt.savefig("fig_02_korelasi.png", dpi=200, bbox_inches="tight"); plt.show()

print("Pasangan dengan |r| >= 0,60 (indikasi redundansi):")
for role in ROLES:
    Cm_ = P[P["position"] == role][FEATURES].corr()
    hits = [(Cm_.index[i], Cm_.columns[j], round(Cm_.iloc[i, j], 2))
            for i in range(len(Cm_)) for j in range(i + 1, len(Cm_))
            if abs(Cm_.iloc[i, j]) >= 0.60]
    print(f"  {role:>3}: {hits if hits else 'tidak ada'}")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 22 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 6. Uji Reliabilitas Variabel
#
# Selain saling berkorelasi, sebuah variabel bisa bermasalah karena **nilainya tidak konsisten bagi
# pemain yang sama**. Variabel semacam itu lebih banyak menangkap situasi pertandingan daripada
# kecenderungan pemain, namun setelah standardisasi tetap memperoleh bobot yang sama dengan variabel
# yang andal.
#
# Pengujian memakai **metode belah-dua**: pertandingan tiap pemain dibagi acak menjadi dua bagian,
# nilai variabel dihitung terpisah pada masing-masing bagian, lalu dikorelasikan antar-pemain.
# Koefisiennya dikoreksi dengan formula Spearman-Brown untuk memperhitungkan berkurangnya jumlah
# pertandingan akibat pembelahan.
#
# > **Hasilnya tidak dipakai untuk membuang variabel secara otomatis.** Reliabilitas rendah justru
# > bisa muncul karena variabel berhasil menyingkirkan konteks tim. `kp` dan `kill_ratio`
# > dinormalisasi terhadap capaian tim sehingga tidak lagi mencerminkan tempo permainan tim, dan yang
# > tersisa hanya sinyal individual yang memang lemah. Sebaliknya, variabel berbasis laju per menit
# > seperti kill per menit jauh lebih reliabel, tetapi sebagian besar kestabilannya berasal dari tempo
# > kill timnya — sel terakhir bagian ini membuktikannya.


# %%
# ==============================================================================
# SEL 23 dari 71  |  KODE
# ==============================================================================
def reliabilitas_belah_dua(frame_game, kolom_kunci, seed=RANDOM_STATE):
    """Reliabilitas belah-dua dgn koreksi Spearman-Brown, dihitung per role.

    Pertandingan tiap pemain dibelah acak jadi dua bagian, nilai variabel
    dihitung terpisah pada tiap bagian dengan aturan agregasi yang sama
    seperti model utama, lalu keduanya dikorelasikan antar-pemain.
    """
    f = frame_game.copy()
    rng = np.random.default_rng(seed)
    f["_acak"] = rng.random(len(f))
    f["_bagian"] = f.groupby(kolom_kunci)["_acak"].rank(pct=True) > 0.5

    def nilai(sub):
        gg = sub.groupby(kolom_kunci)
        out = pd.DataFrame(index=gg.size().index)
        for nama, kolom in KOLOM_SIAP_PAKAI.items():
            out[nama] = gg[kolom].mean()
        for nama, (pemb, peny) in VARIABEL_TURUNAN.items():
            out[nama] = gg[pemb].sum() / gg[peny].sum().replace(0, np.nan)
        return out

    hasil = {}
    for role in ROLES:
        s = f[f["position"] == role]
        A, B = nilai(s[~s["_bagian"]]), nilai(s[s["_bagian"]])
        kol = {}
        for v in FEATURES:
            j = pd.concat([A[v].rename("a"), B[v].rename("b")], axis=1).dropna()
            r = j["a"].corr(j["b"])
            kol[v] = 2 * r / (1 + r) if pd.notna(r) else np.nan   # Spearman-Brown
        hasil[role] = kol
    return pd.DataFrame(hasil)


# %%
# ==============================================================================
# SEL 24 dari 71  |  KODE
# ==============================================================================
ROLES = ["top", "jng", "mid", "bot", "sup"]     # dipakai juga di bagian berikutnya
d_final = d.merge(P[KEY], on=KEY, how="inner")  # hanya pemain yg lolos ambang

rel = reliabilitas_belah_dua(d_final, KEY)
rel["rata-rata"] = rel.mean(axis=1)
rel = rel.loc[FEATURES]

print("Reliabilitas belah-dua (terkoreksi Spearman-Brown)")
print("Mendekati 1 = variasi antar-pemain merupakan sinyal stabil.\n")
print(rel.round(2).to_string())

lemah = rel.index[rel["rata-rata"] < 0.5].tolist()
print()
print("Variabel dengan reliabilitas rata-rata di bawah 0,50:", lemah if lemah else "tidak ada")


# %%
# ==============================================================================
# SEL 25 dari 71  |  KODE
# ==============================================================================
# Mengapa variabel yang "lebih reliabel" belum tentu lebih baik:
# alternatif berbasis laju per menit diuji terhadap tempo kill TIM.
gg = d_final.groupby(KEY)
uji = pd.DataFrame({
    "kills_per_menit":   gg["kills"].sum()   / gg["mins"].sum(),
    "assists_per_menit": gg["assists"].sum() / gg["mins"].sum(),
    "kp":                gg["kills_assists"].sum() / gg["teamkills"].sum(),
    "kill_ratio":        gg["kills"].sum() / gg["kills_assists"].sum(),
    "tempo_kill_tim":    gg["teamkills"].sum() / gg["mins"].sum(),
}).reset_index()

print("Korelasi terhadap tempo kill tim (makin tinggi = makin mengukur tim, bukan pemain)\n")
print(f"{'role':>5} | {'kills/mnt':>10} | {'assists/mnt':>12} | {'kp':>7} | {'kill_ratio':>11}")
print("-" * 58)
for role in ROLES:
    s = uji[uji["position"] == role]
    print(f"{role:>5} | {s['kills_per_menit'].corr(s['tempo_kill_tim']):>10.2f} | "
          f"{s['assists_per_menit'].corr(s['tempo_kill_tim']):>12.2f} | "
          f"{s['kp'].corr(s['tempo_kill_tim']):>7.2f} | "
          f"{s['kill_ratio'].corr(s['tempo_kill_tim']):>11.2f}")
print()
print("kp dan kill_ratio nyaris tidak berkorelasi dgn tempo tim -- itu memang tujuannya.")
print("Laju per menit sangat berkorelasi, sehingga kestabilannya menyesatkan bagi")
print("penelitian yang menyasar gaya individu.")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 26 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 7. Standardisasi Z-Score — **Global per Role**
#
# > **Perubahan metodologis paling penting dari versi sebelumnya.**
# >
# > Versi lama menstandardisasi per `liga + role`. Konsekuensinya setiap liga dipaksa memiliki
# > mean 0 dan simpangan baku 1, sehingga **seluruh perbedaan antar-liga terhapus secara
# > struktural** — padahal justru itu yang ditanyakan RM#4. Pengecekan pada data menunjukkan efek
# > liga memang nyata (variansi yang dijelaskan liga mencapai 29% pada `earnedgoldshare` jungler).
# >
# > Di versi ini standardisasi hanya dilakukan **per role**, dengan seluruh liga digabung. Role
# > tetap dipisah karena perbedaan antar-role bersifat struktural (support memang ber-CS rendah),
# > bukan perbedaan gaya. Versi per-liga tetap dijalankan sebagai analisis sensitivitas di Lampiran A.


# %%
# ==============================================================================
# SEL 27 dari 71  |  KODE
# ==============================================================================
Z_COLS = [f"{c}_z" for c in FEATURES]
for c in FEATURES:
    P[f"{c}_z"] = P.groupby("position")[c].transform(lambda x: (x - x.mean()) / x.std())

chk = P.groupby("position")[Z_COLS].agg(["mean", "std"])
print("Sanity check (mean harus ~0, std harus ~1):")
print(chk.iloc[:, :6].round(3).to_string())
print()
print("Proporsi variansi yang dijelaskan LIGA (eta-squared) -- sinyal yang kini DIPERTAHANKAN:")
rows = []
for role in ROLES:
    s = P[P["position"] == role]
    row = {"role": role}
    for c in FEATURES:
        gg  = s.groupby("league")[c]
        sst = ((s[c] - s[c].mean()) ** 2).sum()
        ssb = (gg.count() * (gg.mean() - s[c].mean()) ** 2).sum()
        row[c] = ssb / sst
    rows.append(row)
print(pd.DataFrame(rows).set_index("role").round(3).to_string())


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 28 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 8. Reduksi Dimensi dengan PCA
#
# Berbeda dari versi lama yang memakai PCA hanya untuk visualisasi, di sini PCA juga menjadi
# bagian dari pipeline clustering. Alasannya:
#
# - Dengan 12 fitur dan hanya ~70 pemain per role, rasio observasi terhadap dimensi menjadi rendah;
#   jarak Euclidean di ruang berdimensi tinggi kehilangan daya diskriminasi (*curse of dimensionality*).
# - Komponen dengan variansi kecil umumnya berisi derau pengukuran.
#
# Komponen dipertahankan sampai variansi kumulatif mencapai 80%. Dua komponen pertama tetap
# dipakai untuk scatter plot.
#
# > **Implikasi untuk naskah:** Batasan Masalah butir 4 saat ini menyatakan PCA dipakai
# > "sebagai teknik reduksi dimensi untuk kebutuhan visualisasi". Kalimat itu perlu direvisi
# > menjadi reduksi dimensi untuk clustering *dan* visualisasi.


# %%
# ==============================================================================
# SEL 29 dari 71  |  KODE
# ==============================================================================
pca_models, X_red = {}, {}
loadings_all = {}

for role in ROLES:
    m = P["position"] == role
    X = P.loc[m, Z_COLS].values

    p_full = PCA(random_state=RANDOM_STATE).fit(X)
    cum    = np.cumsum(p_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, PCA_VAR_TARGET) + 1)

    p = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X)
    X_red[role]      = p.transform(X)
    pca_models[role] = p
    loadings_all[role] = pd.DataFrame(p.components_.T,
                                      index=FEATURES,
                                      columns=[f"PC{i+1}" for i in range(n_comp)])
    P.loc[m, "pc1"] = X_red[role][:, 0]
    P.loc[m, "pc2"] = X_red[role][:, 1]
    print(f"{role:>3}: {n_comp} komponen dipertahankan "
          f"({cum[n_comp-1]:.1%} variansi) | PC1={p.explained_variance_ratio_[0]:.1%} "
          f"PC2={p.explained_variance_ratio_[1]:.1%}")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 30 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ### 8.1 Loading PCA — interpretasi sumbu
#
# Tabel loading memberi tahu **arti** setiap komponen utama. Ini penting untuk Bab IV: scatter plot
# tanpa interpretasi sumbu tidak bisa dibahas secara substantif.


# %%
# ==============================================================================
# SEL 31 dari 71  |  KODE
# ==============================================================================
for role in ROLES:
    print(f"\n=== {role.upper()} — loading PC1..PC3 ===")
    print(loadings_all[role].iloc[:, :3].round(2).to_string())
    for pc in loadings_all[role].columns[:3]:
        s = loadings_all[role][pc].sort_values()
        print(f"  {pc}: (-) {', '.join(s.index[:3])}  <-->  (+) {', '.join(s.index[-3:][::-1])}")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 32 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 9. Penentuan Jumlah Cluster (K)
#
# Versi lama memakai dua metode yang **saling bertentangan** (silhouette menunjuk K=2 untuk semua
# role, elbow menunjuk K=4–5) tanpa aturan penyelesaian. Di sini digunakan lima metrik internal
# ditambah satu uji stabilitas, dengan aturan keputusan yang dinyatakan eksplisit.
#
# ### Metrik yang digunakan
#
# | Metrik | Arah optimal | Yang diukur |
# |---|---|---|
# | WCSS / Elbow | titik siku | Kepadatan dalam cluster |
# | Silhouette | maksimum | Kohesi vs separasi |
# | Davies–Bouldin | **minimum** | Rasio sebaran dalam terhadap jarak antar cluster |
# | Calinski–Harabasz | maksimum | Rasio variansi antar terhadap dalam cluster |
# | Gap Statistic | maksimum | Perbandingan terhadap distribusi acuan acak (Tibshirani et al., 2001) |
# | **Bootstrap ARI** | maksimum | **Stabilitas** partisi terhadap perubahan sampel |
#
# ### Aturan keputusan (dinyatakan di muka, bukan setelah melihat hasil)
#
# Pilih **K terbesar** pada rentang `K_MIN_MEANING..8` yang memenuhi keduanya:
# 1. Bootstrap ARI ≥ 0.60 — partisi tergolong stabil
# 2. Cluster terkecil memuat ≥ 5% dari populasi role dan ≥ 5 pemain — tidak ada cluster hantu
#
# Jika tidak ada K yang memenuhi, pilih K dengan ARI tertinggi pada rentang tersebut.
#
# K = 2 dikecualikan karena secara praktis hanya memisahkan "di atas rata-rata" dan "di bawah
# rata-rata", yang tidak menjawab tujuan profiling gaya bermain.


# %%
# ==============================================================================
# SEL 33 dari 71  |  KODE
# ==============================================================================
def fit_kmeans(X, k, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=seed).fit(X)

def find_elbow_k(ks, wcss):
    """Deteksi titik siku otomatis: jarak tegak lurus terbesar ke garis
    yang menghubungkan titik pertama dan terakhir pada kurva ternormalisasi
    (metode Kneedle, Satopaa et al., 2011)."""
    k = np.asarray(ks, float); w = np.asarray(wcss, float)
    kn = (k - k.min()) / (k.max() - k.min())
    wn = (w - w.min()) / (w.max() - w.min())
    p1, p2 = np.array([kn[0], wn[0]]), np.array([kn[-1], wn[-1]])
    v = p2 - p1; ln = np.linalg.norm(v)
    dist = [abs(v[0] * (p1[1] - y) - v[1] * (p1[0] - x)) / ln for x, y in zip(kn, wn)]
    return int(ks[int(np.argmax(dist))])

def gap_statistic(X, k, n_ref=25, seed=RANDOM_STATE):
    """Gap statistic dengan acuan uniform pada bounding box data (Tibshirani et al., 2001)."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    wk = np.log(fit_kmeans(X, k).inertia_)
    refs = []
    for _ in range(n_ref):
        Xr = rng.uniform(lo, hi, size=X.shape)
        refs.append(np.log(fit_kmeans(Xr, k).inertia_))
    refs = np.array(refs)
    sk = refs.std() * np.sqrt(1 + 1 / n_ref)
    return refs.mean() - wk, sk

def bootstrap_ari(X, k, n_boot=N_BOOTSTRAP, frac=BOOT_FRAC, seed=RANDOM_STATE):
    """Stabilitas: bandingkan label penuh vs label hasil clustering subsample."""
    base = fit_kmeans(X, k).labels_
    rng  = np.random.default_rng(seed)
    n    = len(X); out = []
    for b in range(n_boot):
        idx = rng.choice(n, int(n * frac), replace=False)
        lb  = fit_kmeans(X[idx], k, seed=b).labels_
        out.append(adjusted_rand_score(base[idx], lb))
    return float(np.mean(out)), float(np.std(out))


# %%
# ==============================================================================
# SEL 34 dari 71  |  KODE
# ==============================================================================
diag = {}
for role in ROLES:
    X = X_red[role]; n = len(X)
    rows = []
    for k in K_RANGE:
        km  = fit_kmeans(X, k)
        lab = km.labels_
        gap, sk = gap_statistic(X, k)
        ari_m, ari_s = bootstrap_ari(X, k)
        rows.append({
            "K": k,
            "WCSS": km.inertia_,
            "silhouette": silhouette_score(X, lab),
            "davies_bouldin": davies_bouldin_score(X, lab),
            "calinski_harabasz": calinski_harabasz_score(X, lab),
            "gap": gap, "gap_sk": sk,
            "ARI_boot": ari_m, "ARI_sd": ari_s,
            "min_cluster": int(np.bincount(lab).min()),
            "min_cluster_pct": np.bincount(lab).min() / n,
        })
    diag[role] = pd.DataFrame(rows).set_index("K")
    print(f"\n=== {role.upper()} (n={n}) ===")
    print(diag[role].round(3).to_string())


# %%
# ==============================================================================
# SEL 35 dari 71  |  KODE
# ==============================================================================
optimal_k, elbow_k = {}, {}
print("Penerapan aturan keputusan:\n")
for role in ROLES:
    t = diag[role]
    elbow_k[role] = find_elbow_k(list(K_RANGE), t["WCSS"].values)
    cand = t[(t.index >= K_MIN_MEANING) &
             (t["ARI_boot"] >= ARI_THRESHOLD) &
             (t["min_cluster_pct"] >= 0.05) &
             (t["min_cluster"] >= 5)]
    if len(cand):
        k_sel, why = int(cand.index.max()), f"K terbesar dengan ARI>={ARI_THRESHOLD} & cluster terkecil layak"
    else:
        sub = t[t.index >= K_MIN_MEANING]
        k_sel, why = int(sub["ARI_boot"].idxmax()), "fallback: ARI tertinggi (tidak ada K yang lolos ambang)"
    if role in K_OVERRIDE:
        k_sel, why = K_OVERRIDE[role], f"OVERRIDE MANUAL (aturan menunjuk K={k_sel})"
    optimal_k[role] = k_sel
    print(f"  {role:>3}: K={k_sel}  (ARI={t.loc[k_sel,'ARI_boot']:.3f}, "
          f"silhouette={t.loc[k_sel,'silhouette']:.3f}, "
          f"cluster terkecil={t.loc[k_sel,'min_cluster']}) | elbow menunjuk K={elbow_k[role]} | {why}")
print("\nK final per role:", optimal_k)


# %%
# ==============================================================================
# SEL 36 dari 71  |  KODE
# ==============================================================================
def kurva_metrik(kolom, judul_y, berkas, warna, galat=None, ambang=None, arah="maks"):
    """Satu berkas gambar untuk satu metrik, memuat kelima peran dalam grid 2x3."""
    fig, ax = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for i, role in enumerate(ROLES):
        a = ax[i // 3][i % 3]
        s = diag[role].reset_index().sort_values("K")
        if galat:
            a.errorbar(s["K"], s[kolom], yerr=s[galat], marker="o", ms=4, lw=1.3,
                       color=warna, capsize=3)
        else:
            a.plot(s["K"], s[kolom], marker="o", ms=4, lw=1.3, color=warna)
        k = optimal_k[role]
        a.axvline(k, color="crimson", ls="--", lw=1.1, label=f"k terpilih = {k}")
        if ambang is not None:
            a.axhline(ambang, color="dimgrey", ls=":", lw=1.1, label=f"ambang {ambang:.2f}")
        idx = s[kolom].idxmax() if arah == "maks" else s[kolom].idxmin()
        lbl = "nilai tertinggi" if arah == "maks" else "nilai terendah"
        a.plot(s.loc[idx, "K"], s.loc[idx, kolom], marker="*", ms=13, color="black",
               zorder=5, ls="none", label=f"{lbl} (k={int(s.loc[idx, 'K'])})")
        a.set_title(role, fontsize=10, fontweight="bold")
        a.set_xlabel("Jumlah klaster (k)", fontsize=8)
        a.set_ylabel(judul_y, fontsize=8)
        a.tick_params(labelsize=7.5); a.grid(alpha=.25, lw=.5)
        a.legend(fontsize=6.6, loc="best")
    ax[1][2].axis("off")
    plt.tight_layout(pad=0.6)
    plt.savefig(berkas, dpi=200, bbox_inches="tight"); plt.show()

# Satu metrik satu gambar agar setiap kurva dapat dicermati terpisah.
kurva_metrik("WCSS",              "WCSS",                      "fig_03a_wcss.png",       "tab:blue",   arah="min")
kurva_metrik("silhouette",        "Silhouette Score",          "fig_03b_silhouette.png", "tab:orange")
kurva_metrik("davies_bouldin",    "Indeks Davies-Bouldin",     "fig_03c_davies.png",     "tab:purple", arah="min")
kurva_metrik("calinski_harabasz", "Indeks Calinski-Harabasz",  "fig_03d_calinski.png",   "tab:brown")
kurva_metrik("gap",               "Gap Statistic",             "fig_03e_gap.png",        "tab:green",  galat="gap_sk")
kurva_metrik("ARI_boot",          "Adjusted Rand Index",       "fig_03f_ari.png",        "tab:red",
             galat="ARI_sd", ambang=ARI_THRESHOLD)


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 37 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 10. Eksekusi K-Means Final
#
# Label cluster diurutkan ulang berdasarkan rata-rata PC1 agar **deterministik**. Tanpa langkah ini,
# penomoran cluster bergantung pada inisialisasi acak, sehingga peta nama gaya bermain bisa
# tertukar diam-diam ketika `random_state` diubah.


# %%
# ==============================================================================
# SEL 38 dari 71  |  KODE
# ==============================================================================
models = {}
P["cluster"] = -1

for role in ROLES:
    m = P["position"] == role
    X = X_red[role]
    km = fit_kmeans(X, optimal_k[role])
    lab = km.labels_

    order  = np.argsort([X[lab == c, 0].mean() for c in range(optimal_k[role])])
    remap  = {old: new for new, old in enumerate(order)}
    lab    = np.array([remap[l] for l in lab])

    P.loc[m, "cluster"] = lab
    models[role] = km
    print(f"{role:>3} (K={optimal_k[role]}): ukuran cluster = {np.bincount(lab).tolist()}")

P["role_cluster"] = P["position"] + "_" + P["cluster"].astype(str)


# %%
# ==============================================================================
# SEL 39 dari 71  |  KODE
# ==============================================================================
print("Centroid dalam skala Z (untuk radar chart & interpretasi):\n")
centroids_z = P.groupby(["position", "cluster"])[Z_COLS].mean()
centroids_z.columns = [c[:-2] for c in centroids_z.columns]
print(centroids_z.round(2).to_string())


# %%
# ==============================================================================
# SEL 40 dari 71  |  KODE
# ==============================================================================
print("Centroid dalam skala ASLI (untuk pelaporan di naskah):\n")
centroids_raw = P.groupby(["position", "cluster"])[FEATURES].mean()
print(centroids_raw.round(3).to_string())


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 41 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 11. Pembanding: Ward Hierarchical & Gaussian Mixture
#
# Penguji hampir pasti menanyakan mengapa K-Means yang dipilih. Menjalankan dua algoritma
# alternatif pada data yang sama dan melaporkan tingkat kesepakatannya (ARI) jauh lebih meyakinkan
# daripada argumen normatif. ARI tinggi berarti struktur yang ditemukan bukan artefak K-Means.


# %%
# ==============================================================================
# SEL 42 dari 71  |  KODE
# ==============================================================================
cmp_rows = []
for role in ROLES:
    X = X_red[role]; k = optimal_k[role]
    lab_km   = fit_kmeans(X, k).labels_
    lab_ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    lab_gmm  = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    cmp_rows.append({
        "role": role, "K": k,
        "sil_kmeans": silhouette_score(X, lab_km),
        "sil_ward":   silhouette_score(X, lab_ward),
        "sil_gmm":    silhouette_score(X, lab_gmm),
        "ARI_km_vs_ward": adjusted_rand_score(lab_km, lab_ward),
        "ARI_km_vs_gmm":  adjusted_rand_score(lab_km, lab_gmm),
    })
cmp_df = pd.DataFrame(cmp_rows).set_index("role")
print(cmp_df.round(3).to_string())
print()
print(f"Rata-rata kesepakatan K-Means vs Ward : {cmp_df['ARI_km_vs_ward'].mean():.3f}")
print(f"Rata-rata kesepakatan K-Means vs GMM  : {cmp_df['ARI_km_vs_gmm'].mean():.3f}")
print("\nPanduan tafsir ARI: >0.75 sangat sepakat | 0.50-0.75 sepakat sedang | <0.50 lemah")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 43 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 12. Visualisasi
#
# ### 12.1 Radar chart profil centroid


# %%
# ==============================================================================
# SEL 44 dari 71  |  KODE
# ==============================================================================
LABELS = ["DPM", "Damage\nShare", "Earned Gold\nShare", "CSPM", "Jungle CS\nShare",
          "Deaths/min", "Death\nShare", "Damage\nTaken/min", "Kill\nPartisip.",
          "Kill\nRatio", "Wards/min", "Wards\nCleared/min"]
ang = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist(); ang += ang[:1]

fig, ax = plt.subplots(2, 3, figsize=(10.5, 9.4), subplot_kw=dict(polar=True))
for i, role in enumerate(ROLES):
    a = ax[i // 3][i % 3]
    cen = P[P["position"] == role].groupby("cluster")[Z_COLS].mean()
    for cid, row in cen.iterrows():
        v = row.tolist() + [row.tolist()[0]]
        n = int((P["role_cluster"] == f"{role}_{cid}").sum())
        nama = CLUSTER_NAMES.get(f"{role}_{cid}", f"cluster {cid}")
        a.plot(ang, v, lw=1.4, label=f"{nama} (n={n})")
        a.fill(ang, v, alpha=0.06)
    a.set_xticks(ang[:-1]); a.set_xticklabels(LABELS, fontsize=5.2)
    a.tick_params(axis="y", labelsize=5.5)
    a.set_title(f"{role} (k={optimal_k[role]})", fontsize=10, fontweight="bold", pad=14)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=5.4, frameon=False)
ax[1][2].axis("off")
plt.subplots_adjust(hspace=0.62, wspace=0.30, top=0.94, bottom=0.05)
plt.savefig("fig_04_radar.png", dpi=200); plt.show()


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 45 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ### 12.2 Scatter plot PCA (sumbu diberi label sesuai loading)


# %%
# ==============================================================================
# SEL 46 dari 71  |  KODE
# ==============================================================================
fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.4))
for i, role in enumerate(ROLES):
    a = ax[i // 3][i % 3]
    s = P[P["position"] == role]
    ld = loadings_all[role]
    for cl in sorted(s["cluster"].unique()):
        d_ = s[s["cluster"] == cl]
        nama = CLUSTER_NAMES.get(f"{role}_{cl}", f"cluster {cl}")
        a.scatter(d_["pc1"], d_["pc2"], s=26, alpha=.8, label=nama,
                  edgecolor="white", lw=.4)
    lck = s[s["league"] == FOCUS_LEAGUE]
    a.scatter(lck["pc1"], lck["pc2"], facecolors="none", edgecolors="black", s=110, lw=1.1)
    for _, x in lck.iterrows():
        a.annotate(x["playername"], (x["pc1"], x["pc2"]), fontsize=4.6,
                   xytext=(3, 3), textcoords="offset points")
    a.axhline(0, color="grey", lw=.4); a.axvline(0, color="grey", lw=.4)
    a.set_title(f"{role} (k={optimal_k[role]})", fontsize=10, fontweight="bold")
    a.set_xlabel(f"PC1 ({pca_models[role].explained_variance_ratio_[0]:.0%}) : "
                 f"{ld['PC1'].idxmin()} <-> {ld['PC1'].idxmax()}", fontsize=5.6)
    a.set_ylabel(f"PC2 ({pca_models[role].explained_variance_ratio_[1]:.0%}) : "
                 f"{ld['PC2'].idxmin()} <-> {ld['PC2'].idxmax()}", fontsize=5.6)
    a.tick_params(labelsize=6)
    a.legend(fontsize=5, loc="best", frameon=True, framealpha=.85)
ax[1][2].axis("off")
ax[1][2].scatter([], [], facecolors="none", edgecolors="black", s=110, lw=1.1,
                 label=f"Pemain {FOCUS_LEAGUE}")
ax[1][2].legend(loc="center", fontsize=9, frameon=False)
plt.subplots_adjust(hspace=0.34, wspace=0.26, top=0.95, bottom=0.06)
plt.savefig("fig_05_pca_scatter.png", dpi=200); plt.show()


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 47 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 13. Validasi Domain: Champion Pool per Cluster
#
# RM#3 menjanjikan validasi "berdasarkan kewajaran domain", namun versi lama tidak memiliki
# mekanisme konkret untuk itu — penamaan cluster murni subjektif dari bentuk radar.
#
# Di sini digunakan **lift champion**:
#
# $$\text{lift}(c \mid \text{cluster}) = \frac{P(\text{champion } c \mid \text{cluster})}{P(\text{champion } c \mid \text{role})}$$
#
# Lift > 1 berarti champion tersebut lebih sering dimainkan cluster ini dibanding rata-rata role.
# Frekuensi mentah tidak dipakai karena meta 2025 sangat homogen — enam champion teratas praktis
# sama di semua cluster, sehingga tidak diskriminatif.
#
# Jika cluster yang secara statistik bercirikan "damage tinggi, deaths tinggi" memang didominasi
# champion berkelas assassin/duelist, maka penamaan cluster punya dasar empiris, bukan tafsir bebas.


# %%
# ==============================================================================
# SEL 48 dari 71  |  KODE
# ==============================================================================
MIN_PICKS = 12
champ_val = {}

for role in ROLES:
    m  = P["position"] == role
    mm = d.merge(P.loc[m, ["playerid", "cluster"]], on="playerid", how="inner")
    base = mm["champion"].value_counts(normalize=True)
    print(f"\n{'='*78}\n{role.upper()}")
    for cid in sorted(P.loc[m, 'cluster'].unique()):
        sub  = mm[mm["cluster"] == cid]
        freq = sub["champion"].value_counts(normalize=True)
        cnt  = sub["champion"].value_counts()
        lift = (freq / base).dropna()
        lift = lift[cnt.reindex(lift.index).fillna(0) >= MIN_PICKS].sort_values(ascending=False)
        champ_val[f"{role}_{cid}"] = lift
        print(f"  cluster {cid} (n={int((P['role_cluster']==f'{role}_{cid}').sum())} pemain, "
              f"{len(sub)} pertandingan)")
        print(f"     OVER-PICK : " + ", ".join(f"{i} ({v:.2f}x)" for i, v in lift.head(6).items()))
        print(f"     UNDER-PICK: " + ", ".join(f"{i} ({v:.2f}x)" for i, v in lift.tail(4).items()))


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 49 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 14. Penamaan Cluster
#
# **Jalankan sel di atas terlebih dahulu**, lalu isi kamus di bawah berdasarkan kombinasi
# (a) bentuk centroid Z-score dan (b) champion pool yang over-pick. Kunci kamus mengikuti format
# `{role}_{cluster_id}` dan cluster_id sudah deterministik (diurutkan menurut PC1).
#
# Sel berikutnya mencetak ringkasan otomatis yang bisa dipakai sebagai bahan penamaan.


# %%
# ==============================================================================
# SEL 50 dari 71  |  KODE
# ==============================================================================
def deskripsi(zrow, thr=0.5):
    out = []
    for f, v in zrow.items():
        nm = f[:-2]
        if v >= thr:  out.append(f"{nm} TINGGI ({v:+.2f})")
        elif v <= -thr: out.append(f"{nm} RENDAH ({v:+.2f})")
    return out or ["(mendekati rata-rata di semua fitur)"]

for role in ROLES:
    cen = P[P["position"] == role].groupby("cluster")[Z_COLS].mean()
    print(f"\n--- {role.upper()} ---")
    for cid, row in cen.iterrows():
        key = f"{role}_{cid}"
        n   = int((P["role_cluster"] == key).sum())
        top = ", ".join(champ_val[key].head(4).index) if key in champ_val else "-"
        print(f"  {key} (n={n})")
        print(f"     statistik: " + "; ".join(deskripsi(row)))
        print(f"     champion : {top}")


# %%
# ==============================================================================
# SEL 51 dari 71  |  KODE
# ==============================================================================
# CLUSTER_NAMES didefinisikan pada sel konfigurasi (bagian 0) agar tersedia
# lebih awal bagi sel radar chart dan scatter plot. Sel ini menerapkannya
# sekaligus memeriksa kelengkapannya terhadap klaster yang benar-benar terbentuk.
P["playstyle"] = P["role_cluster"].map(CLUSTER_NAMES)

belum   = sorted(set(P["role_cluster"]) - set(CLUSTER_NAMES))
berlebih = sorted(set(CLUSTER_NAMES) - set(P["role_cluster"]))

if belum:
    print("PERINGATAN -- klaster berikut belum dinamai:", belum)
    print("Aturan keputusan menghasilkan k yang berbeda dari saat kamus disusun.")
    print("Sesuaikan CLUSTER_NAMES pada sel konfigurasi.")
    P["playstyle"] = P["playstyle"].fillna(P["role_cluster"])
if berlebih:
    print("CATATAN -- nama berikut tidak terpakai:", berlebih)
if not belum and not berlebih:
    print(f"Seluruh {len(CLUSTER_NAMES)} profil terpetakan dengan tepat.")

print()
print(P.groupby(["position", "cluster", "playstyle"]).size().rename("n_pemain").to_string())


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 52 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 15. Analisis Fokus LCK vs Region Lain — Jawaban RM#4
#
# Bagian ini **hanya mungkin dilakukan** karena standardisasi tidak lagi per liga (§6).
#
# Dua uji dilakukan:
# 1. **Chi-square** pada tabel kontingensi liga × cluster
# 2. **Uji permutasi** — label liga diacak 5.000 kali untuk membangun distribusi nol empiris.
#    Lebih tepat daripada chi-square asimtotik karena banyak sel memiliki frekuensi harapan < 5.


# %%
# ==============================================================================
# SEL 53 dari 71  |  KODE
# ==============================================================================
from scipy.stats import chi2_contingency

def cramers_v(ct):
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

def perm_test(sub, n_perm=5000, seed=RANDOM_STATE):
    obs = chi2_contingency(pd.crosstab(sub["league"], sub["cluster"]))[0]
    rng = np.random.default_rng(seed)
    lg  = sub["league"].values.copy(); cl = sub["cluster"].values
    null = []
    for _ in range(n_perm):
        rng.shuffle(lg)
        null.append(chi2_contingency(pd.crosstab(pd.Series(lg), pd.Series(cl)))[0])
    return obs, (np.sum(np.array(null) >= obs) + 1) / (n_perm + 1)

for role in ROLES:
    sub = P[P["position"] == role]
    ct  = pd.crosstab(sub["league"], sub["cluster"])
    chi2, p_asym, dof, _ = chi2_contingency(ct)
    obs, p_perm = perm_test(sub)
    print(f"\n=== {role.upper()} ===")
    print(f"chi2={chi2:.2f} dof={dof} | p(asimtotik)={p_asym:.4f} | "
          f"p(permutasi)={p_perm:.4f} | Cramer's V={cramers_v(ct):.3f}")
    print(pd.crosstab(sub["league"], sub["cluster"], normalize="index").round(2).to_string())


# %%
# ==============================================================================
# SEL 54 dari 71  |  KODE
# ==============================================================================
lck = P[P["league"] == FOCUS_LEAGUE].sort_values(["position", "cluster", "playername"])
print(f"Pemain {FOCUS_LEAGUE} dan profil gaya bermainnya ({len(lck)} pemain):\n")
print(lck[["playername", "teamname", "position", "cluster", "playstyle", "n_games"]]
      .to_string(index=False))


# %%
# ==============================================================================
# SEL 55 dari 71  |  KODE
# ==============================================================================
# Jarak tiap pemain LCK ke centroid cluster-nya: seberapa "tipikal" pemain itu bagi profilnya
rows = []
for role in ROLES:
    m   = P["position"] == role
    X   = X_red[role]
    sub = P.loc[m].reset_index(drop=True)
    cen = np.array([X[sub["cluster"].values == c].mean(0) for c in sorted(sub["cluster"].unique())])
    dist = np.linalg.norm(X - cen[sub["cluster"].values], axis=1)
    sub  = sub.assign(dist_centroid=dist)
    rows.append(sub)
P_dist = pd.concat(rows, ignore_index=True)

f = P_dist[P_dist["league"] == FOCUS_LEAGUE].copy()
f["rank_dlm_cluster"] = f.groupby("role_cluster")["dist_centroid"].rank()
print(f"Pemain {FOCUS_LEAGUE} paling representatif untuk tiap profil (jarak terkecil ke centroid):\n")
print(f.sort_values(["position", "cluster", "dist_centroid"])
       [["playername", "position", "cluster", "playstyle", "dist_centroid"]]
       .round(2).to_string(index=False))


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 56 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ## 16. Ekspor Hasil


# %%
# ==============================================================================
# SEL 57 dari 71  |  KODE
# ==============================================================================
export_cols = (["playername", "playerid", "league", "position", "teamname", "n_games"]
               + FEATURES + Z_COLS
               + ["cluster", "role_cluster", "playstyle", "pc1", "pc2"])
P[export_cols].sort_values(["position", "cluster", "league"]) \
              .to_csv("out_hasil_clustering_pemain.csv", index=False)

ringkasan = (P.groupby(["position", "cluster", "role_cluster", "playstyle"])
               .agg(n_pemain=("playername", "count"),
                    **{f"{c}_mean": (c, "mean") for c in FEATURES},
                    **{f"{c}_z": (f"{c}_z", "mean") for c in FEATURES})
               .reset_index())
ringkasan.to_csv("out_ringkasan_cluster.csv", index=False)

pd.concat({r: diag[r] for r in ROLES}, names=["role"]).to_csv("out_diagnostik_k.csv")
cmp_df.to_csv("out_perbandingan_algoritma.csv")
pd.concat({r: loadings_all[r] for r in ROLES}, names=["role"]).to_csv("out_pca_loadings.csv")

# --- champion lift per cluster (bahan penamaan & validasi domain) ---
baris_champ = []
for role in ROLES:
    m  = P["position"] == role
    mm = d.merge(P.loc[m, ["playerid", "cluster"]], on="playerid", how="inner")
    base = mm["champion"].value_counts(normalize=True)
    for cid in sorted(P.loc[m, "cluster"].unique()):
        sub  = mm[mm["cluster"] == cid]
        freq = sub["champion"].value_counts(normalize=True)
        cnt  = sub["champion"].value_counts()
        for champ in cnt.index:
            if cnt[champ] >= MIN_PICKS:
                baris_champ.append({
                    "role": role, "cluster": int(cid), "role_cluster": f"{role}_{cid}",
                    "champion": champ, "n_pick": int(cnt[champ]),
                    "share_cluster": round(float(freq[champ]), 4),
                    "share_role": round(float(base[champ]), 4),
                    "lift": round(float(freq[champ] / base[champ]), 3),
                })
champ_df = (pd.DataFrame(baris_champ)
            .sort_values(["role", "cluster", "lift"], ascending=[True, True, False]))
champ_df.to_csv("out_champion_lift.csv", index=False)

# --- tabel reliabilitas variabel ---
rel.to_csv("out_reliabilitas_variabel.csv")

print("Tersimpan:")
for f_ in ["out_hasil_clustering_pemain.csv", "out_ringkasan_cluster.csv",
           "out_diagnostik_k.csv", "out_perbandingan_algoritma.csv", "out_pca_loadings.csv",
           "out_champion_lift.csv", "out_reliabilitas_variabel.csv"]:
    print("  -", f_)
print("Gambar: fig_01, fig_02, fig_03a-fig_03f, fig_04, fig_05 (.png)")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 58 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ---
# # Lampiran A — Analisis Sensitivitas: Normalisasi per Liga+Role
#
# Menjalankan ulang seluruh pipeline dengan standardisasi versi lama, lalu membandingkan hasilnya
# dengan ARI. Tujuannya menunjukkan secara kuantitatif seberapa besar pilihan normalisasi mengubah
# kesimpulan — ini pembelaan yang kuat di sidang.


# %%
# ==============================================================================
# SEL 59 dari 71  |  KODE
# ==============================================================================
PA = P.copy()
ZA = [f"{c}_za" for c in FEATURES]
for c in FEATURES:
    PA[f"{c}_za"] = PA.groupby(["league", "position"])[c].transform(lambda x: (x - x.mean()) / x.std())

print(f"{'role':>4} | {'ARI global vs per-liga':>22} | {'Cramers V (global)':>18} | {'Cramers V (per-liga)':>20}")
print("-" * 78)
for role in ROLES:
    m  = PA["position"] == role
    Xa = PCA(n_components=X_red[role].shape[1], random_state=RANDOM_STATE) \
            .fit_transform(PA.loc[m, ZA].values)
    lab_a = fit_kmeans(Xa, optimal_k[role]).labels_
    lab_g = PA.loc[m, "cluster"].values
    sub_a = PA.loc[m].assign(cl=lab_a)
    v_g = cramers_v(pd.crosstab(PA.loc[m, "league"], lab_g))
    v_a = cramers_v(pd.crosstab(sub_a["league"], sub_a["cl"]))
    print(f"{role:>4} | {adjusted_rand_score(lab_g, lab_a):>22.3f} | {v_g:>18.3f} | {v_a:>20.3f}")
print()
print("Cramer's V mengukur keterkaitan liga dengan keanggotaan cluster.")
print("Nilai yang jauh lebih rendah pada versi per-liga membuktikan bahwa normalisasi")
print("per-liga menghapus sinyal antar-region yang justru menjadi objek RM#4.")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 60 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ---
# # Lampiran B — Analisis Sensitivitas: Fitur Tempo (LPL Dikecualikan)
#
# Fitur `golddiffat15`, `xpdiffat15`, `csdiffat15` menangkap dimensi gaya yang tidak tercakup 12
# fitur utama, yaitu **dominasi fase awal permainan**. Karena kolom ini hilang total pada LPL,
# analisis dijalankan pada lima liga saja dan hanya berstatus pendukung, bukan model utama.
#
# Yang diuji: apakah struktur cluster tetap serupa ketika dimensi tempo ditambahkan.


# %%
# ==============================================================================
# SEL 61 dari 71  |  KODE
# ==============================================================================
TEMPO = ["golddiffat15", "xpdiffat15", "csdiffat15"]
d5 = d[d["league"] != "LPL"].dropna(subset=TEMPO)
g5 = d5.groupby(KEY)
B = pd.DataFrame({t: g5[t].mean() for t in TEMPO}).reset_index()

PB = P[P["league"] != "LPL"].merge(B, on=KEY, how="inner")
FB = FEATURES + TEMPO
ZB = [f"{c}_zb" for c in FB]
for c in FB:
    PB[f"{c}_zb"] = PB.groupby("position")[c].transform(lambda x: (x - x.mean()) / x.std())

print(f"n pemain (tanpa LPL): {len(PB)}\n")
print(f"{'role':>4} | {'n':>4} | {'komponen':>9} | {'K':>2} | {'silhouette':>10} | {'ARI vs model utama':>19}")
print("-" * 72)
for role in ROLES:
    m = PB["position"] == role
    X = PB.loc[m, ZB].values
    pf  = PCA(random_state=RANDOM_STATE).fit(X)
    nc  = int(np.searchsorted(np.cumsum(pf.explained_variance_ratio_), PCA_VAR_TARGET) + 1)
    Xb  = PCA(n_components=nc, random_state=RANDOM_STATE).fit_transform(X)
    k   = optimal_k[role]
    lab = fit_kmeans(Xb, k).labels_
    print(f"{role:>4} | {m.sum():>4} | {nc:>9} | {k:>2} | {silhouette_score(Xb, lab):>10.3f} | "
          f"{adjusted_rand_score(PB.loc[m, 'cluster'].values, lab):>19.3f}")
print()
print("Cara membaca ARI di atas (bandingkan model utama vs model + fitur tempo):")
print("  ARI >= 0.60 : struktur cluster relatif bertahan -> pengecualian fitur tempo aman")
print("  ARI 0.30-0.60: sebagian struktur berubah -> laporkan sebagai keterbatasan di Bab V")
print("  ARI <  0.30 : dimensi tempo membawa informasi gaya yang berbeda; pertimbangkan")
print("                menjadikannya analisis pendamping khusus untuk role tersebut")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 62 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ---
# # Lampiran C — Analisis Sensitivitas: Cara Agregasi ke Tingkat Pemain
#
# Model utama memakai **dua jalur**: tujuh variabel bawaan dataset dirata-ratakan antar-pertandingan,
# sementara lima variabel turunan dihitung sebagai rasio dari jumlah.
#
# Lampiran ini membandingkannya dengan alternatif **satu jalur**, yaitu seluruh dua belas variabel
# dihitung per pertandingan lalu dirata-ratakan. Contoh perbedaannya: seorang pemain bermain dua kali,
# 20 menit dengan 10.000 damage dan 40 menit dengan 24.000 damage.
#
# - Rata-rata antar-pertandingan: (500 + 600) / 2 = **550 DPM**
# - Rasio dari jumlah: 34.000 / 60 = **567 DPM**
#
# Yang dilaporkan: selisih nilai variabel, perubahan keanggotaan klaster, dan kestabilan kedua
# pendekatan.


# %%
# ==============================================================================
# SEL 63 dari 71  |  KODE
# ==============================================================================
# --- Versi alternatif: SELURUH variabel dirata-ratakan antar-pertandingan ---
dA = d_final.copy()
for nama, (pemb, peny) in VARIABEL_TURUNAN.items():
    dA[nama] = dA[pemb] / dA[peny].replace(0, np.nan)

gA = dA.groupby(KEY)
ALT = pd.DataFrame(index=gA.size().index)
for nama, kolom in KOLOM_SIAP_PAKAI.items():
    ALT[nama] = gA[kolom].mean()
for nama in VARIABEL_TURUNAN:
    ALT[nama] = gA[nama].mean()
ALT = ALT.reset_index()

print(f"Model utama (dua jalur) : {len(P)} pemain")
print(f"Alternatif (satu jalur) : {len(ALT)} pemain")


# %%
# ==============================================================================
# SEL 64 dari 71  |  KODE
# ==============================================================================
# --- Seberapa besar selisih nilai variabelnya? ---
bd = P[KEY + FEATURES].merge(ALT, on=KEY, suffixes=("_utama", "_alt"))
baris = []
for f in FEATURES:
    u, a = bd[f + "_utama"], bd[f + "_alt"]
    rel_ = ((u - a).abs() / u.abs().replace(0, np.nan) * 100)
    baris.append({"variabel": f, "utama": u.mean(), "alternatif": a.mean(),
                  "selisih rata2 (%)": rel_.mean(), "selisih maks (%)": rel_.max()})
print(pd.DataFrame(baris).round(3).to_string(index=False))
print()
print("Tujuh variabel pertama identik nol karena diperlakukan sama pada kedua versi;")
print("perbedaan hanya muncul pada lima variabel turunan.")


# %%
# ==============================================================================
# SEL 65 dari 71  |  KODE
# ==============================================================================
# --- Apakah keanggotaan klaster berubah, dan mana yang lebih stabil? ---
print(f"{'role':>5} | {'ARI utama vs alt':>17} | {'beda klaster':>14} | "
      f"{'stabil utama':>12} | {'stabil alt':>11}")
print("-" * 74)
for role in ROLES:
    k  = optimal_k[role]
    sU = P[P["position"] == role]
    sA = ALT.set_index(KEY).loc[sU.set_index(KEY).index].reset_index()

    def ruang(frame):
        Z = ((frame[FEATURES] - frame[FEATURES].mean()) / frame[FEATURES].std()).values
        pf = PCA(random_state=RANDOM_STATE).fit(Z)
        nc = int(np.searchsorted(np.cumsum(pf.explained_variance_ratio_), PCA_VAR_TARGET) + 1)
        return PCA(n_components=nc, random_state=RANDOM_STATE).fit_transform(Z)

    XU, XA = ruang(sU), ruang(sA)
    labU, labA = fit_kmeans(XU, k).labels_, fit_kmeans(XA, k).labels_
    beda = len(labU) - pd.crosstab(labU, labA).values.max(1).sum()
    stU, _ = bootstrap_ari(XU, k, n_boot=60)
    stA, _ = bootstrap_ari(XA, k, n_boot=60)
    print(f"{role:>5} | {adjusted_rand_score(labU, labA):>17.3f} | "
          f"{beda:>4} dari {len(labU):<5} | {stU:>12.3f} | {stA:>11.3f}")
print()
print("Kolom stabilitas memakai kriteria yang sama dengan pemilihan k (bootstrap ARI).")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 66 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ---
# # Lampiran D — Analisis Sensitivitas Parameter Konfigurasi
#
# Enam parameter pada sel konfigurasi menentukan hasil, namun statusnya berbeda-beda.
#
# | Parameter | Status | Dasar |
# |---|---|---|
# | `N_BOOTSTRAP = 100` | anjuran literatur | Hennig (2007) menganjurkan B = 100 |
# | `ARI_THRESHOLD = 0.60` | adaptasi literatur | Hennig (2007): di bawah 0,60 klaster tidak layak dipercaya |
# | `PCA_VAR_TARGET = 0.80` | konvensi | rentang 70–90% lazim dipakai |
# | `K_RANGE = range(2, 9)` | dibatasi data | pada k ≥ 7 klaster terkecil tinggal 2–4 pemain |
# | `BOOT_FRAC = 0.80` | konvensi | fraksi subsampel pada penilaian kestabilan |
# | `K_MIN_MEANING = 3` | **keputusan substantif** | k = 2 hanya memisahkan di atas/di bawah rata-rata |
#
# > **Catatan penting soal ambang.** Hennig (2007) menyusun panduan tafsirnya untuk **Jaccard per
# > klaster**, sedangkan penelitian ini memakai **Adjusted Rand Index atas keseluruhan partisi**.
# > Keduanya indeks yang berbeda, sehingga nilai 0,60 dipinjam secara analogi dan bukan diterapkan
# > langsung. Hal ini perlu dinyatakan terbuka di naskah.
#
# > **Ambang dan fraksi subsampel adalah satu paket.** Nilai kestabilan tidak bermakna absolut —
# > besarnya bergantung pada `BOOT_FRAC`. Keduanya harus selalu dilaporkan berdampingan.
#
# Lampiran ini menghasilkan tabel sensitivitas bagi keempat parameter yang paling berpengaruh.


# %%
# ==============================================================================
# SEL 67 dari 71  |  KODE
# ==============================================================================
def ruang_pca(frame, var):
    """Ruang komponen utama pada ambang variansi tertentu; kembalikan koordinat & jumlah komponen."""
    Z = ((frame[FEATURES] - frame[FEATURES].mean()) / frame[FEATURES].std()).values
    evr = PCA(random_state=RANDOM_STATE).fit(Z).explained_variance_ratio_
    m = int(np.searchsorted(np.cumsum(evr), var) + 1)
    return PCA(n_components=m, random_state=RANDOM_STATE).fit_transform(Z), m

# ---------- D.1  PCA_VAR_TARGET ----------
VAR_UJI = [0.70, 0.80, 0.90]
acuan = {r: fit_kmeans(ruang_pca(P[P["position"] == r], PCA_VAR_TARGET)[0],
                       optimal_k[r]).labels_ for r in ROLES}

print("D.1  Pengaruh ambang variansi PCA terhadap hasil klaster")
print("     (ARI dibandingkan terhadap konfigurasi yang dipakai)\n")
print(f"{'role':>5} | " + " | ".join(f"{'var ' + str(v):>20}" for v in VAR_UJI))
print("-" * 74)
for r in ROLES:
    sel = []
    for v in VAR_UJI:
        X, m = ruang_pca(P[P["position"] == r], v)
        lab = fit_kmeans(X, optimal_k[r]).labels_
        tanda = " *" if abs(v - PCA_VAR_TARGET) < 1e-9 else "  "
        sel.append(f"{m} komp, ARI {adjusted_rand_score(acuan[r], lab):.2f}{tanda}")
    print(f"{r:>5} | " + " | ".join(f"{s:>20}" for s in sel))
print("\n* = konfigurasi yang dipakai. ARI rendah berarti hasil sensitif terhadap parameter ini.")


# %%
# ==============================================================================
# SEL 68 dari 71  |  KODE
# ==============================================================================
# ---------- D.2  BOOT_FRAC dan N_BOOTSTRAP ----------
FRAC_UJI = [0.5, 0.7, 0.8, 0.9]
BOOT_UJI = [25, 50, 100, 200]

print("D.2a  Pengaruh fraksi subsampel terhadap nilai kestabilan\n")
print(f"{'role':>5} | " + " | ".join(f"{'frac ' + str(f):>12}" for f in FRAC_UJI))
print("-" * 66)
for r in ROLES:
    X, _ = ruang_pca(P[P["position"] == r], PCA_VAR_TARGET)
    nilai = [bootstrap_ari(X, optimal_k[r], n_boot=40, frac=f)[0] for f in FRAC_UJI]
    print(f"{r:>5} | " + " | ".join(f"{v:>12.3f}" for v in nilai))
print("\nSelisih antar-fraksi menunjukkan bahwa nilai kestabilan tidak bermakna absolut;")
print("angka tersebut hanya dapat ditafsirkan bersama fraksi subsampel yang dipakai.")

print("\n\nD.2b  Apakah jumlah replikasi bootstrap sudah memadai?")
print("      (nilai rata-rata dan galat bakunya)\n")
print(f"{'role':>5} | " + " | ".join(f"{'B=' + str(b):>15}" for b in BOOT_UJI))
print("-" * 74)
for r in ROLES:
    X, _ = ruang_pca(P[P["position"] == r], PCA_VAR_TARGET)
    sel = []
    for b in BOOT_UJI:
        m, s = bootstrap_ari(X, optimal_k[r], n_boot=b, frac=BOOT_FRAC)
        sel.append(f"{m:.3f} +/-{s / np.sqrt(b):.3f}")
    print(f"{r:>5} | " + " | ".join(f"{x:>15}" for x in sel))
print("\nGalat baku yang mengecil dan mendatar menandakan jumlah replikasi memadai.")


# %%
# ==============================================================================
# SEL 69 dari 71  |  KODE
# ==============================================================================
# ---------- D.3  ARI_THRESHOLD dan K_MIN_MEANING ----------
# Hitung sekali nilai ARI & ukuran klaster terkecil untuk seluruh k, lalu
# terapkan berbagai kombinasi aturan keputusan pada tabel yang sama.
tabel = {}
for r in ROLES:
    X, _ = ruang_pca(P[P["position"] == r], PCA_VAR_TARGET)
    n = len(X)
    tabel[r] = {k: (bootstrap_ari(X, k, n_boot=60, frac=BOOT_FRAC)[0],
                    int(np.bincount(fit_kmeans(X, k).labels_).min()), n)
                for k in K_RANGE}

print("D.3a  Nilai kestabilan dan ukuran klaster terkecil untuk setiap k\n")
for r in ROLES:
    n = tabel[r][list(K_RANGE)[0]][2]
    print(f"  {r} (n={n}): " + "  ".join(
        f"k={k}:{tabel[r][k][0]:.2f}(min {tabel[r][k][1]})" for k in K_RANGE))

def terapkan(role, ambang, kmin):
    lolos = [k for k in K_RANGE
             if k >= kmin and tabel[role][k][0] >= ambang
             and tabel[role][k][1] >= max(5, 0.05 * tabel[role][k][2])]
    if lolos:
        return str(max(lolos))
    cadangan = max([k for k in K_RANGE if k >= kmin], key=lambda k: tabel[role][k][0])
    return f"({cadangan})*"

print("\n\nD.3b  K yang terpilih pada berbagai kombinasi aturan keputusan\n")
print(f"{'aturan':>22}" + "".join(f"{r:>8}" for r in ROLES))
print("-" * 64)
for kmin in [2, K_MIN_MEANING]:
    for ambang in [0.50, ARI_THRESHOLD, 0.70]:
        dipakai = " <-" if (kmin == K_MIN_MEANING and abs(ambang - ARI_THRESHOLD) < 1e-9) else ""
        label = f"kmin={kmin} ambang={ambang:.2f}"
        print(f"{label:>22}" + "".join(f"{terapkan(r, ambang, kmin):>8}" for r in ROLES) + dipakai)
print("\n(*) tidak ada k yang lolos ambang, dipakai aturan cadangan ARI tertinggi")
print("(<-) konfigurasi yang dipakai pada model utama")
print()
print("Perhatikan baris kmin=2: bila k=2 diizinkan, sebagian role akan terpilih k=2 karena")
print("partisi dua kelompok memang paling stabil. Inilah alasan K_MIN_MEANING ditetapkan,")
print("dan alasannya bersifat substantif, bukan statistik.")


# %%
# ==============================================================================
# SEL 70 dari 71  |  KODE
# ==============================================================================
# ---------- D.4  Justifikasi jumlah variabel ----------
# Menguji klaim Bab II: yang menentukan mutu representasi adalah keseimbangan
# komposisi variabel antar-dimensi, bukan banyaknya variabel.
# Dibentuk himpunan pembanding berisi SELURUH kolom numerik tingkat pemain yang
# tersedia lengkap pada keenam liga, lalu diperlakukan dengan prosedur yang sama.

BUANG = {"year", "playoffs", "game", "patch", "participantid", "gameid", "result",
         "gamelength", "mins", "n", "teamkills", "teamdeaths", "doublekills",
         "triplekills", "quadrakills", "pentakills", "firstblood", "firstbloodkill",
         "firstbloodassist", "firstbloodvictim", "datacompleteness"}

num = [c for c in d.columns
       if pd.api.types.is_numeric_dtype(d[c]) and c not in BUANG]
lengkap = [c for c in num
           if d.groupby("league")[c].apply(lambda s: s.notna().mean()).min() > 0.99
           and d[c].std() > 0]
Q = d.groupby(KEY)[lengkap].mean()
Q = Q.loc[:, Q.std() > 0].dropna(axis=1).reset_index()
FITUR_LUAS = [c for c in Q.columns if c not in KEY]

print(f"D.4  Himpunan 12 variabel terpilih vs himpunan luas ({len(FITUR_LUAS)} variabel)\n")
print(f"{'peran':>6} {'n':>5} | {'PC1 (12 var)':>13} {'PC1 (luas)':>11} | selisih")
print("-" * 58)
for r in ROLES:
    A = P[P["position"] == r][FEATURES].dropna()
    a1 = PCA().fit(((A - A.mean()) / A.std()).values).explained_variance_ratio_[0]
    B = Q[Q["position"] == r][FITUR_LUAS].dropna()
    b1 = PCA().fit(((B - B.mean()) / B.std()).values).explained_variance_ratio_[0]
    print(f"{r:>6} {len(A):>5} | {a1 * 100:12.1f}% {b1 * 100:10.1f}% | {(b1 - a1) * 100:+6.1f} poin")

print("\nMuatan terbesar pada komponen pertama himpunan luas (peran jungle):")
B = Q[Q["position"] == "jng"][FITUR_LUAS].dropna()
m = PCA().fit(((B - B.mean()) / B.std()).values)
l = m.components_[0]
for j in np.argsort(-np.abs(l))[:6]:
    print(f"   {FITUR_LUAS[j]:<26} {l[j]:+.3f}")

print("\nKomponen pertama himpunan luas didominasi beberapa ukuran gold yang secara")
print("konstruk identik. Menambah variabel yang mengukur besaran yang sama tidak")
print("menambah informasi, melainkan menambah bobot pada dimensi yang sudah terwakili.")


# %% [markdown]
# <!-- ============================================================================== -->
# <!-- SEL 71 dari 71  |  MARKDOWN -->
# <!-- ============================================================================== -->
# ---
# # Ringkasan Perubahan untuk Direfleksikan ke Naskah
#
# **Bab I**
# - Batasan Masalah butir 4: PCA kini dipakai untuk reduksi dimensi *clustering* dan visualisasi, bukan visualisasi saja
# - Tambahkan batasan: fitur tempo (`at15`) tidak dipakai karena data LPL berstatus `partial`
# - Tambahkan batasan: unit analisis adalah rata-rata musim; variasi antar-pertandingan tidak dimodelkan
# - Tambahkan batasan: pergeseran patch/meta sepanjang musim 2025 tidak dikontrol
#
# **Bab II**
# - Tambahkan landasan teori: Davies–Bouldin, Calinski–Harabasz, Gap Statistic, Adjusted Rand Index
# - Tambahkan sub-bab perbandingan algoritma clustering (K-Means vs Ward vs GMM)
# - Perluas sub-bab metrik kinerja dari 5 menjadi 12 variabel, kelompokkan menurut dimensi gaya
#
# **Bab III**
# - Ubah sub-bab normalisasi: Z-Score **global per role**, sertakan alasan dan analisis sensitivitas
# - Tambahkan sub-bab uji multikolinearitas sebagai tahap praproses
# - Dokumentasikan metode deteksi elbow otomatis (Kneedle, Satopaa et al., 2011)
# - Nyatakan aturan keputusan pemilihan K secara eksplisit
# - Tambahkan sub-bab validasi stabilitas (bootstrap ARI) dan validasi domain (champion lift)
# - Tambahkan diagram alur penelitian
# - Perbaiki: inisialisasi centroid memakai **K-Means++**, bukan "acak"
#
# **Referensi baru yang diperlukan**
# - Satopaa, V. et al. (2011). *Finding a "Kneedle" in a Haystack.* ICDCS Workshops.
# - Davies, D. L., & Bouldin, D. W. (1979). IEEE TPAMI.
# - Calinski, T., & Harabasz, J. (1974). *Communications in Statistics.*
# - Hubert, L., & Arabie, P. (1985). *Comparing partitions.* Journal of Classification.
# - Hennig, C. (2007). *Cluster-wise assessment of cluster stability.* CSDA.
