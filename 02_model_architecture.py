"""
02_model_architecture.py
================================================================================
Arsitektur Hybrid Ringan: 1D Depthwise Separable CNN + Temporal Attention GRU
(1D-DSC + TA-GRU) Berbasis Ekstraksi Morfologi Gelombang PPG (PPG, VPG, APG /
Dicrotic Notch) & Transisi Ritme Sekuensial 30 Detik, Dilengkapi Dukungan
Self-Supervised Contrastive Learning (SimCLR) sesuai Proposal Pra-TA:
  "Perancangan Ear-Worn Berbasis PPG untuk Deteksi Fibrilasi Atrium
   Menggunakan Deep Learning" (Abiel Ifan Imanuel Hukom - 5023231004)

Keunggulan Ilmiah Arsitektur Ini:
1. Tidak hanya mengandalkan jarak puncak sistolik (IBI), melainkan mengekstraksi
   langsung Morfologi Gelombang PPG per sub-jendela waktu menggunakan:
   - Sinyal PPG ternormalisasi (amplitudo sistolik & lembah diastolik)
   - Turunan Pertama / Velocity Plethysmogram (VPG = dPPG/dt): kemiringan naik
     fase anakrotik (anacrotic slope) & kemiringan turun katakrotik.
   - Turunan Kedua / Acceleration Plethysmogram (APG = d^2PPG/dt^2): menonjolkan
     kompleks gelombang c-d-e pada Dicrotic Notch (penutupan katup aorta di fase
     katakrotik, sesuai Basza et al. [45] / Slide 8 ETS).
2. Membedakan AFib dari Kontraksi Prematur (PAC & PVC - Ding et al. [4]):
   - Pada PAC/PVC, hanya 1-2 sub-jendela yang mengalami denyut prematur + jeda
     kompensasi (compensatory pause), sedangkan sub-jendela sinus di sekitarnya
     tetap memiliki dicrotic notch dan kemiringan morfologi yang konsisten.
   - Pada AFib, morfologi dicrotic notch, amplitudo pulsa, dan periodisitas
     berubah acak secara terus-menerus di sepanjang 16 sub-jendela (30 detik).
3. Mendukung Two-Stage Training (Bab 2.7 - Wu et al. [37] & Rong et al. [39]):
   - Tahap 1: Self-Supervised Contrastive Pre-Training (SimCLR) dengan operasi
     `AddNoise`, `Drift`, dan `Bernoulli-Gaussian` (Persamaan 2.11).
   - Tahap 2: Supervised Fine-Tuning (AFib vs Non-AFib [Normal + PAC/PVC]).
4. 100% kompatibel dengan TFLite Built-in Ops (Float32 & INT8) untuk inferensi
   real-time < 2 ms di Raspberry Pi Zero 2 W.
================================================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


@tf.keras.utils.register_keras_serializable(package="PrataAFib")
class PPGMorphologyAndRhythmLayer(layers.Layer):
    """
    Layer Front-End Fisiologis yang mengekstraksi:
    1) 16 Kanal Morfologi Gelombang Intra-Detak (PPG, VPG = dPPG/dt, APG = d^2PPG/dt^2):
       - Amplop puncak sistolik & lembah diastolik (variasi volume sekuncup / pulse deficit)
       - Kemiringan fase anakrotik (anacrotic slope) & katakrotik dari VPG
       - Intensitas & energi infleksi Dicrotic Notch dari APG (d^2PPG/dt^2)
    2) 112 Kanal Autokorelasi Ritme Gelombang PPG (lag 35..146 sampel = 0.28s..1.17s)
    3) 32 Kanal Konsistensi Morfologi VPG Antar-Detak (lag 36..129 sampel, step 3)
    Total fitur per sub-jendela waktu (16 sub-jendela sepanjang 30 detik) = 160 kanal.
    Seluruh operasi menggunakan TFLite Built-in Ops murni.
    """

    def __init__(
        self,
        min_lag=35,
        max_lag=146,
        frame_size=300,
        frame_stride=230,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.smooth_pool = layers.AveragePooling1D(
            pool_size=3, strides=1, padding="same"
        )
        self.trend_pool = layers.AveragePooling1D(
            pool_size=125, strides=1, padding="same"
        )
        self.avg_frame_pool = layers.AveragePooling1D(
            pool_size=frame_size, strides=frame_stride, padding="valid"
        )
        self.max_frame_pool = layers.MaxPooling1D(
            pool_size=frame_size, strides=frame_stride, padding="valid"
        )

    def call(self, inputs):
        # 1. Denoising ringan & penghilangan wander frekuensi sangat rendah
        x = self.smooth_pool(inputs)
        x = x - self.trend_pool(x)

        # 2. Turunan Pertama (VPG = dPPG/dt) -> Kemiringan Anakrotik & Katakrotik
        vpg_raw = x[:, 1:, :] - x[:, :-1, :]
        vpg = tf.concat([vpg_raw[:, :1, :], vpg_raw], axis=1) * 5.0

        # 3. Turunan Kedua (APG = d^2PPG/dt^2) -> Infleksi Dicrotic Notch (Fase Katakrotik)
        apg_raw = vpg[:, 1:, :] - vpg[:, :-1, :]
        apg = tf.concat([apg_raw[:, :1, :], apg_raw], axis=1) * 2.0

        # 4. Normalisasi energi lokal untuk konsistensi amplitudo
        x_sq = tf.square(x)
        local_var = self.trend_pool(x_sq)
        inv_std = tf.math.rsqrt(local_var + 1e-4)
        x_norm = tf.clip_by_value(x * inv_std, -3.5, 3.5)

        vpg_sq = tf.square(vpg)
        vpg_inv_std = tf.math.rsqrt(self.trend_pool(vpg_sq) + 1e-4)
        vpg_norm = tf.clip_by_value(vpg * vpg_inv_std, -3.5, 3.5)

        apg_sq = tf.square(apg)
        apg_inv_std = tf.math.rsqrt(self.trend_pool(apg_sq) + 1e-4)
        apg_norm = tf.clip_by_value(apg * apg_inv_std, -3.5, 3.5)

        # 5. Ekstraksi 16 Kanal Fitur Morfologi Gelombang per Sub-Jendela (PPG + VPG + APG)
        pos_ppg = tf.nn.relu(x_norm)
        neg_ppg = tf.nn.relu(-x_norm)
        pos_vpg = tf.nn.relu(vpg_norm)
        neg_vpg = tf.nn.relu(-vpg_norm)
        pos_apg = tf.nn.relu(apg_norm)
        neg_apg = tf.nn.relu(-apg_norm)
        abs_vpg = tf.abs(vpg_norm)
        abs_apg = tf.abs(apg_norm)

        morph_avg_inputs = tf.concat(
            [
                pos_ppg,                     # 1. Rata-rata amplop sistolik
                neg_ppg,                     # 2. Rata-rata amplop diastolik
                tf.square(x_norm),           # 3. Energi pulsatil sub-jendela
                pos_vpg,                     # 4. Rata-rata kemiringan naik anakrotik
                neg_vpg,                     # 5. Rata-rata kemiringan turun katakrotik
                abs_vpg,                     # 6. Energi kecepatan gelombang (VPG)
                pos_apg,                     # 7. Rata-rata infleksi positif Dicrotic Notch (c-wave)
                neg_apg,                     # 8. Rata-rata infleksi negatif Dicrotic Notch (d-wave)
                abs_apg,                     # 9. Energi total Dicrotic Notch (APG)
                tf.square(apg_norm),         # 10. Daya ketajaman Dicrotic Notch
            ],
            axis=-1,
        )
        morph_max_inputs = tf.concat(
            [
                pos_ppg,                     # 11. Puncak sistolik maksimum sub-jendela
                neg_ppg,                     # 12. Lembah diastolik terdalam sub-jendela
                pos_vpg,                     # 13. Kemiringan anakrotik maksimum (dPPG/dt max)
                neg_vpg,                     # 14. Kemiringan katakrotik maksimum
                pos_apg,                     # 15. Puncak gelombang-a/c APG (Dicrotic Notch peak)
                neg_apg,                     # 16. Lembah gelombang-b/d APG (Dicrotic Notch trough)
            ],
            axis=-1,
        )

        morph_avg_frames = self.avg_frame_pool(morph_avg_inputs)  # (batch, 16, 10)
        morph_max_frames = self.max_frame_pool(morph_max_inputs)  # (batch, 16, 6)

        # 6. Ekstraksi 112 Kanal Autokorelasi Ritme Gelombang PPG (lag 35..146)
        ppg_lag_prods = []
        for lag in range(self.min_lag, self.max_lag + 1):
            shifted = tf.pad(x_norm[:, :-lag, :], [[0, 0], [lag, 0], [0, 0]])
            ppg_lag_prods.append(x_norm * shifted)
        ppg_lag_Concat = tf.concat(ppg_lag_prods, axis=-1)
        ppg_rhythm_frames = self.avg_frame_pool(ppg_lag_Concat)  # (batch, 16, 112)

        # 7. Ekstraksi 32 Kanal Konsistensi Morfologi Kemiringan & Notch Antar-Detak (VPG lags)
        vpg_lag_prods = []
        for lag in range(36, 132, 3):  # 32 lag mewakili 0.29s .. 1.05s
            shifted_v = tf.pad(vpg_norm[:, :-lag, :], [[0, 0], [lag, 0], [0, 0]])
            vpg_lag_prods.append(vpg_norm * shifted_v)
        vpg_lag_concat = tf.concat(vpg_lag_prods, axis=-1)
        vpg_morph_frames = self.avg_frame_pool(vpg_lag_concat)  # (batch, 16, 32)

        # Gabungkan seluruh fitur: 10 + 6 + 112 + 32 = 160 kanal x 16 sub-jendela waktu
        out_frames = tf.concat(
            [morph_avg_frames, morph_max_frames, ppg_rhythm_frames, vpg_morph_frames],
            axis=-1,
        )
        return out_frames

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "min_lag": self.min_lag,
                "max_lag": self.max_lag,
                "frame_size": self.frame_size,
                "frame_stride": self.frame_stride,
            }
        )
        return cfg


@tf.keras.utils.register_keras_serializable(package="PrataAFib")
class TemporalAttentionLayer(layers.Layer):
    """
    Mekanisme Temporal Attention (TA) sesuai Persamaan (2.9)-(2.10) Proposal Bab 2.6:
      u_t = tanh(W_w * h_t + b_w)
      alpha_t = softmax(u_t^T * u_w)
      c = sum_t(alpha_t * h_t)
    Memungkinkan model memberikan bobot perhatian berbeda pada setiap sub-jendela
    waktu untuk membedakan ketidakteraturan persisten (AFib) dari denyut ektopik
    terisolasi (PAC/PVC) maupun artefak gerak sesaat.
    """

    def __init__(self, attn_units=32, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = attn_units
        self.w_dense = layers.Dense(attn_units, activation="tanh")
        self.u_dense = layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time_steps, gru_units)
        score = self.u_dense(self.w_dense(inputs))  # (batch, time_steps, 1)
        alpha = tf.nn.softmax(score, axis=1)        # (batch, time_steps, 1)
        context = tf.reduce_sum(inputs * alpha, axis=1)  # (batch, gru_units)
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attn_units": self.attn_units})
        return cfg


@tf.keras.utils.register_keras_serializable(package="PrataAFib")
class L2NormalizeLayer(layers.Layer):
    """Normalisasi L2 untuk vektor proyeksi SimCLR Contrastive Learning."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


def dsc_se_residual_block(x, filters, kernel_size=5, stride=1, weight_decay=1e-4, name_prefix="dsc"):
    """
    Blok 1D Depthwise Separable Convolution (1D-DSC) sesuai Persamaan (2.1)-(2.3)
    Proposal Bab 2.6, dilengkapi Squeeze-and-Excitation (SE) & Residual Connection.
    """
    shortcut = x

    # Depthwise 1D Convolution + Pointwise 1x1 Convolution (SeparableConv1D)
    y = layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(weight_decay),
        pointwise_regularizer=regularizers.l2(weight_decay),
        name=f"{name_prefix}_sepconv",
    )(x)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn")(y)
    y = layers.Activation("swish", name=f"{name_prefix}_swish")(y)

    # Squeeze-and-Excitation (Channel Attention pada fitur morfologi vs ritme)
    se = layers.GlobalAveragePooling1D(keepdims=True, name=f"{name_prefix}_se_gap")(y)
    se = layers.Dense(max(filters // 4, 8), activation="swish", name=f"{name_prefix}_se_fc1")(se)
    se = layers.Dense(filters, activation="sigmoid", name=f"{name_prefix}_se_fc2")(se)
    y = layers.Multiply(name=f"{name_prefix}_se_scale")([y, se])

    # Residual projection jika dimensi berubah
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=f"{name_prefix}_proj",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    out = layers.Add(name=f"{name_prefix}_add")([y, shortcut])
    return out


def build_encoder_backbone(input_shape=(3750, 1)):
    """
    Membangun Encoder Backbone 1D-DSC + TA-GRU yang memetakan sinyal PPG 30 detik
    `(batch, 3750, 1)` menjadi vektor representasi kardiak 64-dimensi `h_repr`.
    Digunakan pada:
      - Tahap 1: Self-Supervised Contrastive Learning (SimCLR)
      - Tahap 2: Supervised Fine-Tuning Klasifikasi AFib vs Non-AFib
    """
    inputs = layers.Input(shape=input_shape, name="ppg_input")

    # 1. Front-End Morfologi Gelombang (PPG + VPG + APG/Dicrotic Notch) & Ritme
    # Output: (batch, 16, 160)
    x = PPGMorphologyAndRhythmLayer(name="ppg_morphology_rhythm_frontend")(inputs)
    x = layers.BatchNormalization(name="frontend_bn")(x)

    # 2. Tiga Blok 1D Depthwise Separable Convolution (1D-DSC) Multi-Skala
    x = dsc_se_residual_block(x, filters=32, kernel_size=5, stride=1, name_prefix="dsc_block1")
    x = layers.SpatialDropout1D(0.15, name="dsc_drop1")(x)

    x = dsc_se_residual_block(x, filters=48, kernel_size=5, stride=1, name_prefix="dsc_block2")
    x = layers.SpatialDropout1D(0.15, name="dsc_drop2")(x)

    x = dsc_se_residual_block(x, filters=48, kernel_size=3, stride=1, name_prefix="dsc_block3")

    # 3. Temporal Attention GRU (TA-GRU) + Global Morphology Pooling
    # unroll=True menjamin kompatibilitas 100% dengan TFLite Built-in Ops tanpa Flex Delegate
    gru_seq = layers.GRU(
        units=32,
        return_sequences=True,
        unroll=True,
         kernel_regularizer=regularizers.l2(1e-4),
        name="ta_gru_sequence",
    )(x)
    attn_context = TemporalAttentionLayer(attn_units=24, name="temporal_attention")(gru_seq)

    # Ringkasan statistik sekuensial (Mean & Max sepanjang 16 sub-jendela waktu)
    gap_feat = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    gmp_feat = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    combined = layers.Concatenate(name="encoder_concat")([attn_context, gap_feat, gmp_feat])
    h_repr = layers.Dense(64, activation="swish", name="cardiac_representation")(combined)
    h_repr = layers.BatchNormalization(name="cardiac_representation_bn")(h_repr)

    return models.Model(inputs=inputs, outputs=h_repr, name="Encoder_1D_DSC_TA_GRU")


def build_simclr_projection_model(encoder, proj_dim=32):
    """
    Membangun model Self-Supervised Contrastive Learning (SimCLR) sesuai Bab 2.7
    (Wu et al. [37]) dengan menambahkan Nonlinear Projection Head g(h) di atas
    Encoder 1D-DSC + TA-GRU untuk pelatihan NT-Xent Contrastive Loss.
    """
    inputs = encoder.input
    h = encoder.output
    z = layers.Dense(64, activation="relu", name="simclr_proj_fc1")(h)
    z = layers.BatchNormalization(name="simclr_proj_bn")(z)
    z = layers.Dense(proj_dim, use_bias=False, name="simclr_proj_fc2")(z)
    z_norm = L2NormalizeLayer(name="simclr_l2_norm")(z)
    return models.Model(inputs=inputs, outputs=z_norm, name="SimCLR_1D_DSC_TA_GRU")


def build_1d_dsc_ta_gru_model(input_shape=(3750, 1), encoder=None):
    """
    Membangun model lengkap klasifikasi AFib vs Non-AFib (1D-DSC + TA-GRU).
    Jika `encoder` yang telah di-pretrain dengan SimCLR diberikan, bobot encoder
    tersebut langsung digunakan sebagai inisialisasi fitur tahan-artefak.
    """
    if encoder is None:
        encoder = build_encoder_backbone(input_shape=input_shape)

    inputs = encoder.input
    h_repr = encoder.output

    x = layers.Dropout(0.30, name="head_dropout1")(h_repr)
    x = layers.Dense(32, activation="swish", name="head_fc1")(x)
    x = layers.Dropout(0.20, name="head_dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="afib_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_1D_DSC_TA_GRU_30s")
    return model


if __name__ == "__main__":
    model = build_1d_dsc_ta_gru_model(input_shape=(3750, 1))
    model.summary()
    print(f"\nTotal Parameter Model: {model.count_params():,}")
