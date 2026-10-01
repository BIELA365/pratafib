"""
================================================================================
 02_model_architecture.py
 Arsitektur 1D Temporal Attention Depthwise Separable Convolution (1D-TA-DSC)
================================================================================
 Model deep learning ringan untuk klasifikasi Atrial Fibrillation (AFib)
 dari sinyal PPG (Photoplethysmography).
 
 Dirancang khusus untuk deployment pada mikrokontroler ESP32-S3:
   - Target parameter: < 50.000 (agar model INT8 ≈ 40-50 KB)
   - Input: 625 sampel (5 detik @ 125 Hz) × 1 kanal
   - Output: 2 kelas (Normal, AFib)
 
 Komponen arsitektur:
   1. Initial Conv1D — menangkap morfologi dasar gelombang PPG
   2. Depthwise Separable Conv1D (DSC) — ekstraksi fitur efisien
   3. Temporal Attention — bobot adaptif pada interval temporal
   4. Global Average Pooling + Dense Classifier
   
 Penggunaan:
   from model_architecture import build_1d_ta_dsc_model
   model = build_1d_ta_dsc_model()
   model.summary()
================================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ============================================================================
# LAYER KUSTOM: Temporal Attention
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom", name="TemporalAttention")
class TemporalAttention(layers.Layer):
    """
    Mekanisme Self-Attention 1D untuk sinyal temporal.
    
    Menghitung bobot atensi pada setiap timestep berdasarkan hubungan
    Query-Key, lalu menghasilkan representasi kontekstual melalui 
    penjumlahan tertimbang Value.
    
    Berbeda dari Multi-Head Attention standar, layer ini menggunakan
    single-head attention yang lebih ringan untuk deployment embedded.
    
    Args:
        d_model: dimensi ruang atensi (default: sesuai input)
        
    Input shape:  (batch, timesteps, features)
    Output shape: (batch, timesteps, features)
    """
    
    def __init__(self, d_model=None, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.d_model = d_model
    
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        d = self.d_model if self.d_model else feature_dim
        
        # Proyeksi Query, Key, Value
        self.query_dense = layers.Dense(d, use_bias=False, name="attn_query")
        self.key_dense = layers.Dense(d, use_bias=False, name="attn_key")
        self.value_dense = layers.Dense(d, use_bias=False, name="attn_value")
        
        # Proyeksi output
        self.output_dense = layers.Dense(feature_dim, use_bias=False, 
                                          name="attn_output")
        
        # Layer normalization untuk stabilitas numerik
        self.layer_norm = layers.LayerNormalization(name="attn_norm")
        
        self.scale = tf.math.sqrt(tf.cast(d, tf.float32))
        super(TemporalAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Hitung Q, K, V
        Q = self.query_dense(inputs)    # (batch, T, d)
        K = self.key_dense(inputs)      # (batch, T, d)
        V = self.value_dense(inputs)    # (batch, T, d)
        
        # Scaled dot-product attention
        # scores = Q · K^T / sqrt(d)
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale  # (batch, T, T)
        attention_weights = tf.nn.softmax(scores, axis=-1)         # (batch, T, T)
        
        # Context = weighted sum of Values
        context = tf.matmul(attention_weights, V)  # (batch, T, d)
        
        # Proyeksi output
        output = self.output_dense(context)  # (batch, T, features)
        
        # Residual connection + layer normalization
        output = self.layer_norm(output + inputs)
        
        return output
    
    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        config.update({"d_model": self.d_model})
        return config


# ============================================================================
# BLOK DEPTHWISE SEPARABLE CONVOLUTION
# ============================================================================

def depthwise_separable_conv_block(x, filters, kernel_size, name_prefix):
    """
    Blok Depthwise Separable Convolution 1D.
    
    Terdiri dari:
      1. Depthwise Conv1D — konvolusi per-kanal (hemat parameter)
      2. Pointwise Conv1D (1×1) — kombinasi antar-kanal
      3. BatchNormalization + ReLU
    
    Dibandingkan Conv1D standar:
      - Conv1D standar: in_ch × out_ch × kernel_size parameter
      - DSC: in_ch × kernel_size + in_ch × out_ch parameter
      - Penghematan: ~kernel_size kali lebih sedikit parameter
    
    Args:
        x: input tensor (batch, timesteps, channels)
        filters: jumlah filter output (pointwise)
        kernel_size: ukuran kernel depthwise
        name_prefix: prefix untuk penamaan layer
    
    Returns:
        output tensor (batch, timesteps, filters)
    """
    # Depthwise convolution (konvolusi independen per kanal)
    x = layers.DepthwiseConv1D(
        kernel_size=kernel_size,
        padding='same',
        use_bias=False,
        name=f"{name_prefix}_depthwise"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    
    # Pointwise convolution (1×1 conv untuk mixing antar kanal)
    x = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=f"{name_prefix}_pointwise"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    
    return x


# ============================================================================
# MODEL UTAMA: 1D-TA-DSC
# ============================================================================

def build_1d_ta_dsc_model(
    input_length=625,
    n_channels=1,
    n_classes=2,
    initial_filters=16,
    dsc_configs=None,
    attention_dim=None,
    dropout_rate=0.3,
):
    """
    Membangun model 1D Temporal Attention Depthwise Separable Convolution.
    
    Arsitektur:
        Input(625,1) → Conv1D(16,k7) → MaxPool(2)
                      → DSC(32,k5) → MaxPool(2)  
                      → DSC(64,k5) → MaxPool(2)
                      → TemporalAttention(64)
                      → DSC(64,k3) → GlobalAvgPool
                      → Dense(32) → Dropout → Dense(2,softmax)
    
    Args:
        input_length: panjang sinyal input (default: 625 = 5s × 125Hz)
        n_channels: jumlah kanal input (default: 1 untuk PPG tunggal)
        n_classes: jumlah kelas output (default: 2 = Normal + AFib)
        initial_filters: filter pada layer konvolusi pertama
        dsc_configs: list tuple (filters, kernel_size) untuk blok DSC
        attention_dim: dimensi ruang atensi (None = otomatis)
        dropout_rate: rasio dropout sebelum classifier
    
    Returns:
        model: keras.Model yang siap di-compile dan dilatih
    """
    if dsc_configs is None:
        dsc_configs = [
            (32, 5),   # DSC Block 1
            (64, 5),   # DSC Block 2 (sebelum attention)
            (64, 3),   # DSC Block 3 (setelah attention)
        ]
    
    # ---- Input Layer ----
    inputs = layers.Input(shape=(input_length, n_channels), name="ppg_input")
    
    # ---- Blok 1: Initial Convolution ----
    # Menangkap morfologi dasar gelombang PPG (sistol, diastol, notch)
    x = layers.Conv1D(
        filters=initial_filters,
        kernel_size=7,
        padding='same',
        use_bias=False,
        name="initial_conv"
    )(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.ReLU(name="initial_relu")(x)
    x = layers.MaxPooling1D(pool_size=2, name="initial_pool")(x)
    # Shape: (batch, 312, 16)
    
    # ---- Blok 2: DSC Block 1 ----
    filters_1, ks_1 = dsc_configs[0]
    x = depthwise_separable_conv_block(x, filters_1, ks_1, "dsc1")
    x = layers.MaxPooling1D(pool_size=2, name="dsc1_pool")(x)
    # Shape: (batch, 156, 32)
    
    # ---- Blok 3: DSC Block 2 ----
    filters_2, ks_2 = dsc_configs[1]
    x = depthwise_separable_conv_block(x, filters_2, ks_2, "dsc2")
    x = layers.MaxPooling1D(pool_size=2, name="dsc2_pool")(x)
    # Shape: (batch, 78, 64)
    
    # ---- Blok 4: Temporal Attention ----
    # Memberikan bobot adaptif pada momen temporal yang paling informatif
    # (contoh: interval antar puncak sistolik yang ireguler pada AFib)
    x = TemporalAttention(d_model=attention_dim, name="temporal_attention")(x)
    # Shape: (batch, 78, 64)
    
    # ---- Blok 5: DSC Block 3 ----
    filters_3, ks_3 = dsc_configs[2]
    x = depthwise_separable_conv_block(x, filters_3, ks_3, "dsc3")
    # Shape: (batch, 78, 64)
    
    # ---- Global Average Pooling ----
    # Meringkas seluruh representasi temporal menjadi vektor fitur tunggal
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    # Shape: (batch, 64)
    
    # ---- Classifier Head ----
    x = layers.Dense(32, activation='relu', name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation='softmax', name="classifier")(x)
    # Shape: (batch, 2)
    
    model = Model(inputs=inputs, outputs=outputs, name="1D_TA_DSC_AFib")
    
    return model


# ============================================================================
# VALIDASI ARSITEKTUR
# ============================================================================

def validate_architecture(model, max_params=50000, max_tflite_kb=100):
    """
    Memvalidasi bahwa model memenuhi constraint untuk deployment ESP32-S3.
    
    Checks:
      1. Jumlah parameter total < max_params
      2. Estimasi ukuran model TFLite INT8 < max_tflite_kb
      3. Input/output shape sesuai spesifikasi
    """
    total_params = model.count_params()
    trainable = sum([tf.keras.backend.count_params(w) 
                     for w in model.trainable_weights])
    non_trainable = total_params - trainable
    
    # Estimasi ukuran TFLite INT8 (≈1 byte per parameter + overhead)
    estimated_tflite_kb = (total_params * 1 + 2048) / 1024  # 1 byte/param + 2KB overhead
    
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print("\n" + "=" * 70)
    print(" VALIDASI ARSITEKTUR MODEL 1D-TA-DSC")
    print("=" * 70)
    print(f"  Input shape:        {input_shape}")
    print(f"  Output shape:       {output_shape}")
    print(f"  Total parameter:    {total_params:,}")
    print(f"    - Trainable:      {trainable:,}")
    print(f"    - Non-trainable:  {non_trainable:,}")
    print(f"  Est. TFLite INT8:   ~{estimated_tflite_kb:.1f} KB")
    print("-" * 70)
    
    # Validasi constraints
    checks = []
    
    if total_params <= max_params:
        checks.append(f"  [PASS] Parameter ({total_params:,}) <= {max_params:,}")
    else:
        checks.append(f"  [FAIL] Parameter ({total_params:,}) > {max_params:,} — TERLALU BESAR!")
    
    if estimated_tflite_kb <= max_tflite_kb:
        checks.append(f"  [PASS] Est. TFLite ({estimated_tflite_kb:.1f} KB) <= {max_tflite_kb} KB")
    else:
        checks.append(f"  [FAIL] Est. TFLite ({estimated_tflite_kb:.1f} KB) > {max_tflite_kb} KB — TERLALU BESAR!")
    
    if input_shape == (None, 625, 1):
        checks.append(f"  [PASS] Input shape sesuai: (batch, 625, 1)")
    else:
        checks.append(f"  [WARN] Input shape: {input_shape} (expected (None, 625, 1))")
    
    if output_shape == (None, 2):
        checks.append(f"  [PASS] Output shape sesuai: (batch, 2)")
    else:
        checks.append(f"  [WARN] Output shape: {output_shape} (expected (None, 2))")
    
    for c in checks:
        print(c)
    
    print("=" * 70)
    
    all_pass = all("[PASS]" in c for c in checks)
    if all_pass:
        print("  [INFO] Model LOLOS semua validasi — siap untuk deployment ESP32-S3!")
    else:
        print("  [WARN] Ada constraint yang belum terpenuhi. Review arsitektur.")
    print("=" * 70)
    
    return all_pass


# ============================================================================
# MAIN — Bangun & Tampilkan Arsitektur
# ============================================================================

if __name__ == "__main__":
    print("Membangun arsitektur model 1D-TA-DSC...")
    model = build_1d_ta_dsc_model()
    
    # Tampilkan ringkasan arsitektur
    model.summary(line_length=90)
    
    # Validasi constraint untuk ESP32-S3
    validate_architecture(model)
