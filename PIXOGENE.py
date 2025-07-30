from PIL import Image
import tkinter as tk
from tkinter import filedialog
import hashlib 
import binascii
import textwrap
import cv2
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
from multiprocessing import Pool
from bisect import bisect_left as bsearch

''' 
GLOBAL Constants
'''
# Hyper-chaotic parameters
a, b, c, d = 10, 8/3, 28, 1.3
x0, y0, z0, w0 = 0, 0, 0, 0

# Encryption rounds
ROUNDS = 2

# DNA Encoding Rules (8 variants)
DNA_RULES = [
    # Rule 0: Standard
    {"00": "A", "01": "T", "10": "G", "11": "C",
     "A": [0,0], "T": [0,1], "G": [1,0], "C": [1,1],
     "AA": "A", "TT": "A", "GG": "A", "CC": "A",
     "AG": "G", "GA": "G", "TC": "G", "CT": "G",
     "AC": "C", "CA": "C", "GT": "C", "TG": "C",
     "AT": "T", "TA": "T", "CG": "T", "GC": "T"},
    
    # Rule 1
    {"00": "C", "01": "G", "10": "T", "11": "A",
     "C": [0,0], "G": [0,1], "T": [1,0], "A": [1,1],
     "CC": "C", "GG": "C", "TT": "C", "AA": "C",
     "CT": "T", "TC": "T", "GA": "T", "AG": "T",
     "CA": "A", "AC": "A", "TG": "A", "GT": "A",
     "CG": "G", "GC": "G", "TA": "G", "AT": "G"},
    
    # Rule 2
    {"00": "T", "01": "A", "10": "C", "11": "G",
     "T": [0,0], "A": [0,1], "C": [1,0], "G": [1,1],
     "TT": "T", "AA": "T", "CC": "T", "GG": "T",
     "TA": "A", "AT": "A", "CG": "A", "GC": "A",
     "TG": "G", "GT": "G", "AC": "G", "CA": "G",
     "TC": "C", "CT": "C", "AG": "C", "GA": "C"},
    
    # Rule 3
    {"00": "G", "01": "C", "10": "A", "11": "T",
     "G": [0,0], "C": [0,1], "A": [1,0], "T": [1,1],
     "GG": "G", "CC": "G", "AA": "G", "TT": "G",
     "GC": "C", "CG": "C", "AT": "C", "TA": "C",
     "GA": "A", "AG": "A", "CT": "A", "TC": "A",
     "GT": "T", "TG": "T", "CA": "T", "AC": "T"},
    
    # Additional rules 4-7 would follow similar patterns
]

# Active DNA rule
dna = DNA_RULES[0]

# Maximum time point and total number of time points
tmax, N = 100, 10000

def hyper_lorenz(X, t, a, b, c, d):
    """4D Hyper-chaotic Lorenz system"""
    x, y, z, w = X
    dx = a*(y - x) + w
    dy = c*x - y - x*z
    dz = x*y - b*z
    dw = -d*x
    return dx, dy, dz, dw

def image_selector():
    """GUI for image selection with validation"""
    root = tk.Tk()
    root.withdraw()
    filetypes = (
        ('Image files', '*.jpg *.jpeg *.png *.bmp'),
        ('All files', '*.*')
    )
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path and os.path.isfile(path):
        print(f"Image loaded: {os.path.basename(path)}")
        return path
    print("Error: No image selected")
    sys.exit(1)

def split_into_rgb_channels(image):
    """Split image into RGB channels"""
    red = image[:,:,2]
    green = image[:,:,1]
    blue = image[:,:,0]
    return blue, green, red

def securekey(iname, user_key=""):
    """Generate secure key from image and optional user key"""
    try:
        img = Image.open(iname)
        m, n = img.size
        print(f"Image size: {m}x{n} pixels")
        pix = img.load()
        plainimage = []
        for y in range(n):
            for x in range(m):
                for k in range(3):
                    plainimage.append(pix[x,y][k])
        
        # Combine user key with image data
        combined = user_key.encode() + bytearray(plainimage)
        key = hashlib.sha256(combined).hexdigest()
        return key, m, n
    except Exception as e:
        print(f"Key generation error: {str(e)}")
        sys.exit(1)

def update_lorentz(key):
    """Update initial conditions using key material"""
    global x0, y0, z0, w0
    key_bin = bin(int(key, 16))[2:].zfill(256)
    key_parts = textwrap.wrap(key_bin, 8)
    
    t1 = t2 = t3 = t4 = 0
    for i in range(8):  # First 8 bytes
        t1 ^= int(key_parts[i], 2)
    for i in range(8, 16):  # Next 8 bytes
        t2 ^= int(key_parts[i], 2)
    for i in range(16, 24):  # Next 8 bytes
        t3 ^= int(key_parts[i], 2)
    for i in range(24, 32):  # Last 8 bytes
        t4 ^= int(key_parts[i], 2)
    
    # Update initial conditions
    x0 += t1 / 255.0
    y0 += t2 / 255.0
    z0 += t3 / 255.0
    w0 += t4 / 255.0
    
    # Select DNA rule based on initial conditions
    rule_idx = int((abs(w0) * 1000) % len(DNA_RULES))
    global dna
    dna = DNA_RULES[rule_idx]
    print(f"Using DNA Rule #{rule_idx}")

def decompose_matrix(iname):
    """Decompose image into RGB matrices with error handling"""
    try:
        image = cv2.imread(iname)
        if image is None:
            raise ValueError("Could not read image")
        return split_into_rgb_channels(image)
    except Exception as e:
        print(f"Image decomposition error: {str(e)}")
        sys.exit(1)

def dna_encode(channel):
    """Vectorized DNA encoding for a single channel"""
    bits = np.unpackbits(channel, axis=1)
    high, low = bits[:, 0::2], bits[:, 1::2]
    indices = (high << 1) + low
    shape = indices.shape
    flat = indices.flatten()
    
    # Vectorized mapping
    encoded = np.vectorize(lambda x: dna[f"{x:02b}"])(flat)
    return encoded.reshape(shape)

def dna_encode_parallel(b, g, r):
    """Parallel DNA encoding for all channels"""
    with Pool(3) as pool:
        results = pool.map(dna_encode, [b, g, r])
    return results

def key_matrix_encode(key, channel):
    """Generate encoded key matrix"""
    bits = np.unpackbits(channel, axis=1)
    m, n = bits.shape
    key_bin = bin(int(key, 16))[2:].zfill(256)
    
    # Generate key matrix
    Mk = np.zeros((m, n), dtype=np.uint8)
    for j in range(m):
        for i in range(n):
            Mk[j, i] = int(key_bin[(j * n + i) % 256])
    
    # DNA encode key matrix
    high, low = Mk[:, 0::2], Mk[:, 1::2]
    indices = (high << 1) + low
    shape = indices.shape
    flat = indices.flatten()
    
    # Vectorized mapping
    Mk_enc = np.vectorize(lambda x: dna[f"{x:02b}"])(flat)
    return Mk_enc.reshape(shape)

def xor_operation(b, g, r, mk):
    """Parallel XOR operation for all channels"""
    def xor_channel(channel, mk):
        m, n = channel.shape
        result = np.chararray((m, n))
        for i in range(m):
            for j in range(n):
                result[i, j] = dna[f"{channel[i,j]}{mk[i,j]}"]
        return result.astype(str)
    
    with Pool(3) as pool:
        results = pool.starmap(xor_channel, [(b, mk), (g, mk), (r, mk)])
    return results

def gen_chaos_seq(length, precision=np.float64):
    """Generate hyper-chaotic sequences with fixed precision"""
    global x0, y0, z0, w0, a, b, c, d
    t = np.linspace(0, tmax, length, dtype=precision)
    f = odeint(hyper_lorenz, (x0, y0, z0, w0), t, 
                args=(a, b, c, d), rtol=1e-12, atol=1e-12)
    x, y, z, w = f.T
    
    # Update initial conditions for next round
    x0, y0, z0, w0 = x[-1], y[-1], z[-1], w[-1]
    return x, y, z, w

def sequence_indexing(seq):
    """Efficient sequence indexing using argsort"""
    indices = np.argsort(seq)
    ranks = np.argsort(indices)
    return ranks

def scramble(fx, fy, fz, b, g, r):
    """Scramble channels using chaotic indices"""
    def apply_scramble(indices, channel):
        flat = channel.reshape(-1)
        scrambled = flat[indices].reshape(channel.shape)
        return scrambled
    
    with Pool(3) as pool:
        results = pool.starmap(apply_scramble, 
                              [(fz, b), (fy, g), (fx, r)])
    return results

def dna_decode(channel_enc):
    """Vectorized DNA decoding for a single channel"""
    m, n = channel_enc.shape
    decoded = np.zeros((m, n*2), dtype=np.uint8)
    
    for j in range(m):
        for i in range(n):
            base = channel_enc[j, i]
            decoded[j, 2*i] = dna[base][0]
            decoded[j, 2*i+1] = dna[base][1]
    
    return np.packbits(decoded, axis=1)

def dna_decode_parallel(b_enc, g_enc, r_enc):
    """Parallel DNA decoding for all channels"""
    with Pool(3) as pool:
        results = pool.map(dna_decode, [b_enc, g_enc, r_enc])
    return results

def apply_chaotic_diffusion(channel, chaotic_seq):
    """Apply chaotic diffusion to a channel"""
    # Normalize and quantize chaotic sequence
    seq_min, seq_max = chaotic_seq.min(), chaotic_seq.max()
    normalized = (chaotic_seq - seq_min) / (seq_max - seq_min)
    int_seq = (normalized * 255).astype(np.uint8)
    
    # Reshape to match channel dimensions
    int_seq = int_seq[:channel.size].reshape(channel.shape)
    
    # Apply XOR diffusion
    return np.bitwise_xor(channel, int_seq)

def recover_image(b, g, r, iname, suffix="_enc"):
    """Save encrypted/decrypted image"""
    dir_name, file_name = os.path.split(iname)
    base_name, ext = os.path.splitext(file_name)
    output_path = os.path.join(dir_name, f"{base_name}{suffix}{ext}")
    
    # Create empty RGB image
    img = np.zeros((b.shape[0], b.shape[1], 3), dtype=np.uint8)
    img[:,:,0] = b
    img[:,:,1] = g
    img[:,:,2] = r
    
    cv2.imwrite(output_path, img)
    print(f"Saved image: {output_path}")
    return img

def security_metrics(orig, enc):
    """Calculate NPCR and UACI for security validation"""
    if orig.shape != enc.shape:
        raise ValueError("Images must have same dimensions")
    
    # Calculate NPCR (Number of Pixels Change Rate)
    diff = orig != enc
    npcr = np.mean(diff) * 100
    
    # Calculate UACI (Unified Average Changing Intensity)
    uaci = np.mean(np.abs(orig.astype(int) - enc.astype(int)) / 255 * 100
    
    return npcr, uaci

def encrypt_image(iname, user_key="", rounds=ROUNDS):
    """Full encryption pipeline"""
    start_time = time.time()
    key, m, n = securekey(iname, user_key)
    update_lorentz(key)
    
    # Load and decompose original image
    orig_image = cv2.imread(iname)
    b_orig, g_orig, r_orig = decompose_matrix(iname)
    
    # Encryption rounds
    for round_idx in range(rounds):
        print(f"\n--- Encryption Round {round_idx+1}/{rounds} ---")
        
        # Generate chaotic sequences
        seq_length = m * n * 3
        x, y, z, w = gen_chaos_seq(seq_length)
        
        # Sequence indexing
        fx = sequence_indexing(x)
        fy = sequence_indexing(y)
        fz = sequence_indexing(z)
        
        # DNA encoding
        b_enc, g_enc, r_enc = dna_encode_parallel(b_orig, g_orig, r_orig)
        
        # Key matrix
        Mk_enc = key_matrix_encode(key, b_orig)
        
        # XOR operation
        b_xor, g_xor, r_xor = xor_operation(b_enc, g_enc, r_enc, Mk_enc)
        
        # Scrambling
        b_scram, g_scram, r_scram = scramble(fz, fy, fx, b_xor, g_xor, r_xor)
        
        # DNA decoding
        b_dec, g_dec, r_dec = dna_decode_parallel(b_scram, g_scram, r_scram)
        
        # Apply chaotic diffusion
        b_orig = apply_chaotic_diffusion(b_dec, w[:m*n].reshape(m, n))
        g_orig = apply_chaotic_diffusion(g_dec, w[m*n:2*m*n].reshape(m, n))
        r_orig = apply_chaotic_diffusion(r_dec, w[2*m*n:].reshape(m, n))
    
    # Save encrypted image
    enc_image = recover_image(b_orig, g_orig, r_orig, iname, "_enc")
    
    # Security analysis
    npcr, uaci = security_metrics(cv2.imread(iname), enc_image)
    print(f"\nSecurity Metrics:")
    print(f"NPCR: {npcr:.6f}% (Target > 99.6%)")
    print(f"UACI: {uaci:.6f}% (Target > 33.4%)")
    
    print(f"\nEncryption completed in {time.time()-start_time:.2f} seconds")
    return enc_image, (x, y, z, w, fx, fy, fz, Mk_enc, key)

def decrypt_image(iname, params, user_key="", rounds=ROUNDS):
    """Full decryption pipeline"""
    start_time = time.time()
    x, y, z, w, fx, fy, fz, Mk_enc, orig_key = params
    
    # Load encrypted image
    enc_image = cv2.imread(iname)
    b_enc, g_enc, r_enc = split_into_rgb_channels(enc_image)
    m, n = b_enc.shape
    
    # Verify key
    test_key, _, _ = securekey(iname, user_key)
    if test_key != orig_key:
        print("Warning: User key doesn't match encryption key!")
    
    # Decryption rounds (in reverse order)
    for round_idx in range(rounds-1, -1, -1):
        print(f"\n--- Decryption Round {rounds-round_idx}/{rounds} ---")
        
        # Reverse chaotic diffusion
        w_seq = w[:m*n].reshape(m, n)
        b_enc = apply_chaotic_diffusion(b_enc, w_seq)
        g_enc = apply_chaotic_diffusion(g_enc, w_seq)
        r_enc = apply_chaotic_diffusion(r_enc, w_seq)
        
        # DNA encoding
        b_enc, g_enc, r_enc = dna_encode_parallel(b_enc, g_enc, r_enc)
        
        # Scrambling (reverse)
        b_scram, g_scram, r_scram = scramble(fz, fy, fx, b_enc, g_enc, r_enc)
        
        # XOR operation
        b_xor, g_xor, r_xor = xor_operation(b_scram, g_scram, r_scram, Mk_enc)
        
        # DNA decoding
        b_dec, g_dec, r_dec = dna_decode_parallel(b_xor, g_xor, r_xor)
        
        # Prepare for next round
        b_enc, g_enc, r_enc = b_dec, g_dec, r_dec
    
    # Save decrypted image
    dec_image = recover_image(b_enc, g_enc, r_enc, iname, "_dec")
    print(f"\nDecryption completed in {time.time()-start_time:.2f} seconds")
    return dec_image

# Main program
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Encryption: python pixogene.py encrypt [user_key]")
        print("  Decryption: python pixogene.py decrypt [user_key]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    user_key = sys.argv[2] if len(sys.argv) > 2 else ""
    
    if mode not in ["encrypt", "decrypt"]:
        print("Invalid mode. Use 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    file_path = image_selector()
    
    if mode == "encrypt":
        # Encryption process
        enc_image, params = encrypt_image(file_path, user_key)
        
        # Save parameters for decryption
        param_path = os.path.splitext(file_path)[0] + "_params.npy"
        np.save(param_path, params)
        print(f"Saved encryption parameters to {param_path}")
        
    elif mode == "decrypt":
        # Load parameters
        param_path = os.path.splitext(file_path)[0] + "_params.npy"
        if not os.path.exists(param_path):
            print("Error: Encryption parameters not found")
            sys.exit(1)
            
        params = np.load(param_path, allow_pickle=True)
        decrypt_image(file_path, params, user_key)
