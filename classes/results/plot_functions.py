import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Funkce pro načítání dat ---
def get_data_stats(base_path, num_files=10):
    steps_data = {}
    
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        try:
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        epoch = int(parts[0])
                        step = int(parts[2])

                        if epoch not in steps_data:
                            steps_data[epoch] = []
                        steps_data[epoch].append(step)
        except FileNotFoundError:
            # print(f"Varování: Soubor {filename} nebyl nalezen.") # Tiché
            pass

    epochs_avg = []
    steps_avg = []
    steps_p25 = []
    steps_p75 = []

    for epoch in sorted(steps_data.keys()):
        steps_for_this_epoch = steps_data[epoch]
        
        median_step = np.median(steps_for_this_epoch)
        p25 = np.percentile(steps_for_this_epoch, 25)
        p75 = np.percentile(steps_for_this_epoch, 75)

        epochs_avg.append(epoch)
        steps_avg.append(median_step)
        steps_p25.append(p25)
        steps_p75.append(p75)
        
    return epochs_avg, steps_avg, steps_p25, steps_p75

# --- NOVÁ FUNKCE: Odstranění mezery a posun dat ---
def remove_gap_and_shift(epochs, values, p25, p75):
    """
    Odstraní data mezi 100 a 150.
    Data >= 150 posune o 50 doleva.
    """
    new_e, new_v, new_p25, new_p75 = [], [], [], []
    
    for e, v, p2, p7 in zip(epochs, values, p25, p75):
        if e < 100:
            # Původní data do 100
            new_e.append(e)
            new_v.append(v)
            new_p25.append(p2)
            new_p75.append(p7)
        elif e >= 150:
            # Data od 150 posuneme o 50 zpět (přilepíme je k 100)
            new_e.append(e - 50) 
            new_v.append(v)
            new_p25.append(p2)
            new_p75.append(p7)
            
    return new_e, new_v, new_p25, new_p75

# --- 2. Konfigurace a Načtení dat ---
num_files_cur = 10

# Definice modelů
models = [
    # (Název, Cesta, Barva, Styl)
    ("Linear Regression", "/Users/jakubmuller/Desktop/WORK/iwinac2026/results_data/cur_lin_", "hotpink", "--"),
    ("SVR",               "/Users/jakubmuller/Desktop/WORK/iwinac2026/results_data/cur_svr_", "grey", "--"),
    ("Gradient Boosting", "/Users/jakubmuller/Desktop/WORK/iwinac2026/results_data/cur_grad_", "gold", "--"),
    ("MLP",               "/Users/jakubmuller/Desktop/WORK/iwinac2026/results_data/cur_mlp_", "lime", "--"),
    ("Motivational Engine", "/Users/jakubmuller/Desktop/WORK/iwinac2026/results_data/cur_", "blue", "-")
]

# --- 3. Vykreslení grafu ---
fig, ax = plt.subplots(figsize=(12, 8))

for label, path, color, style in models:
    print(f"Zpracovávám: {label}...")
    ep, avg, p25, p75 = get_data_stats(path, num_files_cur)
    
    # === APLIKACE POSUNU DAT ===
    ep, avg, p25, p75 = remove_gap_and_shift(ep, avg, p25, p75)
    
    # Vykreslení čáry (median)
    ax.plot(ep, avg, color=color, linestyle=style, label=label, linewidth=1.5)
    
    # Vykreslení oblasti chyby
    if color == 'gold':
        alpha_val = 0.1
    else: 
        alpha_val = 0.2
        
    ax.fill_between(ep, p25, p75, color=color, alpha=alpha_val)

# --- 4. Formátování a popisky ---

# Svislá čára pro změnu kontextu - POSUNUTA NA 100 (původně 150)
ax.axvline(x=100, color='black', linestyle=':', linewidth=1.5, alpha=0.8)

# Text posunut k čáře
ax.text(100 + 30, 190, 'Goal position change', fontsize=14)

# Mřížka a popisky os
ax.grid(linestyle='-', alpha=0.3)
ax.set_xlabel("Epochs", fontsize=14)
ax.set_ylabel("Steps", fontsize=14)

# Ticks - upraveno pro zkrácenou osu
x_ticks = np.arange(0, 201, 25)
y_ticks = np.arange(0, 201, 25)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.tick_params(axis='both', which='major', labelsize=14)

# Limity - ZKRÁCENO NA 200
ax.set_xlim(0, 200)
ax.set_ylim(0, 205)

# Legenda
ax.legend(loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig("/Users/jakubmuller/Desktop/WORK/iwinac2026/plot_comparison_all_in_one.png", dpi=300)
plt.show()