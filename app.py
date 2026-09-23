# -*- coding: utf-8 -*-
"""
Synapsit v1.4 - El juego de ciencia ciudadana para personalizar la neuromodulación.
Incluye guía educativa individual debajo de cada perilla con fondo oscuro.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hilbert, butter, filtfilt
from scipy.optimize import minimize
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import json
import os
import hashlib
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN GLOBAL
# =============================================================================
FS = 200.0
DURATION = 10.0
NPERSEG = 256
NOVERLAP = 128
BANDS = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 80)}

# =============================================================================
# DATOS DE LOS PACIENTES (REALES)
# =============================================================================
TD_COMUN = {
    "g_nmda_ei": 0.8256212946855831,
    "g_nmda_ee": 1.847900162072316,
    "low_freq_amp": 0.05964095845013163
}

LEVELS = {
    "1": {
        "name": "caso1",
        "td": TD_COMUN.copy(),
        "asd": {
            "g_nmda_ei": 0.8256262038208563,
            "g_nmda_ee": 1.8479050691979817,
            "low_freq_amp": 0.05964096618456388
        },
        "desc": "Paciente ASD_EEG5 (TD común)"
    },
    "2": {
        "name": "caso2",
        "td": TD_COMUN.copy(),
        "asd": {"g_nmda_ei": 0.82, "g_nmda_ee": 1.85, "low_freq_amp": 0.06},
        "desc": "Paciente ASD_EEG3 - reemplazar con datos reales"
    },
    "3": {
        "name": "caso3",
        "td": TD_COMUN.copy(),
        "asd": {"g_nmda_ei": 0.80, "g_nmda_ee": 1.90, "low_freq_amp": 0.05},
        "desc": "Paciente ASD_EEG4 - reemplazar con datos reales"
    }
}

# =============================================================================
# PERILLAS
# =============================================================================
PARAM_NAMES = ['GABA_A', 'NMDA_NR2A', 'D2_Dopamine', 'NMDA_NR2B',
               'GABA_B', '5HT2A', 'Dopamine', 'Serotonin', 'Norepinephrine']
PARAM_RANGES = {
    'GABA_A': (-0.8, 0.8), 'NMDA_NR2A': (-0.8, 0.8), 'D2_Dopamine': (-0.8, 0.8),
    'NMDA_NR2B': (-0.8, 0.8), 'GABA_B': (-0.8, 0.8), '5HT2A': (-0.8, 0.8),
    'Dopamine': (-1.0, 1.0), 'Serotonin': (-1.0, 1.0), 'Norepinephrine': (-1.0, 1.0)
}
PARAM_DEFAULTS = {name: 0.0 for name in PARAM_NAMES}

# =============================================================================
# DICCIONARIO DE DEFINICIONES EDUCATIVAS (HTML)
# =============================================================================
DEFINICIONES = {
    "GABA_A": """
    <b>GABA_A</b>: Receptor ionotrópico de GABA (canal de cloro).
    <br>• <b>Agonista</b> → aumenta la inhibición neuronal (efecto calmante).
    <br>• <b>Inhibidor</b> → disminuye la inhibición (efecto excitante).
    <br>• <b>Fármacos</b>: benzodiazepinas, barbitúricos (agonistas); flumazenil (antagonista).
    """,
    "NMDA_NR2A": """
    <b>NMDA (NR2A)</b>: Subtipo de receptor de glutamato (ionotrópico).
    <br>• <b>Agonista</b> → aumenta la excitación y plasticidad sináptica.
    <br>• <b>Inhibidor</b> → reduce la excitación (efecto neuroprotector).
    <br>• <b>Fármacos</b>: ketamina, memantina (antagonistas); D-cicloserina (modulador).
    """,
    "NMDA_NR2B": """
    <b>NMDA (NR2B)</b>: Subtipo con subunidad NR2B, mayor afinidad por glutamato.
    <br>• <b>Agonista</b> → potencia la señal excitatoria.
    <br>• <b>Inhibidor</b> → bloquea selectivamente NR2B (efecto antiexcitotóxico).
    <br>• <b>Fármacos</b>: ifenprodil, eliprodil (antagonistas NR2B).
    """,
    "GABA_B": """
    <b>GABA_B</b>: Receptor metabotrópico de GABA (acoplado a proteína G).
    <br>• <b>Agonista</b> → inhibición prolongada (efecto relajante).
    <br>• <b>Inhibidor</b> → reduce la inhibición (efecto desinhibidor).
    <br>• <b>Fármacos</b>: baclofeno (agonista); saclofen (antagonista).
    """,
    "5HT2A": """
    <b>5HT2A</b>: Receptor de serotonina (metabotrópico, acoplado a Gq).
    <br>• <b>Agonista</b> → aumenta excitación cortical (efecto psicoactivo).
    <br>• <b>Inhibidor</b> → bloquea la señal serotoninérgica.
    <br>• <b>Fármacos</b>: psilocibina, LSD (agonistas); ciproheptadina (antagonista).
    """,
    "D2_Dopamine": """
    <b>D2 (Dopamina)</b>: Receptor dopaminérgico D2 (metabotrópico, acoplado a Gi).
    <br>• <b>Agonista</b> → modula la entrada tónica (efecto antiparkinsoniano).
    <br>• <b>Inhibidor</b> → bloquea la señal dopaminérgica (efecto antipsicótico).
    <br>• <b>Fármacos</b>: bromocriptina, pramipexol (agonistas); haloperidol, risperidona (antagonistas).
    """,
    "Dopamine": """
    <b>Dopamina</b>: Neuromodulador catecolaminérgico.
    <br>• <b>Valor positivo</b> → aumenta liberación dopaminérgica (motivación, movimiento).
    <br>• <b>Valor negativo</b> → reduce la actividad dopaminérgica.
    <br>• <b>Rol</b>: implicado en Parkinson, adicción, esquizofrenia.
    """,
    "Serotonin": """
    <b>Serotonina (5HT)</b>: Neuromodulador indolaminérgico.
    <br>• <b>Valor positivo</b> → aumenta la transmisión serotoninérgica (ánimo, sueño).
    <br>• <b>Valor negativo</b> → reduce la actividad serotoninérgica.
    <br>• <b>Rol</b>: implicado en depresión, ansiedad, migraña.
    """,
    "Norepinephrine": """
    <b>Noradrenalina (NE)</b>: Neuromodulador catecolaminérgico (locus coeruleus).
    <br>• <b>Valor positivo</b> → aumenta la alerta y atención.
    <br>• <b>Valor negativo</b> → reduce la actividad noradrenérgica.
    <br>• <b>Rol</b>: implicado en TDAH, depresión, estrés.
    """
}

# =============================================================================
# 2. MODELO NEURAL MASS
# =============================================================================
class NeuralMass:
    """Modelo de masa neural de Jansen-Rit simplificado."""
    def __init__(self, g_nmda_ei=1.0, g_nmda_ee=1.0, low_freq_amp=0.0,
                 g_i_delta=0.0, g_e_delta=0.0, p_base_delta=0.0,
                 g_nmda_ee_delta=0.0, gaba_b_delta=0.0, nmda_nr2b_delta=0.0,
                 dopamine_mod=0.0, serotonin_mod=0.0, norepinephrine_mod=0.0,
                 w_ee=15.0, w_ei1_factor=1.0, tau_e=0.005, fs=FS):
        self.fs = fs; self.dt = 1.0/fs
        self.g_nmda_ei = g_nmda_ei + g_i_delta
        self.g_nmda_ee = g_nmda_ee + g_nmda_ee_delta + g_e_delta
        self.gaba_b = 1.0 + gaba_b_delta
        self.nmda_nr2b = 1.0 + nmda_nr2b_delta
        self.dopamine = dopamine_mod
        self.serotonin = serotonin_mod
        self.norepinephrine = norepinephrine_mod
        p_base_extra = 0.0
        if self.dopamine != 0:
            p_base_extra += 0.15 * self.dopamine
            self.g_nmda_ei -= 0.1 * self.dopamine
        if self.serotonin != 0:
            self.g_nmda_ee += 0.2 * self.serotonin
            self.g_nmda_ei += 0.1 * self.serotonin
        if self.norepinephrine != 0:
            self.g_nmda_ee += 0.15 * self.norepinephrine
            tau_e = tau_e * (1.0 - 0.2 * self.norepinephrine)
        self.low_freq_amp = low_freq_amp + p_base_delta + p_base_extra
        self.tau_e = max(0.002, min(0.02, tau_e))
        self.tau_i1 = 0.008; self.tau_i2 = 0.050
        self.w_ee = w_ee
        self.w_ei1 = 12.0 * self.g_nmda_ei * w_ei1_factor
        self.w_ei2 = 6.0
        self.w_i1e = -10.0; self.w_i2e = -4.0
        self.w_ei1 = self.w_ei1 * self.gaba_b
        self.p_base = 3.0 + p_base_delta + p_base_extra
        self.sigma_p = 0.8; self.noise_amp = 0.4

    def derivs(self, state, t):
        E, I1, I2 = state
        def sig(x, thr=2.5):
            return 1.0/(1.0+np.exp(-(x-thr)))
        I_slow = self.low_freq_amp * np.sin(2*np.pi*2.0*t)
        p = self.p_base + self.sigma_p*np.random.randn() + I_slow
        I_syn_E = self.w_ee * self.g_nmda_ee * sig(E) + self.w_i1e*sig(I1) + self.w_i2e*sig(I2) + p
        I_syn_I1 = self.w_ei1 * sig(E)
        I_syn_I2 = self.w_ei2 * sig(E)
        dE = (-E + I_syn_E)/self.tau_e + self.noise_amp*np.random.randn()
        dI1 = (-I1 + I_syn_I1)/self.tau_i1 + self.noise_amp*np.random.randn()
        dI2 = (-I2 + I_syn_I2)/self.tau_i2 + self.noise_amp*np.random.randn()
        return np.array([dE, dI1, dI2])

    def simulate(self, duration_sec=DURATION, seed=42, n_runs=3):
        np.random.seed(seed)
        steps = int(duration_sec*self.fs)
        all_eeg = []
        for run in range(n_runs):
            state = np.array([0.1, 0.0, 0.0])
            eeg = np.zeros(steps)
            for i in range(steps):
                eeg[i] = state[0]
                state = state + self.dt * self.derivs(state, i*self.dt) + np.sqrt(self.dt) * self.noise_amp * np.random.randn(3)
            all_eeg.append(eeg)
            np.random.seed(seed + run + 1)
        return np.mean(all_eeg, axis=0)

# =============================================================================
# 3. FUNCIONES DE ANÁLISIS
# =============================================================================
def spectrogram_db(signal):
    f, t, Sxx = spectrogram(signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    return 10*np.log10(Sxx+1e-10), f, t

def band_power_mean(spec, f):
    pow_dict = {}
    for band, (fmin, fmax) in BANDS.items():
        idx = (f>=fmin) & (f<fmax)
        pow_dict[band] = np.mean(spec[idx, :]) if np.any(idx) else 0.0
    return pow_dict

def distance_spectral(pow_a, pow_b):
    return sum(abs(pow_a.get(b,0.0) - pow_b.get(b,0.0)) for b in BANDS)

def compute_phase_metrics(signals, fs=FS):
    if signals.ndim == 1:
        signals = signals.reshape(1,-1)
    n_ch, n_samp = signals.shape
    if n_ch < 2:
        return 0.0, 0.0
    b, a = butter(4, [0.5,45], btype='band', fs=fs)
    filtered = filtfilt(b, a, signals, axis=1)
    phases = np.zeros_like(filtered)
    for ch in range(n_ch):
        phases[ch,:] = np.angle(hilbert(filtered[ch,:]))
    plv_sum = 0.0; n_pairs = 0
    for i in range(n_ch):
        for j in range(i+1,n_ch):
            plv_sum += np.abs(np.mean(np.exp(1j*(phases[i,:]-phases[j,:]))))
            n_pairs += 1
    plv_avg = plv_sum/n_pairs if n_pairs>0 else 0.0
    r_avg = np.mean(np.abs(np.mean(np.exp(1j*phases), axis=0)))
    return plv_avg, r_avg

def compute_combined_distance(eeg_a, eeg_b, weight_phase=0.5):
    if eeg_a.ndim==1: eeg_a = eeg_a.reshape(1,-1)
    if eeg_b.ndim==1: eeg_b = eeg_b.reshape(1,-1)
    sig_a = np.mean(eeg_a, axis=0); sig_b = np.mean(eeg_b, axis=0)
    spec_a, f, _ = spectrogram_db(sig_a)
    spec_b, _, _ = spectrogram_db(sig_b)
    pow_a = band_power_mean(spec_a, f); pow_b = band_power_mean(spec_b, f)
    d_esp = distance_spectral(pow_a, pow_b)
    plv_a, r_a = compute_phase_metrics(eeg_a, FS)
    plv_b, r_b = compute_phase_metrics(eeg_b, FS)
    d_phase = abs(plv_a-plv_b) + abs(r_a-r_b)
    return d_esp + weight_phase*d_phase, d_esp, d_phase

# =============================================================================
# 4. GENERACIÓN DE EEG
# =============================================================================
def generate_eeg_from_params(params_dict, seed=42, n_runs=3):
    model = NeuralMass(**params_dict, fs=FS)
    return model.simulate(seed=seed, n_runs=n_runs)

def generate_patient_pair(level_key):
    level = LEVELS[level_key]
    td_eeg = generate_eeg_from_params(level['td'], seed=42, n_runs=5)
    asd_eeg = generate_eeg_from_params(level['asd'], seed=123, n_runs=5)
    return td_eeg, asd_eeg

# =============================================================================
# 5. EVALUACIÓN DE MODULACIÓN
# =============================================================================
def evaluate_modulation(base_params, deltas, td_eeg, weight_phase=0.5):
    model_mod = NeuralMass(
        g_nmda_ei=base_params['g_nmda_ei'],
        g_nmda_ee=base_params['g_nmda_ee'],
        low_freq_amp=base_params['low_freq_amp'],
        g_i_delta=deltas[0], g_e_delta=deltas[1],
        p_base_delta=deltas[2], g_nmda_ee_delta=deltas[3],
        gaba_b_delta=deltas[4], nmda_nr2b_delta=deltas[5],
        dopamine_mod=deltas[6], serotonin_mod=deltas[7],
        norepinephrine_mod=deltas[8], fs=FS
    )
    eeg_mod = model_mod.simulate(seed=456, n_runs=3)
    model_base = NeuralMass(**base_params, fs=FS)
    eeg_base = model_base.simulate(seed=42, n_runs=3)
    d_total_base, _, _ = compute_combined_distance(eeg_base, td_eeg, weight_phase)
    d_total_mod, _, _ = compute_combined_distance(eeg_mod, td_eeg, weight_phase)
    improvement = (d_total_base - d_total_mod)/d_total_base*100 if d_total_base>0 else 0.0
    return {
        'improvement': improvement,
        'distance_base': d_total_base,
        'distance_mod': d_total_mod,
        'eeg_base': eeg_base,
        'eeg_mod': eeg_mod,
        'deltas': deltas
    }

# =============================================================================
# 6. RANKING LOCAL
# =============================================================================
RANKING_FILE = 'synapsit_ranking.json'
CONTRIBUTIONS_FILE = 'synapsit_contributions.json'

def load_ranking():
    if os.path.exists(RANKING_FILE):
        with open(RANKING_FILE, 'r') as f:
            return json.load(f)
    return []

def save_ranking(ranking):
    with open(RANKING_FILE, 'w') as f:
        json.dump(ranking, f, indent=2)

def add_score_to_ranking(player_name, level_key, improvement, deltas):
    ranking = load_ranking()
    entry = {
        'player': player_name,
        'level': level_key,
        'level_name': LEVELS[level_key]['name'],
        'score': round(improvement, 2),
        'deltas': [round(d, 3) for d in deltas],
        'timestamp': datetime.now().isoformat()
    }
    ranking.append(entry)
    ranking.sort(key=lambda x: x['score'], reverse=True)
    save_ranking(ranking)
    return ranking

def load_contributions():
    if os.path.exists(CONTRIBUTIONS_FILE):
        with open(CONTRIBUTIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_contribution(entry):
    contribs = load_contributions()
    contribs.append(entry)
    with open(CONTRIBUTIONS_FILE, 'w') as f:
        json.dump(contribs, f, indent=2)

# =============================================================================
# 7. COMUNICACIÓN CON GOOGLE APPS SCRIPT
# =============================================================================
URL_APPS_SCRIPT = "https://script.google.com/macros/s/AKfycbzmO9LUuhj8i1tVlTrvNYfv-HLzKPZI1vcCL1tjuzaApXLQpr0vTiL8sx-WYDIQThOB/exec"  # <-- REEMPLAZA

def enviar_a_apps_script(payload):
    try:
        response = requests.post(URL_APPS_SCRIPT, json=payload, timeout=15)
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('status') == 'ok':
                    return True, "OK"
                else:
                    return False, f"Error del servidor: {data.get('message')}"
            except:
                return False, f"Respuesta no JSON: {response.text[:100]}"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, str(e)

def cargar_ranking_desde_drive():
    """Obtiene el ranking desde Google Drive y actualiza el archivo local."""
    try:
        response = requests.get(URL_APPS_SCRIPT + "?action=get", timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                rows = data.get('data', [])
                new_ranking = []
                for row in rows:
                    try:
                        score_val = row.get('Puntuacion', 0)
                        if isinstance(score_val, str):
                            score_val = float(score_val)
                        new_ranking.append({
                            'player': row.get('Jugador', 'Anónimo'),
                            'level': row.get('Nivel', ''),
                            'level_name': row.get('NombrePaciente', ''),
                            'score': score_val,
                            'deltas': json.loads(row.get('Deltas', '[]')),
                            'timestamp': row.get('Timestamp', datetime.now().isoformat())
                        })
                    except Exception as e:
                        print(f"Error al procesar fila: {e} - Fila: {row}")
                        continue
                new_ranking.sort(key=lambda x: x['score'], reverse=True)
                save_ranking(new_ranking)
                return True, new_ranking
            else:
                return False, f"Error en respuesta: {data.get('message')}"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

# =============================================================================
# 8. INTERFAZ DE USUARIO (UI)
# =============================================================================
class SynapsitGame:
    def __init__(self):
        # Inicializar atributos
        self.current_level = list(LEVELS.keys())[0]
        self.td_eeg = None
        self.asd_eeg = None
        self.asd_params = None
        self.weight_phase = 0.5
        self.current_score = 0.0
        self.best_score = -float('inf')
        self.best_deltas = None
        self.player_name = "Jugador"
        self.sliders = {}  # <-- Inicializar aquí

        self._build_ui()
        self.load_level(self.current_level)
        self._display_ui()

    def _build_ui(self):
        self.slider_boxes = []

        for name in PARAM_NAMES:
            min_val, max_val = PARAM_RANGES[name]
            default = PARAM_DEFAULTS[name]

            slider = widgets.FloatSlider(
                value=default, min=min_val, max=max_val, step=0.01,
                description=name,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='300px')
            )
            slider.observe(self.on_param_change, 'value')
            self.sliders[name] = slider

            definition_html = DEFINICIONES.get(name, "Definición no disponible.")
            details = widgets.HTML(value=f"""
            <details style="margin-left: 10px; margin-bottom: 10px; background-color: #2c3e50; padding: 5px; border-radius: 5px; border: 1px solid #34495e;">
                <summary style="font-weight: bold; cursor: pointer; color: #ecf0f1; font-size: 0.85em;">
                    📖 Ver definición de {name}
                </summary>
                <div style="padding: 10px; background-color: #1a252f; border-radius: 4px; border: 1px solid #34495e; margin-top: 5px; font-size: 0.9em; line-height: 1.5; color: #ecf0f1;">
                    {definition_html}
                </div>
            </details>
            """)

            box = widgets.VBox([slider, details])
            self.slider_boxes.append(box)

        # Botones
        self.btn_reset = widgets.Button(description='🔄 Resetear Perillas', button_style='warning', layout=widgets.Layout(width='150px'))
        self.btn_reset.on_click(self.reset_sliders)

        self.btn_submit = widgets.Button(description='🏆 Enviar Puntuación', button_style='success', layout=widgets.Layout(width='150px'))
        self.btn_submit.on_click(self.submit_score)

        self.btn_sync = widgets.Button(description='🔄 Sincronizar con Drive', button_style='info', layout=widgets.Layout(width='180px'))
        self.btn_sync.on_click(self.sincronizar_ranking)

        self.btn_next = widgets.Button(description='➡️ Siguiente Paciente', button_style='primary', layout=widgets.Layout(width='180px'))
        self.btn_next.on_click(self.next_level)

        self.level_selector = widgets.Dropdown(
            options=[(f"{k}: {v['name']}", k) for k, v in LEVELS.items()],
            value=self.current_level,
            description='Paciente:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.level_selector.observe(self.on_level_change, 'value')

        self.name_input = widgets.Text(
            value='Jugador',
            description='Nombre:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.name_input.observe(self.on_name_change, 'value')

        self.score_display = widgets.HTML(
            value="<h2 style='color: #2c3e50;'>Puntuación: <span style='color: #27ae60;'>0.0%</span></h2>"
        )
        self.best_display = widgets.HTML(
            value="<h4>Mejor: <span style='color: #2980b9;'>-</span></h4>"
        )
        self.level_display = widgets.HTML(
            value="<h3>Paciente: <span style='color: #8e44ad;'>Cargando...</span></h3>"
        )

        self.graph_output = widgets.Output()
        self.ranking_output = widgets.Output()

        self.progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Modulando...',
            bar_style='info',
            layout=widgets.Layout(width='100%')
        )
        self.status_text = widgets.HTML(value="<i>Listo para jugar.</i>")

    def _display_ui(self):
        control_panel = widgets.VBox([
            widgets.HTML("<h2>🎛️ Panel de Control</h2>"),
            self.level_display,
            self.level_selector,
            widgets.HTML("<hr>"),
            widgets.HTML("<b>🧬 Receptores y Neuromoduladores</b>"),
            widgets.HTML("<i style='font-size:0.85em; color:#6c757d;'>Haz clic en <b>📖 Ver definición</b> debajo de cada perilla para aprender más.</i>"),
            widgets.HTML("<br>"),
            *self.slider_boxes,
            widgets.HTML("<hr>"),
            widgets.HBox([self.btn_reset, self.btn_submit, self.btn_sync, self.btn_next]),
            widgets.HTML("<hr>"),
            widgets.HBox([self.name_input, self.progress]),
            self.status_text
        ], layout=widgets.Layout(width='420px', padding='10px'))

        view_panel = widgets.VBox([
            widgets.HTML("<h2>📊 Visor del Paciente</h2>"),
            self.score_display,
            self.best_display,
            self.graph_output,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>🏅 Ranking Local</h3>"),
            self.ranking_output
        ], layout=widgets.Layout(width='60%', padding='10px'))

        main_panel = widgets.HBox([control_panel, view_panel])

        display(HTML("""
        <style>
            .widget-label { font-weight: bold; }
            .jupyter-widgets { margin: 2px; }
            .ranking-table {
                background-color: #2c3e50;
                color: white;
                border-radius: 8px;
                overflow: hidden;
            }
            .ranking-table th {
                background-color: #1a252f;
                color: #ecf0f1;
                padding: 8px;
            }
            .ranking-table td {
                padding: 8px;
                border-bottom: 1px solid #34495e;
            }
            .ranking-table tr:nth-child(even) {
                background-color: #34495e;
            }
            .ranking-table tr:nth-child(odd) {
                background-color: #2c3e50;
            }
            .ranking-table tr:hover {
                background-color: #3d566e;
            }
            details summary {
                outline: none;
            }
            details summary:hover {
                color: #3498db;
            }
            details[open] summary {
                color: #5dade2;
                border-bottom: 1px solid #34495e;
                padding-bottom: 5px;
            }
        </style>
        """))
        display(main_panel)

    # =========================================================================
    # LÓGICA DEL JUEGO
    # =========================================================================
    def load_level(self, level_key):
        self.current_level = level_key
        self.td_eeg, self.asd_eeg = generate_patient_pair(level_key)
        self.asd_params = LEVELS[level_key]['asd'].copy()
        self.current_score = 0.0
        self.best_score = -float('inf')
        self.best_deltas = None
        self.update_display()
        self.update_ranking()
        self.status_text.value = f"<i>Paciente '{LEVELS[level_key]['name']}' cargado. ¡A modular!</i>"
        self.on_param_change(None)

    def on_level_change(self, change):
        self.load_level(change['new'])

    def on_name_change(self, change):
        self.player_name = change['new'] if change['new'] else "Jugador"

    def reset_sliders(self, b=None):
        for name, slider in self.sliders.items():
            slider.value = PARAM_DEFAULTS[name]
        self.status_text.value = "<i>Perillas reseteadas.</i>"

    def on_param_change(self, change):
        deltas = self._get_current_deltas()
        if self.asd_params is not None and self.td_eeg is not None:
            result = evaluate_modulation(self.asd_params, deltas, self.td_eeg, self.weight_phase)
            self.current_score = result['improvement']
            if self.current_score > self.best_score:
                self.best_score = self.current_score
                self.best_deltas = deltas.copy()
            self.update_display(result)
            self.status_text.value = f"<i>Modulando... Mejora: {self.current_score:.1f}%</i>"

    def _get_current_deltas(self):
        return [self.sliders[name].value for name in PARAM_NAMES]

    def update_display(self, result=None):
        score_color = '#27ae60' if self.current_score >= 0 else '#e74c3c'
        self.score_display.value = f"<h2 style='color: #2c3e50;'>Puntuación: <span style='color: {score_color};'>{self.current_score:.1f}%</span></h2>"
        best_text = f"{self.best_score:.1f}%" if self.best_score > -float('inf') else "-"
        self.best_display.value = f"<h4>Mejor: <span style='color: #2980b9;'>{best_text}</span></h4>"
        level_name = LEVELS[self.current_level]['name']
        self.level_display.value = f"<h3>Paciente: <span style='color: #8e44ad;'>{level_name}</span></h3>"
        with self.graph_output:
            clear_output(wait=True)
            if result is not None:
                self._plot_results(result)
            else:
                if self.asd_params is not None and self.td_eeg is not None:
                    dummy_deltas = [0.0] * 9
                    dummy_result = evaluate_modulation(self.asd_params, dummy_deltas, self.td_eeg, self.weight_phase)
                    self._plot_results(dummy_result)

    def _plot_results(self, result):
        fig, ax = plt.subplots(figsize=(8, 4))
        eeg_base = result['eeg_base']; eeg_mod = result['eeg_mod']; eeg_td = self.td_eeg
        sig_base = np.mean(eeg_base) if eeg_base.ndim > 1 else eeg_base
        sig_mod = np.mean(eeg_mod) if eeg_mod.ndim > 1 else eeg_mod
        sig_td = np.mean(eeg_td) if eeg_td.ndim > 1 else eeg_td
        spec_base, f, _ = spectrogram_db(sig_base)
        spec_mod, _, _ = spectrogram_db(sig_mod)
        spec_td, _, _ = spectrogram_db(sig_td)
        ax.plot(f, np.mean(spec_base, axis=1), label='Problema (ASD)', linestyle='--', alpha=0.7, color='#e67e22')
        ax.plot(f, np.mean(spec_mod, axis=1), label=f'Modulado (mejora: {result["improvement"]:.1f}%)', linewidth=2, color='#27ae60')
        ax.plot(f, np.mean(spec_td, axis=1), label='Objetivo (TD)', linestyle=':', alpha=0.7, color='#2980b9')
        ax.set_xlabel('Frecuencia (Hz)'); ax.set_ylabel('Potencia (dB)')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title(f'Espectro EEG - Paciente: {LEVELS[self.current_level]["name"]}')
        plt.tight_layout()
        plt.show()

    def submit_score(self, b=None):
        if self.current_score <= 0:
            self.status_text.value = "<i style='color: #e74c3c;'>⚠️ La puntuación debe ser positiva para enviarla.</i>"
            return

        ranking = add_score_to_ranking(
            self.player_name,
            self.current_level,
            self.current_score,
            self._get_current_deltas()
        )
        self.status_text.value = f"<i style='color: #27ae60;'>✅ Guardado localmente. Mejora: {self.current_score:.1f}%</i>"

        contribution = {
            'player': self.player_name,
            'level': self.current_level,
            'level_name': LEVELS[self.current_level]['name'],
            'asd_params': self.asd_params,
            'deltas': self._get_current_deltas(),
            'score': self.current_score,
            'timestamp': datetime.now().isoformat()
        }
        save_contribution(contribution)

        payload = {
            'player': contribution['player'],
            'level': contribution['level'],
            'level_name': contribution['level_name'],
            'score': round(contribution['score'], 2),
            'deltas': [round(d, 3) for d in contribution['deltas']],
            'asd_params': contribution['asd_params'],
            'timestamp': contribution['timestamp']
        }
        ok, mensaje = enviar_a_apps_script(payload)
        if ok:
            self.status_text.value = f"<i style='color: #27ae60;'>✅ ¡Enviado a Drive! ({self.current_score:.1f}%)</i>"
        else:
            self.status_text.value = f"<i style='color: #e67e22;'>⚠️ Guardado localmente (Drive: {mensaje})</i>"

        self.update_ranking()

    def sincronizar_ranking(self, b=None):
        self.status_text.value = "<i>🔄 Sincronizando con Drive...</i>"
        ok, resultado = cargar_ranking_desde_drive()
        if ok:
            self.status_text.value = f"<i style='color: #27ae60;'>✅ Ranking sincronizado ({len(resultado)} registros)</i>"
        else:
            self.status_text.value = f"<i style='color: #e74c3c;'>❌ Error al sincronizar: {resultado}</i>"
        self.update_ranking()

    def next_level(self, b=None):
        keys = list(LEVELS.keys())
        current_idx = keys.index(self.current_level)
        next_idx = (current_idx + 1) % len(keys)
        self.level_selector.value = keys[next_idx]
        self.status_text.value = f"<i>Avanzando al paciente '{LEVELS[keys[next_idx]]['name']}'...</i>"

    def update_ranking(self):
        with self.ranking_output:
            clear_output(wait=True)
            ranking = load_ranking()
            if not ranking:
                display(HTML("<i style='color: #ecf0f1;'>No hay puntuaciones aún. ¡Sé el primero!</i>"))
                return

            html = """<table class="ranking-table" style="width:100%; border-collapse: collapse; background-color: #2c3e50; color: white; border-radius: 8px; overflow: hidden;">
            <thead>
                <tr style="background-color: #1a252f; color: #ecf0f1;">
                    <th style="padding: 8px; text-align: left;">#</th>
                    <th style="padding: 8px; text-align: left;">Jugador</th>
                    <th style="padding: 8px; text-align: left;">Paciente</th>
                    <th style="padding: 8px; text-align: left;">Puntuación</th>
                </tr>
            </thead>
            <tbody>"""
            for i, entry in enumerate(ranking[:10]):
                medal = '🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}.'
                bg_color = '#34495e' if i % 2 == 0 else '#2c3e50'
                html += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 8px;">{medal}</td>
                    <td style="padding: 8px;">{entry['player']}</td>
                    <td style="padding: 8px;">{entry['level_name']}</td>
                    <td style="padding: 8px;"><b>{entry['score']:.1f}%</b></td>
                </tr>"""
            html += "</tbody></table>"
            display(HTML(html))

# =============================================================================
# 9. INICIALIZACIÓN (para Voilà)
# =============================================================================
display(HTML("""
<div style="background: linear-gradient(135deg, #2c3e50, #3498db); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
    <h1 style="margin: 0;">🧠 SYNAPSIT</h1>
    <p style="margin: 5px; font-size: 1.2em;">El juego de ciencia ciudadana para personalizar la neuromodulación</p>
    <p style="margin: 5px; font-size: 0.9em; opacity: 0.8;">Ajusta las perillas para "mejorar" el EEG de cada paciente. ¡Encuentra la mejor combinación!</p>
</div>
"""))

game = SynapsitGame()

display(HTML("""
<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px; text-align: center; border: 1px solid #dee2e6;">
    <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
        💡 Haz clic en "📖 Ver definición" debajo de cada perilla para aprender sobre receptores y neuromoduladores.
    </p>
</div>
"""))