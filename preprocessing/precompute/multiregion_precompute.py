# This function is designed for separating trials, bouts and actions from the whole dataset.
# It will label different trials and automatically cut out activity bouts versus decision bouts.
# 
# [INPUT]   datafile_path.xlsx [required]
#           frame_rate [optional]
#
# [OUTPUT]  a struct contains:
#                NeuralData
#                ├── num_trial
#                ├── trials (list)
#                │     ├── TrialData
#                │           ├── drink (list of BoutAct)
#                │           ├── press
#                │           ├── delay
#                │           ├── decision
#                │
#                └── col_name
#
#
# [CALL]
# 
#           paths = [
#               "full_path",
#               "full_path",
#               ...]
#
#           labels_session = ['dataset_name1', 'dataset_name2', '...']
#           save_dir = "your_save_path"
#           os.makedirs(save_dir, exist_ok=True)
#            
#           framerate = 30 
#           datasets = [multiregion_precompute(p, framerate) for p in paths]
#
#
#
#

def multiregion_precompute(datafile_path, fr=30):
    import numpy as np
    import pandas as pd
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class BoutAct:
        caltrace: np.ndarray
        start: int
        end: int

    @dataclass
    class TrialData:
        drink: Optional[BoutAct]
        press: Optional[BoutAct]
        delay: Optional[BoutAct]
        decision: Optional[BoutAct]

    @dataclass
    class NeuralData:
        num_trial: int
        trials: list
        col_name: list

    # =========================
    # load datasets
    # =========================
    data = pd.read_excel(datafile_path)
    data.columns = data.columns.str.strip().str.lower()

    press_bout = pd.to_numeric(data['press bout'], errors='coerce').fillna(0).astype(int).values
    drink_bout = pd.to_numeric(data['drink bout'], errors='coerce').fillna(0).astype(int).values

    col_name = data.columns[5:].tolist()
    cal = data.iloc[:, 5:].values.T

    T = cal.shape[1]

    # =========================
    # detection: bouts
    # =========================
    def detect_bout(signal):
        start = np.where((signal[1:] - signal[:-1]) == 1)[0]
        end   = np.where((signal[1:] - signal[:-1]) == -1)[0]

        # add "start" when series started by "1"
        if signal[0] == 1:
            start = np.insert(start, 0, 0)

        # add "end" when series ended by "1"
        if signal[-1] == 1:
            end = np.append(end, len(signal)-1)

        # length alignment
        n = min(len(start), len(end))
        start = start[:n]
        end   = end[:n]

        return start, end

    press_start, press_end = detect_bout(press_bout)
    drink_start, drink_end = detect_bout(drink_bout)

    # =========================
    # construct trial
    # =========================
    trials = []
    n_press = len(press_start)
    drink_ptr = 0

    
    pending_decision = None
    for i in range(n_press):

        ps = press_start[i]
        pe = press_end[i]

        # ===== decision section for this trial =====
        decision_bout_obj = pending_decision
        pending_decision = None  # delete after used

        # ===== decide: press =====
        s = max(ps - 4*fr, 0)
        e = min(ps + 4*fr, T-1)
        press_bout_obj = BoutAct(cal[:, s:e], s, e)

        drink_bout_obj = None
        delay_bout_obj = None

        # ===== decide: drink =====
        while drink_ptr < len(drink_start) and drink_start[drink_ptr] <= pe:
            drink_ptr += 1

        has_drink = False

        if drink_ptr < len(drink_start):
            ds = drink_start[drink_ptr]
            de = drink_end[drink_ptr]

            next_ps = press_start[i+1] if i+1 < n_press else T

            if pe < ds < next_ps:
                has_drink = True

        # ===== case A：have drink session =====
        if has_drink:

            margin = int(1 * fr)
            s_delay = pe + margin
            e_delay = ds - margin

            if e_delay > s_delay:
                delay_bout_obj = BoutAct(cal[:, s_delay:e_delay], s_delay, e_delay)

            s = max(ds - 4*fr, 0)
            e = min(ds + 4*fr, T-1)
            drink_bout_obj = BoutAct(cal[:, s:e], s, e)

            # generate decision section for next trial session
            s_dec = pe
            e_dec = de

            if e_dec > s_dec:
                pending_decision = BoutAct(cal[:, s_dec:e_dec], s_dec, e_dec)

            drink_ptr += 1

        # ===== case B：no existing drink session =====
        else:
            if i + 1 < n_press:
                next_ps = press_start[i+1]

                s_dec = pe
                e_dec = next_ps

                if e_dec > s_dec:
                    pending_decision = BoutAct(cal[:, s_dec:e_dec], s_dec, e_dec)

        # construct class: trial
        trial_data = TrialData(
            decision=decision_bout_obj,
            press=press_bout_obj,
            delay=delay_bout_obj,
            drink=drink_bout_obj
        )

        trials.append(trial_data)

    return NeuralData(
        num_trial=len(trials),
        trials=trials,
        col_name=col_name
    )
