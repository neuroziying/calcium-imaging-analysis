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


def multiregion_precompute(datafile_path, fr=30):
    import numpy as np
    import pandas as pd
    from dataclasses import dataclass

    @dataclass
    class BoutAct:
        caltrace: np.ndarray
        start: int
        end: int

    @dataclass
    class TrialData:
        drink: list
        press: list
        delay: list
        decision: list

    @dataclass
    class NeuralData:
        num_trial: int
        trials: list
        col_name: list

    # load data 
    data = pd.read_excel(datafile_path)
    
    trial = data['trial'].values
    press_bout = data['press bout'].values
    drink_bout = data['drink bout'].values
    data.columns = data.columns.str.strip().str.lower()
    col_name = data.columns[5:].tolist()

    cal = data.iloc[:, 5:].values.T   # col 6 ~ col 20 in this exp.

    # trial segmentation: if index changes, cut into pieces.
    change = trial[1:] != trial[:-1]
    change = np.insert(change, 0, True)
    label = np.cumsum(change)

    trials = []

    for t in np.unique(label):
        idx = np.where(label == t)[0]

        cal_t = cal[:, idx]
        press_t = press_bout[idx]
        drink_t = drink_bout[idx]

        # detect bouts
        press_start = np.where((press_t[1:] - press_t[:-1]) == 1)[0]
        press_end   = np.where((press_t[1:] - press_t[:-1]) == -1)[0]

        drink_start = np.where((drink_t[1:] - drink_t[:-1]) == 1)[0]
        drink_end   = np.where((drink_t[1:] - drink_t[:-1]) == -1)[0]

        press_list = []
        for s, e in zip(press_start, press_end):
            s = max(s - 4*fr, 0)
            e = min(s + 4*fr, cal_t.shape[1]-1)
            press_list.append(BoutAct(
                caltrace=cal_t[:, s:e],
                start=s,
                end=e))

        drink_list = []
        for s, e in zip(drink_start, drink_end):
            s = max(s - 4*fr, 0)
            e = min(s + 4*fr, cal_t.shape[1]-1)
            drink_list.append(BoutAct(
                caltrace=cal_t[:, s:e],
                start=s,
                end=e))

        # delay: press_end → drink_start
        delay_list = []
        for pe, ds in zip(press_end, drink_start):
            if ds > pe:
                # prevent data pollution from events
                margin = int(1 * fr)

                s = pe + margin
                e = ds - margin

                if e > s:
                    delay_list.append(
                        BoutAct(
                            caltrace=cal_t[:, s:e],
                            start=s,
                            end=e
                        )
                    )

        # decision: drink_end → next press_start
        decision_list = []

        n_drink = len(drink_end)

        for i in range(n_drink):

            de = drink_end[i]

            # case: press again
            if i + 1 < len(press_start):
                ps = press_start[i + 1]

                if ps > de:
                    s = de + int(1 * fr)
                    e = ps - int(1 * fr)

                    if e > s:
                        decision_list.append(
                            BoutAct(
                                caltrace=cal_t[:, s:e],
                                start=s,
                                end=e
                            )
                        )

            # case: lick -> end of the trial
            else:
                s = de + int(1 * fr)
                e = cal_t.shape[1]

                if e > s:
                    decision_list.append(
                        BoutAct(
                            caltrace=cal_t[:, s:e],
                            start=s,
                            end=e
                        )
                    )

        # build trial 
        trial_data = TrialData(
            drink=drink_list,
            press=press_list,
            delay=delay_list,
            decision=decision_list
        )

        trials.append(trial_data)

    return NeuralData(
        num_trial=len(trials),
        trials=trials,
        col_name = col_name
    )