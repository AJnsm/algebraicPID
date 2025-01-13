import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm

def mutInf(prob_dict, vars1, vars2):
    '''
    prob_dict is a dict that maps strings to floats
    vars1 and vars2 are lists of integer indices
    '''
    mi=0

    comb_marginal = {}
    for state, prob in prob_dict.items():
        state_comb = ''.join([state[i] for i in list(vars1)+list(vars2)])
        comb_marginal[state_comb] = comb_marginal.get(state_comb, 0) + prob
    
    marginal1 = {}
    marginal2 = {}
    for state, prob in comb_marginal.items():
        state1 = ''.join(state[:len(vars1)])
        state2 = ''.join(state[len(vars1):])
        marginal1[state1] = marginal1.get(state1, 0) + prob
        marginal2[state2] = marginal2.get(state2, 0) + prob

    for state, prob in comb_marginal.items():
        state1 = ''.join(state[:len(vars1)])
        state2 = ''.join(state[len(vars1):])
        if prob == 0:
            continue
        mi += prob * np.log2(prob/(marginal1[state1]*marginal2[state2]))
    return mi

# Helper functions:
def Immi_ditless(d, sources, targets):
    return min([mutInf(d, source, target) for source, target in product(sources, targets)])

def acStringToTuple(ac):
    return tuple([tuple([int(x) for x in block]) for block in ac.split('|')])

def acTupleToString(ac):
    return '|'.join([''.join([str(x) for x in block]) for block in ac])

def acTupleToVoice(ac):
        return tuple(tuple(voices[i] for i in block) for block in ac)

def compute_mobius_inversion(redsDict,  mfMat):
    atoms, reds = list(zip(redsDict.items()))
    return dict(list(zip(atoms, np.dot(reds, mfMat))))

def constructDist(data, delay):
    pairedData = np.array([np.hstack([data[i], data[i+delay]]) for i in range(len(data)-delay)])
    states, counts = np.unique(pairedData.astype(str), axis=0, return_counts=True)
    pmf = {''.join(state): count/len(pairedData) for state, count in zip(states, counts)}
    return pmf

def calc_immi_ditless(args):
    d, ac1, ac2 = args
    return acTupleToString(ac1), acTupleToString(ac2), Immi_ditless(d, ac1, ac2)


if __name__ == '__main__':
    N = 4
    for shuffled in [False, True]:
        for composer in ['bach', 'corelli']:
            for mood in ['major']:
                for delay in [1]:
                    print(f'Computing reds for {mood} {composer} with delay {delay}')

                    mfMat = pd.read_csv(f'FMT_outputs/antiChainLattice_mobiusFns_N={N}.csv', index_col=0)
                    acs_t = [acStringToTuple(x) for x in (mfMat.columns)]
                    acs_tp1 = [tuple(tuple(e+N for e in a) for a in ac) for ac in acs_t]

                    if shuffled:
                        music = pd.read_csv(f'musicData/{composer}_{mood}_shuffled.csv', index_col=0)
                    else:
                        music = pd.read_csv(f'musicData/{composer}_{mood}.csv', index_col=0)

                    # Keep only the chords that are different from the previous one
                    music = music.loc[(music.shift() != music).any(axis=1)]

                    d = constructDist(music.values[:, :N], delay)
                    voices = ['S', 'A', 'T', 'B', 's', 't', 'a', 'b']

                    mu_pid = mfMat.values
                    mu_phi = np.kron(mu_pid, mu_pid)

                    pool = Pool()
                    reds = []
                    total_iterations = len(list(product(acs_t, acs_tp1)))


                    with tqdm(total=total_iterations, desc='Progress') as pbar:
                        for result in pool.imap(calc_immi_ditless, [(d, ac1, ac2) for ac1, ac2 in product(acs_t, acs_tp1)]):
                            reds.append(result)
                            pbar.update(1)

                    pool.close()
                    pool.join()

                    reds_dict = {ac1+','+ac2: value for (ac1, ac2, value) in reds}

                    # Save the dictionary as a CSV file
                    reds_df = pd.DataFrame.from_dict(reds_dict, orient='index', columns=['value'])
                    if shuffled:
                        reds_df.to_csv(f'musicData/reds_{composer}_{mood}_N{N}_delay{delay}_shuffled.csv')
                    else:
                        reds_df.to_csv(f'musicData/reds_{composer}_{mood}_N{N}_delay{delay}.csv')