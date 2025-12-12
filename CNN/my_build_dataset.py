"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np
from tqdm.auto import tqdm
import os
import random
import scipy
from npy_append_array import NpyAppendArray

import h5py

np.random.seed(1234)


def highpass(traces: np.ndarray,
             Wn: float = 0.001) -> np.ndarray:
    b, a = scipy.signal.butter(3, Wn)
    y = scipy.signal.filtfilt(b, a, traces).astype(np.float32)

    return (traces - y).astype(np.float32)

######################################################################################################################################
def debug(all_traces, keys, pts):
    #rounds = []
    #print(f"Keys: {keys}")
    key_broadcast = [keys[0]] * len(pts) # Create a copy of the key for every pt
    #print(f"Keys Broadcast: {keys}")
    print(f"Number of Plaintexts: {len(pts)}")
    # Build metadata as a 2-column OBJECT array
    N = len(pts)
    rounds_metadata = np.empty((N, 2), dtype=object)
    for i, trace in enumerate(all_traces):
        #rounds.append(trace[:])
        trace_max_len = len(trace)
        print(f"Length of Trace: {trace_max_len}")
        #rounds.append(trace[:])
        # Append metadata for this trace:
        #   - plaintext for trace i
        #   - corresponding key
        rounds_metadata[i, 0] = pts[i]              # plaintext array (16 bytes)
        rounds_metadata[i, 1] = key_broadcast[i]    # key array (16 bytes)

        
    
    #Key is unique per trace so we don't care
    
    datasets = {"rounds_md": rounds_metadata}
    for name, data in datasets.items():
        print(f"{name}:")
        print("  type:", type(data))
        print("  shape:", data.shape)
        print("  dtype:", data.dtype)
        #print("  min/max:", data.min(), "/", data.max())
        print()
    #print("Metadata:", rounds_metadata)
    
    return rounds_metadata

######################################################################################################################################

def _writeConfig(dataset_folder,
                 train_shape,
                 valid_shape,
                 test_shape,
                 N_WIN_FIRST,
                 #N_WIN_SECOND,
                 #N_WIN_OTHER,
                 #N_WIN_NOISE,
                 filter):

    with open(dataset_folder + "/config.txt", "w") as file:
        file.write(f"N. training traces: {train_shape[0]}\n")
        file.write(f"N. validation traces: {valid_shape[0]}\n")
        file.write(f"N. test traces: {test_shape[0]}\n")
        file.write(f"N. input samples: {train_shape[1]}\n")
        file.write(f"Filter: {filter}\n")
        file.write(f"N. AES round windows: {N_WIN_FIRST} - class: 1\n")
        #file.write(f"N. second round windows: {N_WIN_SECOND} - class: 0\n")
        #file.write(f"N. other round windows: {N_WIN_OTHER} - class: 0\n")
        #file.write(f"N. noise windows: {N_WIN_NOISE} - class: 2\n")


def _getFilePathes(traces_folder):
    traces_files = np.sort(
        [traces_folder + f for f in os.listdir(traces_folder) if f.endswith('.h5')]).tolist()

    assert len(
        traces_files) == 16, f"Expected 16 traces files, found {len(traces_files)}"

    # keep last file for testing
    traces_files = traces_files[:-1]

    return traces_files

def _getOrderRounds(all_traces, keys, pts, n_win, win_width, start=0):
    rounds = []
    rounds_pts = []
    rounds_keys = []
    #all_traces = random.sample(all_traces, n_win)
    all_traces = all_traces[:n_win] #First n_win traces to keep correlation with Key/plaintext data
    pts = pts[:n_win]   #First n_win Plaintexts
    #Key is unique per trace so we don't care
    for trace in all_traces:
        rounds.append(trace[start:start+win_width])
    return rounds

############################################################################################################################################################
def _getAESRounds(all_traces, keys, pts):
    #min_len = min(len(trace) for trace in all_traces)
    rounds = []
    #rounds_metadata = []
    key_broadcast = [keys[0]] * len(pts) # Create a copy of the key for every pt
    # Build metadata as a 2-column OBJECT array
    N = len(pts)
    rounds_metadata = np.empty((N, 2), dtype=object)    
    
    #min(len(all_traces)) + cut all_traces to minimum length

    #Key is unique per trace so we don't care
    for i, trace in enumerate(all_traces):
        rounds.append(trace[:min_len]) #add cut to desired length, so all traces can be equal
        # Append metadata for this trace:
        #   - plaintext for trace i
        #   - corresponding key
        rounds_metadata[i, 0] = pts[i]              # plaintext array (16 bytes)
        rounds_metadata[i, 1] = key_broadcast[i]    # key array (16 bytes)
    return rounds, rounds_metadata
############################################################################################################################################################
def _getRandomRounds(all_traces, n_win, win_width, start=0):
    rounds = []
    #all_traces = random.sample(all_traces, n_win)
    all_traces = all_traces[:n_win] #First n_win traces to keep correlation with Key/plaintext data
    for trace in all_traces:
        start_cut = random.sample(range(start, len(trace)-win_width), 1)[0]
        rounds.append(trace[start_cut:start_cut+win_width])
    return rounds


def _getNoiseRounds(noise_traces, n_win, win_width):
    rounds = []
    noise_traces = random.sample(noise_traces, n_win)
    for noise_trace in noise_traces:
        if len(noise_trace)-win_width == 0:
            start_cut = 0
        else:
            start_cut = random.sample(range(len(noise_trace)-win_width), 1)[0]
        rounds.append(noise_trace[start_cut:start_cut+win_width])
    return rounds


def _validNoiseRounds(noise_traces, win_width):
    valid_traces = []
    for trace in noise_traces:
        if len(trace) >= win_width:
            valid_traces.append(trace)
    assert len(valid_traces) > 0, "No valid noise traces found"
    return valid_traces


def _appendDataset(npa_set: NpyAppendArray, npa_meta: NpyAppendArray,
                   AES_rounds, Metadata):

    npa_set.append(np.array(AES_rounds))
    #npa_set.append(np.array(second_rounds))
    #npa_set.append(np.array(other_rounds))
    #npa_set.append(np.array(noise_rounds))

    #npa_tar.append(np.array([1]*len(first_rounds), dtype=np.uint8))
    #npa_tar.append(np.array([0]*len(second_rounds), dtype=np.uint8))
    #npa_tar.append(np.array([0]*len(other_rounds), dtype=np.uint8))
    #npa_tar.append(np.array([2]*len(noise_rounds), dtype=np.uint8))
    
    npa_meta.append(np.array(Metadata))
    # Add part for crypto labels


def _cutCOs(trace, sample_labels, crypto_k_labels, crypto_pt_labels):   #We expect to have as many plaintexts in a trace as pinpoint-sets and only 1 key
    traces_cut = []
    #noise_cut = []
    pt = []
    key = []
    #print(f"Keys: {crypto_k_labels}") #for debug
    for i in range(len(sample_labels) - 1):
        pins = sample_labels[i]
        #next_pins = sample_labels[i + 1]
        traces_cut.append(trace[pins['start']:pins['end']])
        #noise_cut.append(trace[pins['end']:next_pins['start']])
        pt.append(crypto_pt_labels[i]) #Save also plaintext information of AES round
        key.append(crypto_k_labels[0]) #Save also key information of AES round

    # Handle the last segment if necessary
    if sample_labels and sample_labels[-1]['end'] < 134_217_550:
        last_pins = sample_labels[-1]
        traces_cut.append(trace[last_pins['start']:last_pins['end']])
        #noise_cut.append(trace[last_pins['end']:])
        pt.append(crypto_pt_labels[-1]) #Save also plaintext information of AES round
        key.append(crypto_k_labels) #Save also key information of AES round

    return traces_cut, pt, key


def _dataLoader(chunk_files):
    for chunk in chunk_files:
        with h5py.File(chunk, 'r', libver='latest') as hf_chunk:
            chunk_len = len(hf_chunk['metadata/ciphers/'].keys())
            for n in range(0, chunk_len):
                traces = hf_chunk[f'data/traces/trace_{n}']
                labels = hf_chunk[f'metadata/pinpoints/pinpoints_{n}']
                crypto_k_labels = hf_chunk[f'metadata/ciphers/ciphers_{n}/key']
                crypto_pt_labels = hf_chunk[f'metadata/ciphers/ciphers_{n}/plaintexts']
                traces = highpass(traces)
                yield traces, labels, crypto_k_labels, crypto_pt_labels


def createSubsets(dataset_dir: str,
                  out_data_dir: str,
                  window_size: int = 10_000,
                  split_traces: float = 0.8):
    '''
    Create subsets for training and testing a CNN from the Chameleon dataset.

    Parameters
    ----------
    `dataset_dir` : str
        Path to the Chameleon dataset.
    `out_data_dir` : str
        Path to the output directory.
    `window_size` : int, optional
        Size of the window in sample (default is 500).
    `split_traces` : float, optional
        The proportion of traces to use for training (default is 0.8).
    '''

    # Get all traces files ----------------------
    traces_files = _getFilePathes(dataset_dir)
    # -------------------------------------------------------------------------------------------------------------
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    # -------------------------------------------

    train_set_path = os.path.join(out_data_dir, 'train_set.npy')    #AES TRACES file
    #train_tar_path = os.path.join(out_data_dir, 'train_labels.npy')
    train_tar2_path = os.path.join(out_data_dir, 'train_meta.npy')  #PTEXTS | KEY file
    valid_set_path = os.path.join(out_data_dir, 'valid_set.npy')    #AES TRACES file
    #valid_tar_path = os.path.join(out_data_dir, 'valid_labels.npy')
    valid_tar2_path = os.path.join(out_data_dir, 'valid_meta.npy')  #PTEXTS | KEY file
    test_set_path = os.path.join(out_data_dir, 'test_set.npy')      #AES TRACES file
    #test_tar_path = os.path.join(out_data_dir, 'test_labels.npy')
    test_tar2_path = os.path.join(out_data_dir, 'test_meta.npy')    #PTEXTS | KEY file

    with NpyAppendArray(train_set_path, delete_if_exists=True) as npa_train_set, \
            NpyAppendArray(train_tar2_path, delete_if_exists=True) as npa_train_tar2,  \
            NpyAppendArray(valid_set_path, delete_if_exists=True) as npa_valid_set,  \
            NpyAppendArray(valid_tar2_path, delete_if_exists=True) as npa_valid_tar2,  \
            NpyAppendArray(test_set_path, delete_if_exists=True) as npa_test_set,    \
            NpyAppendArray(test_tar2_path, delete_if_exists=True) as npa_test_tar2:
            #NpyAppendArray(train_tar_path, delete_if_exists=True) as npa_train_tar,  \
            #NpyAppendArray(valid_tar_path, delete_if_exists=True) as npa_valid_tar,  \
            #NpyAppendArray(test_tar_path, delete_if_exists=True) as npa_test_tar,   \

        dataLoader = _dataLoader(traces_files)
        for all_traces, sample_labels, crypto_k_labels, crypto_pt_labels in tqdm(dataLoader, total=len(traces_files*16), desc="Creating dataset"):

            # Cut traces --------------------------------------------------------------------------------------------
            # Cut idling parts from traces
            all_traces, pt, key = _cutCOs(all_traces, sample_labels, crypto_k_labels, crypto_pt_labels)  #This does what we want!
            #all_noises = _validNoiseRounds(all_noises, window_size)

            N_WIN_AES = len(all_traces)
            #N_WIN_SECOND = min(len(all_traces), len(all_noises)) // 2
            #N_WIN_OTHER = min(len(all_traces), len(all_noises)) // 2
            #N_WIN_NOISE = min(len(all_traces), len(all_noises))
            #print(f"First: {N_WIN_FIRST} - Second: {N_WIN_SECOND} - Other: {N_WIN_OTHER} - Noise: {N_WIN_NOISE}")

            n_AES_round_train = round(N_WIN_AES*split_traces)
            #n_second_round_train = round(N_WIN_SECOND*split_traces)
            #n_other_round_train = round(N_WIN_OTHER*split_traces)
            #n_noise_round_train = round(N_WIN_NOISE*split_traces)

            n_AES_round_valid = round(N_WIN_AES*(1-split_traces)/2)
            #n_second_round_valid = round(N_WIN_SECOND*(1-split_traces)/2)
            #n_other_round_valid = round(N_WIN_OTHER*(1-split_traces)/2)
            #n_noise_round_valid = round(N_WIN_NOISE*(1-split_traces)/2)
            # -------------------------------------------------------------------------------------------------------

            
            aes_rounds , aes_meta = _getAESRounds(all_traces, key, pt)
            # Get first round of AES
            #first_rounds = _getOrderRounds(all_traces, N_WIN_FIRST, window_size)    #115=n_win 10.000=window_size

            # Get second round of AES
            #second_rounds = _getOrderRounds(all_traces, N_WIN_SECOND, window_size, window_size)

            # Get other parts of AES
            #start = 2*window_size if N_WIN_SECOND > 0 else window_size
            #other_rounds = _getRandomRounds(all_traces, N_WIN_OTHER, window_size, start=start)
            # -------------------------------------------------------------------------------------------------------


            # Get noise
            #noise_rounds = _getNoiseRounds(all_noises, N_WIN_NOISE, window_size)
            # -------------------------------------------------------------------------------------------------------

            # Incrementally build dataset
            # Train set ---------------------------------------------------------------------------------------------
            _appendDataset(npa_train_set, npa_train_tar2,
                           aes_rounds[:n_AES_round_train],
                           aes_meta[:n_AES_round_train])
                           #second_rounds[:n_second_round_train],
                           #other_rounds[:n_other_round_train])#,
                           #noise_rounds[:n_noise_round_train])

            # Validation set ----------------------------------------------------------------------------------------
            _appendDataset(npa_valid_set, npa_valid_tar2,
                           aes_rounds[n_AES_round_train:n_AES_round_train     +   n_AES_round_valid],
                           aes_meta[n_AES_round_train:n_AES_round_train  +   n_AES_round_valid])#,
                           #other_rounds[n_other_round_train:n_other_round_train     +   n_other_round_valid])#,
                           #noise_rounds[n_noise_round_train:n_noise_round_train     +   n_noise_round_valid])

            # Test set ----------------------------------------------------------------------------------------------
            _appendDataset(npa_test_set, npa_test_tar2,
                           aes_rounds[n_AES_round_train     +   n_AES_round_valid:N_WIN_AES],
                           aes_meta[n_AES_round_train     +   n_AES_round_valid:N_WIN_AES])
                           #second_rounds[n_second_round_train   +   n_second_round_valid:N_WIN_SECOND],
                           #other_rounds[n_other_round_train     +   n_other_round_valid:N_WIN_OTHER])#,
                           #noise_rounds[n_noise_round_train     +   n_noise_round_valid:N_WIN_NOISE])

        _writeConfig(out_data_dir,
                     npa_train_set.shape,
                     npa_valid_set.shape,
                     npa_test_set.shape,
                     N_WIN_AES,
                     "0.01")

def debug_createSubsets(dataset_dir: str,
                  out_data_dir: str,
                  window_size: int = 10_000,
                  split_traces: float = 0.8):
    # Get all traces files ----------------------
    traces_files = _getFilePathes(dataset_dir)
    print(traces_files)
    # -------------------------------------------------------------------------------------------------------------
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    # -------------------------------------------
    dataLoader = _dataLoader(traces_files)
    
    ########################################################################
    # TODO: implement cutting function for traces based on minimum length
    ########################################################################
    #min_len = min(len(trace) for trace in all_traces)
    #print(min_len)
    for all_traces, sample_labels, crypto_k_labels, crypto_pt_labels in tqdm(dataLoader, total=len(traces_files*16), desc="Creating dataset"):
        # Cut traces --------------------------------------------------------------------------------------------
        # Cut idling parts from traces
        all_traces, pt, key = _cutCOs(all_traces, sample_labels, crypto_k_labels, crypto_pt_labels)  #This does what we want!

        N_WIN_AES = len(all_traces)
        
        n_AES_round_train = round(N_WIN_AES*split_traces)
        n_AES_round_valid = round(N_WIN_AES*(1-split_traces)/2)
        
        debug(all_traces, key, pt)
       
