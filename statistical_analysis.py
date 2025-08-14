import numpy as np
from pathlib import Path
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import csv
import re
import librosa
from running_mean import RunningStatsArray

class CrossCorrelationProcessor:
    """Class to process cross-correlation data for statistical analysis.
    This class handles the calculation of grand averages, significance testing, and plotting of results.

    Parameters
    ----------
    main_path : str
        The path where the experimental data and the results of run_DPOAE_analysis.py are stored.
    participants : list
        A list of participant identifiers (e.g., ['P01', 'P02', ...]) to process.

    Attributes
    ----------
    chapter_conditions : defaultdict
        A dictionary mapping conditions to RunningStatsArray objects for storing running statistics.
    chapter_conditions_counter : defaultdict
        A dictionary mapping conditions to the number of files processed for that condition.
    participants : list
        A list of participant identifiers.
    main_path : str
        The path where the experimental data and results are stored.
    res_path : str
        The path where the results of the statistical evaluation will be saved.
    plot_path : str
        The path where the plots will be saved.
    fs : int
        The sampling frequency used in the analysis.
    sig_border : int
        The percentile threshold for determining significance in the analysis (default is 97).
    """
    
    def __init__(self, main_path, participants):
        self.chapter_conditions = defaultdict(RunningStatsArray)  # Dict mapping conditions to RunningStatsArray
        self.chapter_conditions_counter = defaultdict(int)  # Dict mapping conditions to number of files
        self.participants = participants
        self.main_path = main_path
        self.res_path = main_path + f'Statistical_Evaluation/{participants[0]}_to_{participants[-1]}/'
        self.plot_path = self.res_path + f'Plots/'
        self.fs = 44100
        self.sig_border = 97 # percentile for significance

        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        plt.rcParams.update({
            'font.size': 14,          # General font size
            'axes.titlesize': 16,     # Title font size
            'axes.labelsize': 16,     # X and Y label font size
            'xtick.labelsize': 12,    # X-tick label size
            'ytick.labelsize': 14,    # Y-tick label size
            'legend.fontsize': 14     # Legend font size
        })

    ###########################################################################

    def extract_chapter_and_condition(self, filename, filepath, stim_info=None):
        """Extract the chapter number and condition from the filename.
        
        Parameters
        ----------
        filename : str
            The name of the file.
        filepath : str
            The full path to the file.
        stim_info : pd.Series
            Series containing stimulus information for the chapters. Defaults to None.
            Is only provided for attended video conditions to distinguish between stimuli.
            
        Returns
        -------
        chapter : int or None
            The chapter number extracted from the filename, or None if not applicable.
        condition : str or None
            The condition extracted from the filename, or None if not applicable.
        """

        chapter = None
        # Extract chapter (in the form ChXX_..._.npy)
        if filename.startswith("Ch"):
            ch = filename.split("_")[0]
            chapter = int(ch.split("Ch")[1])  # Extract chapter number

        # Extract condition
        condition = None

        # Sort conditions for the Single Speaker Scenario
        if "Single_Speaker" in filepath:

            # Female Single Speaker: attended Polarnacht female, ignored Darum male (which was not even presented, but include for demonstration purposes)
            if "Target_female" in filepath:
                if 'Attended_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Single Speaker Attended female unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Single Speaker Attended female resolved (DP5)"
                elif 'Ignored_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Single Speaker Ignored male unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Single Speaker Ignored male resolved (DP4)"

            # Male Single Speaker: attended Polarnacht male, ignored Darum female (which was not even presented, but include for demonstration purposes)
            elif "Target_male" in filepath:
                if 'Attended_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Single Speaker Attended male unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Single Speaker Attended male resolved (DP4)"
                elif 'Ignored_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Single Speaker Ignored female unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Single Speaker Ignored female resolved (DP5)"

        # Sort conditions for the Competing Speaker Scenario
        elif "Competing_Speaker" in filepath:

            # Competing Speaker with attention on Polarnacht female, ignored Darum male
            if "Target_female" in filepath:
                if 'Attended_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Competing Speaker Attended female unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Competing Speaker Attended female resolved (DP5)"
                elif 'Ignored_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Competing Speaker Ignored male unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Competing Speaker Ignored male resolved (DP4)"

            # Competing Speaker with attention on Polarnacht male, ignored Darum female
            elif "Target_male" in filepath:
                if 'Attended_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Competing Speaker Attended male unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Competing Speaker Attended male resolved (DP4)"
                elif 'Ignored_DPs' in filepath:
                    if "_unresolved" in filename:
                        condition = "Competing Speaker Ignored female unresolved (DP12)"
                    elif "_resolved" in filename:
                        condition = "Competing Speaker Ignored female resolved (DP5)"

            # Competing Speaker with attention on video, ignored male and female speaker
            elif "Target_video" in filepath:

                # Stimuli match Polarnacht male, Darum female and therefore provide baseline values for condition Attended_male (for male voice); matching ignored female conditions should be comparable to Darum female stimuli
                if stim_info[chapter-1] == "Stimuli_Polarnacht_male_Darum_female":
                    if "Attended_DPs" in filepath:
                        if "_unresolved" in filename:
                            condition = "Competing Speaker Attended video Baseline male unresolved (DP12)"
                        elif "_resolved" in filename:
                            condition = "Competing Speaker Attended video Baseline male resolved (DP4)"
                    elif "Ignored_DPs" in filepath:
                        if "_unresolved" in filename:
                            condition = "Competing Speaker Attended video Baseline female unresolved (DP12)"
                        elif "_resolved" in filename:
                            condition = "Competing Speaker Attended video Baseline female resolved (DP5)"
                
                # Stimuli match Polarnacht female, Darum male and therefore provide baseline values for condition Attended_female (for female voice); matching ignored male conditions should be comparable to Darum male stimui
                elif stim_info[chapter-1] == "Stimuli_Polarnacht_female_Darum_male":
                    if "Attended_DPs" in filepath:
                        if "_unresolved" in filename:
                            condition = "Competing Speaker Attended video Baseline female unresolved (DP12)"
                        elif "_resolved" in filename:
                            condition = "Competing Speaker Attended video Baseline female resolved (DP5)"
                    elif "Ignored_DPs" in filepath:
                        if "_unresolved" in filename:
                            condition = "Competing Speaker Attended video Baseline male unresolved (DP12)"
                        elif "_resolved" in filename:
                            condition = "Competing Speaker Attended video Baseline male resolved (DP4)"

        return chapter, condition  # Return both chapter and condition

    def find_peak_amplitude(self, env, lag, roi=7):
        """Find the peak amplitude in the envelope within a specified range.

        Parameters
        ----------
        env : np.ndarray
            The envelope of the complex signal.
        lag : np.ndarray
            The lag values corresponding to the envelope.
        roi : int or str, optional
            The region of interest in milliseconds or 'GA_data' for grand average based information. Default is 7 ms.
            'GA_data' only works if GA_peak_pos.csv is available in the results path, which contains the expected peak position and the border for the peak area.
            GA_peak_pos.csv can be created via identify_peak_area method from StatsPlotter class.

        Returns
        -------
        lag_peak : float
            The lag at which the peak amplitude occurs, in frames.
        mag_peak : float
            The magnitude of the peak amplitude.
        """

        if roi == 'GA_data': 
            df = pd.read_csv(self.res_path + '/Data_Plots/GA_peak_pos.csv')
            peak_pos = df['Expected peak position [frames]'][0]
            border = df['Peak area border [frames]'][0]
            idx_min = np.argmin(np.abs(lag - (peak_pos - border)))
            idx_max = np.argmin(np.abs(lag - (peak_pos + border)))
        else:
            border = roi / 1000 * self.fs # in frames
            idx_min = np.argmin(np.abs(lag + border))
            idx_max = np.argmin(np.abs(lag - border))

        env_range = env[idx_min:idx_max + 1]
        mag_peak, idx = np.max(env_range), np.argmax(env_range)
        idx = idx + idx_min
        lag_peak = lag[idx] # in frames
        lag_peak = np.real(lag_peak)
        mag_peak = np.real(mag_peak)

        return lag_peak, mag_peak
    
    def determine_noise_level(self, env, lag, return_env_noise=False):
        """Determine the noise level in the envelope based on the lag values.

        Parameters
        ----------
        env : np.ndarray
            The envelope of the complex signal.
        lag : np.ndarray
            The lag values corresponding to the envelope.
        return_env_noise : bool, optional
            If True, return the noise level and the noise envelope. If False, return only the noise level. Default is False.

        Returns
        -------
        noise_level : float
            The noise level in the envelope, defined as the mean of envelope values in the noise region.
        env_noise : np.ndarray, optional
            The envelope values within the noise region, if `return_env_noise` is True.
        """

        l, h = 70, 750 # ms
        l, h = l / 1000 * self.fs, h / 1000 * self.fs # frames
        idx_noise = np.where(((lag < -l) & (lag > -h)) | ((lag > l) & (lag < h)))[0]
        env_noise = env[idx_noise]
        noise_level = np.real(np.mean(env_noise))

        if return_env_noise:
            return noise_level, env_noise
        else:
            return noise_level

    def get_means(self):
        """Return the computed running means for all (chapter, condition) pairs."""
        return {key: stats.mean for key, stats in self.chapter_conditions.items() if stats.mean is not None}
    
    def get_conditions_counter(self):
        """Return the number of files per condition."""
        cond_list = {}
        for key, value in self.chapter_conditions_counter.items():
            print(f"{key}: \t \t {value}")
            cond_list[key] = value
        return cond_list
    
    def sort_files_to_conditions(self):
        """Sort the files into the correct conditions. Results are saved in a .npz file."""
        cond_files = defaultdict(list)
        for p in self.participants:
            p_path = self.main_path + f'results_{p}/'
            stim_info = pd.read_csv(p_path + 'stats_CompetingSpeaker.csv')['Stimulus Shortcut']
            directory = Path(p_path)
            all_files = list(directory.rglob('*.npy'))

            for filename in all_files:
                filepath = str(filename)
                filename = filepath.split("/")[-1]
                chapter, condition = self.extract_chapter_and_condition(filename, filepath, stim_info)
                cond_files[condition].append(filepath)

        # save the dictionary with the sorted files
        np.savez(self.res_path + 'files_per_condition.npz', **cond_files)
        return
    
    def sort_psd_csv_to_conditions(self):
        """Sort the PSD analysis CSV files into the correct conditions. Results are saved in a .npz file."""
        cond_files = defaultdict(list)
        for p in self.participants:
            p_path = self.main_path + f'results_{p}/'
            stim_info = pd.read_csv(p_path + 'stats_CompetingSpeaker.csv')['Stimulus Shortcut']
            directory = Path(p_path + 'DP_PSD_Analysis/')
            all_files = list(directory.rglob('Ch*.csv'))

            for filename in all_files:
                filepath = str(filename)
                filename = filepath.split("/")[-1]
                chapter, condition = self.extract_chapter_and_condition(filename, filepath, stim_info)
                cond_files[condition].append(filepath)
        
        # save the dictionary with the sorted files
        np.savez(self.res_path + 'psd_files_per_condition.npz', **cond_files)
        return
            
    def write_results_csv(self, columns, results, save_path):
        """Write the results to a csv file

        Parameters
        ----------
        columns : list
            The column names
        results : list
            The results, number of columns must match the number of column names
        save_path : str
            The path where the results should be saved

        Returns
        -------
        None
        """
        with open(save_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(results)
        return
    
    #############################################################################################
    
    def calculate_grand_average(self, data_dir):
        """Iterate over .npy files, group them by (chapter, condition), and compute running means.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the .npy files to process.

        Returns
        -------
        None
        """
        
        file_count = 0  # Number of files processed
        stim_info = pd.read_csv(data_dir + 'stats_CompetingSpeaker.csv')['Stimulus Shortcut']
        directory = Path(data_dir)
        all_files = list(directory.rglob('*.npy'))

        for filename in all_files:
            file_count += 1
            filepath = str(filename)
            filename = filepath.split("/")[-1]
            chapter, condition = self.extract_chapter_and_condition(filename, filepath, stim_info)

            # extract participant number from filepath
            p = re.search(r'P\d{2}', filepath).group(0)

            if condition is not None:
                data = np.load(filepath) # Load the .npy file - contains correlation results in the format [lag, real, imag, env]
                # check that data is aligned correctly
                idx_min = np.where(data[0] == -40000)[0][0]
                idx_max = np.where(data[0] == 40000)[0][0]
                data = data[:, idx_min:idx_max + 1]
                self.chapter_conditions[condition].push(data)
                self.chapter_conditions_counter[condition] += 1
        return
    
    def plot_means_via_envelope(self, plot_path):
        """Plot the means of the running stats for each condition using the envelope of the complex signal.

        Parameters
        ----------
        plot_path : str
            The path where the plots should be saved.

        Returns
        -------
        peak_stats : dict
            A dictionary containing the peak statistics for each condition, including lag peak in ms, lag peak in frames, peak magnitude, noise level, and SNR.
        """

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        means = self.get_means()
        peak_stats = {}

        for condition, mean in means.items():
            lag = mean[0, :] # in frames
            env = mean[3, :]
            
            lag_peak, mag_peak = self.find_peak_amplitude(env, lag)
            noise_level, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
            if mag_peak > max(env_noise):
                sig_peak = True
            else:
                sig_peak = False

            lag_peak = int(lag_peak)
            lag_peak_ms = lag_peak / self.fs * 1000
            snr = 20 * np.log10(mag_peak / noise_level)
            save_name = 'GA_envelope_' + condition.replace(' ', '_')

            plt.figure(figsize=(8, 5))
            plt.plot(lag/self.fs *1000, env, label='env', color='black')
            plt.title(f'{condition} - env')
            plt.xlim(-70, 70)
            plt.ylim(0, 0.0025)
            if sig_peak:
                plt.axvline(lag_peak_ms, c='silver', ls='--', label=f'{lag_peak_ms:.2f} ms \nPeak significant', alpha=0.5)
            else:
                plt.axvline(lag_peak_ms, c='silver', ls='--', label=f'{lag_peak_ms:.2f} ms \nPeak not significant', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(plot_path + f'/{save_name}.png', dpi=1200)
            plt.close()

            peak_stats[condition] = (lag_peak_ms, lag_peak, mag_peak, noise_level, snr)
        
        return peak_stats
    
    def run_ga_analysis(self):
        """Run the grand average analysis for all participants and conditions.
        This method calculates the grand average for each participant, plots the means via envelope, and saves the results in a CSV file.

        Returns
        -------
        formatted_means : dict
            A dictionary containing the means for each condition, formatted for saving.
        GA_results : np.ndarray
            A numpy array containing the grand average results, including condition, count, lag peak in ms, lag peak in frames, magnitude peak, noise level, and SNR.
        """

        for p in self.participants:
            p_path = self.main_path + f'results_{p}/'
            self.calculate_grand_average(p_path)

        GA_stats_envelope = self.plot_means_via_envelope(self.plot_path + 'GA_envelope/')
        GA_cond_counts = self.get_conditions_counter()

        # check that all conditions are sorted the same way
        assert GA_stats_envelope.keys() == GA_cond_counts.keys()
        
        GA_columns = ['Condition', 'Count', 'ENVELOPE Lag Peak [ms]', 'ENVELOPE Lag Peak [frames]', 'ENVELOPE Magnitude Peak', 'ENVELOPE Noise Level', 'ENVELOPE SNR [dB]']
        GA_conds = list(GA_stats_envelope.keys())
        r, c = np.array([*GA_stats_envelope.values()]).shape
        GA_results = np.concatenate((np.array(GA_conds).reshape(r, 1), np.array([*GA_cond_counts.values()]).reshape(r, 1), np.array([*GA_stats_envelope.values()])), axis=1)
        self.write_results_csv(GA_columns, GA_results, self.res_path + 'GA_stats.csv')

        formatted_means = {f"{condition}": mean for condition, mean in self.get_means().items()}
        np.save(self.res_path + 'means.npy', formatted_means, allow_pickle=True)

        return formatted_means, GA_results
    
    def attentional_modulation_coefficient(self, data_source='Mag', only_sig=False, ga_delay=False):
        """Calculate the attentional modulation coefficient (AMC) for the given data source.
        
        Parameters
        ----------
        data_source : str, optional
            The data source to use for the AMC calculation. Options are 'Mag', 'SNR'
            or 'PSD'. Default is 'Mag'.
        only_sig : bool, optional
            If True, only significant peaks are considered for the AMC calculation. Default is False.
        ga_delay : bool, optional
            If True, the GA delay is used to determine the peak position. Default is False.

        Returns
        -------
        amc : dict
            A dictionary containing the attentional modulation coefficients for each participant and condition.
        """
        
        if data_source == 'PSD':
            unit = 'Pa^2'
            files_per_condition = np.load(self.res_path + 'psd_files_per_condition.npz', allow_pickle=True)
            files_per_condition = dict(files_per_condition)
        else:
            unit=''
            files_per_condition = np.load(self.res_path + 'files_per_condition.npz', allow_pickle=True)
            files_per_condition = dict(files_per_condition)

            ga_info = pd.read_csv(self.res_path + 'GA_stats.csv')
            delay_ga = ga_info['ENVELOPE Lag Peak [frames]']
            conds_ga = ga_info['Condition']

        inner_keys = ['Atd_Ign_res', 'Atd_Ign_unres', 'Atd_BL_res', 'Atd_BL_unres', 'Ign_BL_res', 'Ign_BL_unres']
        amc = {
            'female': {p: {k: None for k in inner_keys} for p in self.participants},
            'male': {p: {k: None for k in inner_keys} for p in self.participants}
        } 
        
        # remove Single Speaker conditions
        conditions = [cond for cond in files_per_condition.keys() if 'Single' not in cond]
        data_per_P_cond = {p: {cond: [] for cond in conditions} for p in self.participants}

        for key in conditions:
            all_files = files_per_condition[key]

            for file in all_files:
                # determine participant
                p = re.search(r'P\d{2}', file).group(0)

                if data_source == 'PSD':
                    data = pd.read_csv(file)
                    val = data[f'Band Power [{unit}]'][0]
                    if only_sig:
                        # extract the chapter
                        file_info = file.split('Ch')[1].split('_')
                        ch = file_info[0]
                        harm = file_info[2]
                        # load the correlation data by removing 'DP_PSD_Analysis from file
                        corr_path = file.replace('DP_PSD_Analysis/', '').split('/')[:-1]
                        corr_file = '/'.join(corr_path) + f'/Ch{ch}_corr_{harm}.npy'
                        data_corr = np.load(corr_file)
                        lag = data_corr[0, :]
                        env = data_corr[3, :]
                        _, mag_peak = self.find_peak_amplitude(env, lag)
                        _, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
                        
                        if mag_peak > np.percentile(env_noise, self.sig_border):
                            data_per_P_cond[p][key].append(val)
                    else:
                        data_per_P_cond[p][key].append(val)

                else:
                    data = np.load(file)
                    lag = data[0, :]
                    env = data[3, :]

                    if ga_delay:
                        i = np.where(conds_ga == key)[0][0]
                        peak_val = int(delay_ga[i]) #in frames
                        peak_pos = np.where(lag == peak_val)[0][0]
                        mag_peak = env[peak_pos]
                        noise_level, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
                        snr = 20 * np.log10(mag_peak / noise_level)
                        if data_source == 'Mag':
                            data_per_P_cond[p][key].append(mag_peak)
                        elif data_source == 'SNR':
                            data_per_P_cond[p][key].append(snr)
                    
                    elif only_sig:
                        _, mag_peak = self.find_peak_amplitude(env, lag)
                        noise_level, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
                        snr = 20 * np.log10(mag_peak / noise_level)
                        if mag_peak > np.percentile(env_noise, self.sig_border):
                            if data_source == 'Mag':
                                data_per_P_cond[p][key].append(mag_peak)
                            elif data_source == 'SNR':
                                data_per_P_cond[p][key].append(snr)
                    else:
                        _, mag_peak = self.find_peak_amplitude(env, lag)
                        noise_level, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
                        snr = 20 * np.log10(mag_peak / noise_level)
                        if data_source == 'Mag':
                            data_per_P_cond[p][key].append(mag_peak)
                        elif data_source == 'SNR':
                            data_per_P_cond[p][key].append(snr)

        for p in self.participants:        
            for g in ['male', 'female']:
                for harm in ['resolved', 'unresolved']:
                    # choose matching condition attended and ignored
                    if harm == 'resolved':
                        dp_no = 'DP5' if g == 'female' else 'DP4'
                        amc_cond_ending = 'res'
                    else:
                        dp_no = 'DP12'
                        amc_cond_ending = 'unres'
                    
                    cond_atd = f'Competing Speaker Attended {g} {harm} ({dp_no})'
                    cond_ign = f'Competing Speaker Ignored {g} {harm} ({dp_no})'
                    cond_BL = f'Competing Speaker Attended video Baseline {g} {harm} ({dp_no})'
                    # get the data for the conditions
                    atd_data = data_per_P_cond[p][cond_atd]
                    ign_data = data_per_P_cond[p][cond_ign]
                    BL_data = data_per_P_cond[p][cond_BL]
                    # calculate the means
                    atd_mean = np.mean(atd_data)
                    ign_mean = np.mean(ign_data)
                    BL_mean = np.mean(BL_data)
                    # calculate the attentional modulation coefficient
                    amc[g][p][f'Atd_Ign_{amc_cond_ending}'] = 2 * (atd_mean - ign_mean) / (atd_mean + ign_mean)
                    amc[g][p][f'Atd_BL_{amc_cond_ending}'] = 2 * (atd_mean - BL_mean) / (atd_mean + BL_mean)
                    amc[g][p][f'Ign_BL_{amc_cond_ending}'] = 2 * (ign_mean - BL_mean) / (ign_mean + BL_mean)

        # sort amcs for boxplots with inner_keys as keys and values for all participants as list
        amc_vals = {'male': {k: [] for k in inner_keys},
                    'female': {k: [] for k in inner_keys}}

        for g in ['male', 'female']:
            for p in self.participants:
                for key in inner_keys:
                    amc_vals[g][key].append(amc[g][p][key])
        
        return amc_vals
    
    def delay_cycles(self, delays, only_sig=False, avg_participant=False):       
        """Calculate the delay cycles for the given delays and conditions.

        Parameters
        ----------
        delays : dict
            A dictionary where keys are condition names and values are lists of delays in milliseconds.
        only_sig : bool, optional
            If True, only significant delays are considered for the cycle calculation. Default is False.
        avg_participant : bool, optional
            If True, the cycles are averaged across participants. Default is False.
        
        Returns
        -------
        cycles : dict
            A dictionary where keys are condition names and values are lists of cycles in seconds.
        """

        if only_sig:
            # load the sig_files_per_condition.npz
            files_per_condition = np.load(self.res_path + 'sig_files_per_condition.npz', allow_pickle=True)
            files_per_condition = dict(files_per_condition)
        else:
            # load the files_per_condition.npz
            files_per_condition = np.load(self.res_path + 'files_per_condition.npz', allow_pickle=True)
            files_per_condition = dict(files_per_condition)

        dp_main_path = self.main_path + 'Stimuli/'
        dp_wf_to_cond = {'Single Speaker Attended female resolved (DP5)': ['Polarnacht_female', 'n5_Polarnacht_female_single'],
                         'Single Speaker Attended female unresolved (DP12)': ['Polarnacht_female', 'n12_Polarnacht_female_single'],
                         'Single Speaker Attended male resolved (DP4)': ['Polarnacht_male', 'n4_Polarnacht_male_single'],
                         'Single Speaker Attended male unresolved (DP12)': ['Polarnacht_male', 'n12_Polarnacht_male_single'],
                         'Single Speaker Ignored female resolved (DP5)': ['Darum_female', 'n5_Darum_female'],
                         'Single Speaker Ignored female unresolved (DP12)': ['Darum_female', 'n12_Darum_female'],
                         'Single Speaker Ignored male resolved (DP4)': ['Darum_male', 'n4_Darum_male'],
                         'Single Speaker Ignored male unresolved (DP12)': ['Darum_male', 'n12_Darum_male'],
                         'Competing Speaker Attended female resolved (DP5)': ['Polarnacht_female', 'n5'],
                         'Competing Speaker Attended female unresolved (DP12)': ['Polarnacht_female', 'n12'],
                         'Competing Speaker Ignored male resolved (DP4)': ['Darum_male', 'n4'],
                         'Competing Speaker Ignored male unresolved (DP12)': ['Darum_male', 'n12'],
                         'Competing Speaker Attended male resolved (DP4)': ['Polarnacht_male', 'n4'],
                         'Competing Speaker Attended male unresolved (DP12)': ['Polarnacht_male', 'n12'],
                         'Competing Speaker Ignored female resolved (DP5)': ['Darum_female', 'n5'],
                         'Competing Speaker Ignored female unresolved (DP12)': ['Darum_female', 'n12'],
                         'Competing Speaker Attended video Baseline female resolved (DP5)': ['', 'n5'],
                         'Competing Speaker Attended video Baseline female unresolved (DP12)': ['', 'n12'],
                         'Competing Speaker Attended video Baseline male resolved (DP4)': ['', 'n4'],
                         'Competing Speaker Attended video Baseline male unresolved (DP12)': ['', 'n12']}
        
        cycles = {cond: [] for cond in files_per_condition.keys()}

        for key in dp_wf_to_cond.keys():
            if avg_participant:
                Px_cycles = {p: RunningStatsArray() for p in self.participants}
            # extract the path to the DP waveforms from key
            all_files = files_per_condition[key]
            for i, file in enumerate(all_files):
                ch = file.split('Ch')[1].split('_')[0]
                participant = re.search(r'P\d{2}', file).group(0)
                res_path = file.split('results')[0] + f'results_{participant}/'
                stim_info = pd.read_csv(res_path + 'stats_CompetingSpeaker.csv')['Stimulus Shortcut']
                if ' video' in key and ' female' in key:
                    if stim_info[int(ch)-1] == 'Stimuli_Polarnacht_female_Darum_male':
                        dp_wf_to_cond[key][0] = 'Polarnacht_female'
                    elif stim_info[int(ch)-1] == 'Stimuli_Polarnacht_male_Darum_female':
                        dp_wf_to_cond[key][0] = 'Darum_female'
                elif ' video' in key and ' male' in key:
                    if stim_info[int(ch)-1] == 'Stimuli_Polarnacht_female_Darum_male':
                        dp_wf_to_cond[key][0] = 'Darum_male'
                    elif stim_info[int(ch)-1] == 'Stimuli_Polarnacht_male_Darum_female':
                        dp_wf_to_cond[key][0] = 'Polarnacht_male' 
                
                d = delays[key][i] / 1000 # convert to seconds

                freq_info = pd.read_csv(dp_main_path + f'pyin_mean_freq_{dp_wf_to_cond[key][0]}.csv')
                freq_info = {row['File']: row['Mean Frequency [Hz]'] for _, row in freq_info.iterrows()}

                if 'Single Speaker' in key:
                    dp_name = f'dp_{dp_wf_to_cond[key][1]}_{ch}.wav'
                else:
                    dp_name = f'dp_{dp_wf_to_cond[key][1]}_{dp_wf_to_cond[key][0]}_{ch}.wav'

                mean_freq = freq_info[dp_name]
                c = d * mean_freq
                if avg_participant:
                    Px_cycles[participant].push(c)
                else:
                    cycles[key].append(c)
                    
            if avg_participant:
                cycles[key] = [Px_cycles[p].mean for p in Px_cycles.keys()]

        return cycles
    
    def calculate_mean_freq_for_all_DP(self):
        """Calculate the mean frequency for all DP waveforms and save the results as .csv files."""

        # calculate mean frequency for all dp waveforms and save as .csv
        dp_main_path = self.main_path + 'Stimuli/'

        for book in ['Darum', 'Polarnacht']:
                for speaker, harm in zip(['male', 'female'], ['n4', 'n5']):
                    dp_path = dp_main_path + f'{book}_{speaker}/'
                    # load all .wav files that contain harm
                    dp_files_res = [f for f in os.listdir(dp_path) if f.endswith('.wav') and harm in f and f.startswith('dp_')]
                    dp_files_unres = [f for f in os.listdir(dp_path) if f.endswith('.wav') and 'n12' in f and f.startswith('dp_')]
                    dp_files = dp_files_res + dp_files_unres

                    freq_results = []
                    
                    for file in dp_files:
                        # load the waveform
                        wf, _ = librosa.load(dp_path + file, sr=self.fs, mono=True)
                        # extract the number of harmonics from the filename
                        if 'n12' in file:
                            nharm = 12
                        else:
                            nharm = int(harm.split('n')[1])
                        # extract the mean frequency
                        mean_freq, mean_freq_std = self.extract_mean_frequency(wf, nharm, speaker)
                        freq_results.append([file, mean_freq, mean_freq_std])
                    
                    # save the results to a .csv file
                    freq_results_df = pd.DataFrame(freq_results, columns=['File', 'Mean Frequency [Hz]', 'Mean Frequency Std [Hz]'])
                    freq_results_df.to_csv(dp_main_path + f'pyin_mean_freq_{book}_{speaker}.csv', index=False)    

        return    

    def extract_mean_frequency(self, wf, nharm, voice):
        """Extract the mean frequency from the waveform using the Pyin algorithm.

        Parameters
        ----------
        wf : np.ndarray
            The waveform data.
        nharm : int
            The harmonic number of the distortion product.
        voice : str
            The voice type, either 'male' or 'female'

        Returns
        -------
        freqs_mean : float
            The mean frequency of the waveform.
        freqs_std : float
            The standard deviation of the frequencies.
        """

        fr_length = 4000
        win_length = 3300
        hop_length = 1250
        if voice == 'female':
            low, high = 80, 300
        elif voice == 'male':
            low, high = 50, 200
        freqs, voiced_sec, _ = librosa.pyin(wf, sr=self.fs, fmin=(nharm*low), fmax=(nharm*high), frame_length=fr_length, hop_length=hop_length, win_length=win_length)
        freqs_mean = np.mean(freqs[voiced_sec])
        freqs_std = np.std(freqs[voiced_sec])
        
        return freqs_mean, freqs_std
    
    def count_significant_peaks(self, per_participant=False):  
        """Count the number of significant peaks in the correlation data for each condition and participant.
        
        Parameters
        ----------
        per_participant : bool, optional
            If True, the number of significant peaks is counted for each participant separately.
            If False, the number of significant peaks is counted for each condition across all participants.
            Default is False.
        
        Returns
        -------
        perc_sig_peaks : dict
            A dictionary containing the percentage of significant peaks for each condition or participant.
        no_sig_peaks : dict
            A dictionary containing the number of significant peaks for each condition or participant.
        no_all_peaks : dict
            A dictionary containing the total number of peaks for each condition or participant.
        """

        files_per_condition = np.load(self.res_path + 'files_per_condition.npz', allow_pickle=True)
        files_per_condition = dict(files_per_condition)

        conditions = files_per_condition.keys()

        # count the number of significant peaks for each condition
        if per_participant:
            no_sig_peaks = {p: {cond: 0 for cond in conditions} for p in self.participants}
            no_all_peaks = {p: {cond: 0 for cond in conditions} for p in self.participants}
        else:
            no_sig_peaks = {cond: 0 for cond in conditions}
            no_all_peaks = {cond: 0 for cond in conditions}
        
        for cond in conditions:
            all_files = files_per_condition[cond]
            if per_participant == False:
                no_all_peaks[cond] = len(all_files)

            for file in all_files:

                if per_participant:
                    p = re.search(r'P\d{2}', file).group(0)
                    no_all_peaks[p][cond] += 1

                # load the correlation data
                data = np.load(file)
                lag = data[0, :]
                env = data[3, :]

                _, mag_peak = self.find_peak_amplitude(env, lag, roi='GA_data')
                _, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)

                if mag_peak > np.percentile(env_noise, self.sig_border):
                    if per_participant:
                        no_sig_peaks[p][cond] += 1
                    else:
                        no_sig_peaks[cond] += 1
        
        if per_participant:
            perc_sig_peaks = {p: {cond: no_sig_peaks[p][cond] / no_all_peaks[p][cond] for cond in conditions} for p in self.participants}
        else:
            perc_sig_peaks = {cond: no_sig_peaks[cond] / no_all_peaks[cond] for cond in conditions}

        return perc_sig_peaks, no_sig_peaks, no_all_peaks

    def dp_psd_statistics(self, unit, avg_participant=False, only_sig=False):
        """Calculate the power spectral density (PSD) statistics for the DP data.
        
        Parameters
        ----------
        unit : str
            The unit of the PSD, either 'dB SPL' or 'Pa^2'.
        avg_participant : bool, optional
            If True, the PSD is averaged over all participants.
            If False, the PSD is calculated for each condition.
            Default is False.
        only_sig : bool, optional
            If True, only significant peaks are considered.
            If False, all peaks are considered.
            Default is False.
        
        Returns
        -------
        conditions : list
            A list of conditions for which the PSD is calculated.
        psd : dict
            A dictionary containing the PSD values for each condition.
        """

        files_per_condition = np.load(self.res_path + 'psd_files_per_condition.npz', allow_pickle=True)
        files_per_condition = dict(files_per_condition)

        conditions = files_per_condition.keys()

        psd = {cond: [] for cond in conditions}
        
        for cond in conditions:
            all_files = files_per_condition[cond]
            
            if avg_participant:
                Px_psd = {p: RunningStatsArray() for p in self.participants}
            
            for file in all_files:
                # load the psd data from the csv file
                data = pd.read_csv(file)
                val = data[f'Band Power [{unit}]'][0]

                if only_sig:
                    # extract the chapter
                    file_info = file.split('Ch')[1].split('_')
                    ch = file_info[0]
                    harm = file_info[2]
                    # load the correlation data by removing 'DP_PSD_Analysis from file
                    corr_path = file.replace('DP_PSD_Analysis/', '').split('/')[:-1]
                    corr_file = '/'.join(corr_path) + f'/Ch{ch}_corr_{harm}.npy'
                    data_corr = np.load(corr_file)
                    lag = data_corr[0, :]
                    env = data_corr[3, :]
                    _, mag_peak = self.find_peak_amplitude(env, lag)
                    _, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
                    
                    if mag_peak < np.percentile(env_noise, self.sig_border):
                        continue

                if avg_participant:
                    p = re.search(r'P\d{2}', file).group(0)
                    Px_psd[p].push(val)
                else:
                    psd[cond].append(val)

            if avg_participant:
                psd[cond] = [Px_psd[p].mean for p in Px_psd.keys()]
                # if there are None values in the psd, remove them
                psd[cond] = [val for val in psd[cond] if val is not None]

        return conditions, psd

    def prep_boxplot_ga(self, only_sig=False, ga_delay=False, avg_participant=False, peak_roi='GA_data'):
        """Prepare data for boxplots of grand average results.
       
        Parameters
        ----------
        only_sig : bool, optional
            If True, only significant peaks are considered for the boxplots. Default is False.
        ga_delay : bool, optional
            If True, the value at GA delay is used instead of finding the maximum peak amplitude. Default is False.
        avg_participant : bool, optional
            If True, the boxplots are averaged over participants. Default is False.
        peak_roi : str or int, optional
            The region of interest for the peak amplitude. If 'GA_data', the peak position and
            border are taken from the GA data. If an integer, it specifies the region of interest in milliseconds.
            Default is 'GA_data'.
        
        Returns
        -------
        conditions_ga : list
            A list of conditions for the grand average analysis.
        peak_snrs : dict
            A dictionary containing the peak SNRs for each condition.
        peak_mags : dict
            A dictionary containing the peak magnitudes for each condition.
        peak_delays : dict
            A dictionary containing the peak delays for each condition.
        """

        if only_sig and ga_delay:
            raise ValueError("only_sig and ga_delay cannot be True at the same time")
        
        # load the info from the saved .csv regarding the GA delays
        ga_info = pd.read_csv(self.res_path + 'GA_stats.csv')
        conditions_ga = ga_info['Condition']
        delay_ga = ga_info['ENVELOPE Lag Peak [frames]']

        # load the files_per_condition.npz
        files_per_condition = np.load(self.res_path + 'files_per_condition.npz', allow_pickle=True)
        files_per_condition = dict(files_per_condition)

        significant_files_per_cond = {cond: [] for cond in conditions_ga}

        peak_snrs = {cond: [] for cond in conditions_ga}
        peak_mags = {cond: [] for cond in conditions_ga}
        peak_delays = {cond: [] for cond in conditions_ga}

        for i, key in enumerate(conditions_ga):
            all_files = files_per_condition[key]

            if avg_participant:
                Px_peak_snrs = {p: RunningStatsArray() for p in self.participants}
                Px_peak_mags = {p: RunningStatsArray() for p in self.participants}
                Px_peak_delays = {p: RunningStatsArray() for p in self.participants}

            for file in all_files:
                # load the correlation data
                data = np.load(file)
                lag = data[0, :]
                env = data[3, :]
                if ga_delay:
                    peak_val = int(delay_ga[i]) #in frames
                    lag_peak_ms = peak_val / self.fs * 1000
                    peak_pos = np.where(lag == peak_val)[0][0]
                    mag_peak = env[peak_pos]
                else:
                    lag_peak, mag_peak = self.find_peak_amplitude(env, lag, roi=peak_roi)
                    lag_peak_ms = lag_peak / self.fs * 1000
                
                noise_level, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)
                snr = 20 * np.log10(mag_peak / noise_level)

                if avg_participant:
                    p = re.search(r'P\d{2}', file).group(0)
                    if only_sig:
                        if mag_peak > np.percentile(env_noise, self.sig_border):
                            Px_peak_snrs[p].push(snr)
                            Px_peak_mags[p].push(mag_peak)
                            Px_peak_delays[p].push(lag_peak_ms)
                            significant_files_per_cond[key].append(file)
                    else:
                        Px_peak_snrs[p].push(snr)
                        Px_peak_mags[p].push(mag_peak)
                        Px_peak_delays[p].push(lag_peak_ms)
                    
                else:
                    if only_sig:
                        # check if the peak magnitude is above the self.sig_borderth percentile of the noise level
                        if mag_peak > np.percentile(env_noise, self.sig_border):
                            peak_snrs[key].append(snr)
                            peak_mags[key].append(mag_peak)
                            peak_delays[key].append(lag_peak_ms)
                            significant_files_per_cond[key].append(file)
                    else:
                        peak_snrs[key].append(snr)
                        peak_mags[key].append(mag_peak)
                        peak_delays[key].append(lag_peak_ms)
            
            if avg_participant:
                peak_snrs[key] = [Px_peak_snrs[p].mean for p in Px_peak_snrs.keys()]
                peak_mags[key] = [Px_peak_mags[p].mean for p in Px_peak_mags.keys()]
                peak_delays[key] = [Px_peak_delays[p].mean for p in Px_peak_delays.keys()]

        if only_sig:
            # save the significant files per condition
            np.savez(self.res_path + 'sig_files_per_condition.npz', **significant_files_per_cond)

        return conditions_ga, peak_snrs, peak_mags, peak_delays
    
    ###########################################################################################
    
    def evaluate_listening_effort(self, save_path=None):
        """Evaluate the listening effort from the comprehension questions and create boxplots.
        
        Parameters
        ----------
        save_path : str, optional
            The path where the results are saved. If None, the results are not saved.
            Default is None.
        """

        for speaker in ['Single', 'Competing']:
            if speaker == 'Single':
                keys = ['female', 'male']
            else:
                keys = ['female', 'male', 'video']

            le_results = {target: {p: [] for p in self.participants} for target in keys}
            for p in self.participants:
                data = pd.read_csv(self.main_path + f'results_{p}/cq_{speaker}Speaker.csv')
                le = data['Listening Effort'].values
                target = data['Target'].values
                for t, val in zip(target, le):
                    le_results[t][p].append(val)
            
            for target in le_results.keys():
                # calculate the mean and std for each participant
                le_results[target] = {p: np.mean(le_results[target][p]) for p in le_results[target].keys()}
            
            bp_data = [list(le_results[target].values()) for target in le_results.keys()]
            
            # create a boxplot for the listening effort, import necessary methods from StatsPlotter
            from plot_statistics import StatsPlotter
            sp = StatsPlotter(self.res_path, self.participants)

            plt.figure()
            bp = plt.boxplot(bp_data, positions=np.arange(len(bp_data)), labels=le_results.keys())
            if speaker == 'Single':
                sig_comb = sp.determine_significant_combinations(bp_data, related=True)
                sig_comb_fdr = sp.perform_fdr_correction(sig_comb)
                sig_comb_fdr = [sig_comb_fdr]
            else:
                sig_comb = []
                for c in [(1, 3)]:
                    comp = [bp_data[c[0]-1], bp_data[c[1]-1]]
                    sig_comb.append(sp.determine_significant_combinations(comp, [c[0], c[1]], related=True))

                level2 = []
                for i in range(1, len(bp_data)):
                    level2 += sp.determine_significant_combinations([bp_data[i-1], bp_data[i]], [i, i+1], related=True)
                sig_comb.append(level2)
                sig_comb_fdr = sp.perform_fdr_correction(sig_comb)

            sp.draw_significance(bp, sig_comb_fdr)
            plt.title(f'Listening Effort for {speaker} Speaker')
            plt.ylabel('Listening Effort')
            if speaker == 'Single':
                plt.ylim(0, 17.5)
            else:
                plt.ylim(0, 24)
            plt.tight_layout()
            plt.savefig(self.plot_path + f'Listening_Effort_{speaker}Speaker.png', dpi=600)
            plt.close()

            le_means  = {target: np.mean(list(le_results[target].values())) for target in le_results.keys()}
            le_stds = {target: np.std(list(le_results[target].values())) for target in le_results.keys()}
            # save the results as csv
            le_df = pd.DataFrame([[key, le_means[key], le_stds[key]] for key in le_means.keys()])
            le_df.columns = ['Target', 'Mean Listening Effort', 'Std Listening Effort']
            le_df.to_csv(save_path + f'meanLE_{speaker}Speaker.csv', index=False)

        return
    
    def evaluate_comprehension_questions(self, save_path=None):
        """Evaluate the comprehension questions and create boxplots for the results.
        
        Parameters
        ----------
        save_path : str, optional
            The path where the results are saved. If None, the results are not saved.
            Default is None.
        """

        for speaker in ['Single', 'Competing']:
            if speaker == 'Single':
                keys = ['female', 'male']
            else:
                keys = ['female', 'male', 'video']

            cq_results = {target: {p: [] for p in self.participants} for target in keys}
            for p in self.participants:
                data = pd.read_csv(self.main_path + f'results_{p}/cq_{speaker}Speaker.csv')
                no_cq = data['Number Correct Answers'].values
                target = data['Target'].values
                for t, val in zip(target, no_cq):
                    cq_results[t][p].append(val/3)
            
            for target in cq_results.keys():
                # calculate the mean and std for each participant
                cq_results[target] = {p: np.mean(cq_results[target][p]) for p in cq_results[target].keys()}
            
            bp_data = [list(cq_results[target].values()) for target in cq_results.keys()]
            
            # create a boxplot for the listening effort
            from plot_statistics import StatsPlotter
            sp = StatsPlotter(self.res_path, self.participants)

            plt.figure()
            bp = plt.boxplot(bp_data, positions=np.arange(len(bp_data)), labels=cq_results.keys())
            sig_comb = sp.determine_significant_combinations(bp_data, related=True)
            sig_comb_fdr = sp.perform_fdr_correction(sig_comb)
            sig_comb_fdr = [sig_comb_fdr]
            sp.draw_significance(bp, sig_comb_fdr)
            plt.title(f'Perc. of correct CQs for {speaker} Speaker')
            plt.ylabel('Percentage')
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(self.plot_path + f'Correct_CQs_{speaker}Speaker.png', dpi=600)
            plt.close()

            cq_means  = {target: np.mean(list(cq_results[target].values())) for target in cq_results.keys()}
            cq_stds = {target: np.std(list(cq_results[target].values())) for target in cq_results.keys()}
            # save the results as csv
            cq_df = pd.DataFrame([[key, cq_means[key], cq_stds[key]] for key in cq_means.keys()])
            cq_df.columns = ['Target', 'Mean Listening Effort', 'Std Listening Effort']
            cq_df.to_csv(save_path + f'meanCQ_{speaker}Speaker.csv', index=False)

        return

    def run_all_statistics(self, stats_data_path): 
        """ Run all statistics and save the results in the specified path. Statistics can subsequently be used for plotting via StatsPlotter class.
        Statistics include:
        - Grand Average Analysis (GA) for all participants
        - Sorting of files to conditions
        - Calculation of mean frequency for all DP waveforms
        - Sorting of PSD data to conditions
        - Preparation of boxplot data for parameters peak magnitude, peak delay, and attentional modulation coefficient
        - Calculation of number of cycles on the BM
        - Determination of the percentage of significant peaks, number of significant peaks, and total number of peaks for each condition.

        Parameters
        ----------
        stats_data_path : str
            The path where the results are saved. If the path does not exist, it will be created.
        """ 

        if not os.path.exists(stats_data_path):
            os.makedirs(stats_data_path)

        GA_columns = ['Condition', 'Count', 'ENVELOPE Lag Peak [ms]', 'ENVELOPE Lag Peak [frames]', 'ENVELOPE Magnitude Peak', 'ENVELOPE Noise Level', 'ENVELOPE SNR [dB]']
        means, GA_results = self.run_ga_analysis() 
        np.save(stats_data_path + 'GA_means.npy', means, allow_pickle=True)
        self.write_results_csv(GA_columns, GA_results, stats_data_path + 'GA_results.csv')

        self.sort_files_to_conditions()
        self.calculate_mean_freq_for_all_DP()
        self.sort_psd_csv_to_conditions()

        from plot_statistics import StatsPlotter
        prep_GA_stats = StatsPlotter(self.res_path, self.participants)
        prep_GA_stats.GA_plots()

        # Magnitudes of DPOAE peaks
        conditions_sig, snrs_sig, mags_sig, delays_sig = self.prep_boxplot_ga(only_sig=True, avg_participant=False)
        conditions_delay, snrs_delay, mags_delay, delays_delay = self.prep_boxplot_ga(ga_delay=False, avg_participant=False)
        conditions_sig_Px, snrs_sig_Px, mags_sig_Px, delays_sig_Px = self.prep_boxplot_ga(only_sig=True, avg_participant=True)
        conditions_delay_Px, snrs_delay_Px, mags_delay_Px, delays_delay_Px = self.prep_boxplot_ga(ga_delay=True, avg_participant=True)

        np.save(stats_data_path + 'Mag_onlysig.npy', mags_sig, allow_pickle=True)
        np.save(stats_data_path + 'Mag_GAdelay.npy', mags_delay, allow_pickle=True)
        np.save(stats_data_path + 'Mag_onlysig_avgP.npy', mags_sig_Px, allow_pickle=True)
        np.save(stats_data_path + 'Mag_GAdelay_avgP.npy', mags_delay_Px, allow_pickle=True)

        np.save(stats_data_path + 'Delay_onlysig.npy', delays_sig, allow_pickle=True)
        np.save(stats_data_path + 'Delay_GAdelay.npy', delays_delay, allow_pickle=True)
        np.save(stats_data_path + 'Delay_onlysig_avgP.npy', delays_sig_Px, allow_pickle=True)
        np.save(stats_data_path + 'Delay_GAdelay_avgP.npy', delays_delay_Px, allow_pickle=True)

        # Attentional Modulation Coefficients
        amc_sig = self.attentional_modulation_coefficient(data_source='Mag', only_sig=True)
        amc_delay = self.attentional_modulation_coefficient(data_source='Mag', ga_delay=True)

        np.save(stats_data_path + 'AMC_onlysig.npy', amc_sig, allow_pickle=True)
        np.save(stats_data_path + 'AMC_GAdelay.npy', amc_delay, allow_pickle=True)

        # Cycle data
        cycles_sig = self.delay_cycles(delays_sig, only_sig=True, avg_participant=False)
        cycles_sig_Px = self.delay_cycles(delays_sig, only_sig=True, avg_participant=True)

        np.save(stats_data_path + 'Cycles_onlysig.npy', cycles_sig, allow_pickle=True)
        np.save(stats_data_path + 'Cycles_onlysig_avgP.npy', cycles_sig_Px, allow_pickle=True)

        # Count significant peaks data
        perc_sig_peaks, no_sig_peaks, no_all_peaks = self.count_significant_peaks(per_participant=False)
        perc_sig_peaks_Px, no_sig_peaks_Px, no_all_peaks_Px = self.count_significant_peaks(per_participant=True)

        np.save(stats_data_path + 'perc_sig_peaks.npy', perc_sig_peaks, allow_pickle=True)
        np.save(stats_data_path + 'no_sig_peaks.npy', no_sig_peaks, allow_pickle=True)
        np.save(stats_data_path + 'no_all_peaks.npy', no_all_peaks, allow_pickle=True)
        np.save(stats_data_path + 'perc_sig_peaks_perP.npy', perc_sig_peaks_Px, allow_pickle=True)
        np.save(stats_data_path + 'no_sig_peaks_perP.npy', no_sig_peaks_Px, allow_pickle=True)
        np.save(stats_data_path + 'no_all_peaks_perP.npy', no_all_peaks_Px, allow_pickle=True)

        return


                