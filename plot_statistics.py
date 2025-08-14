import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
from pathlib import Path
import scipy.stats as sts
from statsmodels.stats.multitest import multipletests
from collections.abc import Iterable
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

class StatsPlotter:
    """Class for plotting statistical analysis results.
    This class provides methods to create various statistical plots, including boxplots,
    histograms, and significance annotations.

    Parameters
    ----------
    main_path : str
        The path to the parent directory where the statistical analysis results from CrossCorrelationProcessor Class are stored.
    participants : list
        List of participant identifiers (e.g., ['P01', 'P02', ...]).

    Attributes
    ----------
    participants : list
        List of participant identifiers.
    main_path : str
        The path to the parent directory where the statistical analysis results are stored.
    stim_path : str
        Path to the stimuli directory.
    data_path : str
        Path to the data directory for statistical analysis results.
    plot_path : str
        Path to the directory where plots will be saved.
    save_svg : bool
        If True, saves plots additionally in SVG format; otherwise, saves only in PNG format.
    fontsize : int
        Font size for the plots.
    plotpad : int
        Padding for the plots.
    markersize : int
        Size of markers in the plots.
    median_linewidth : int
        Line width for the median line in boxplots.
    hatch_patterns : dict
        Dictionary containing hatch patterns for different conditions.
    color_palettes : dict
        Dictionary containing color palettes for different conditions.
    """

    def __init__(self, main_path, participants, save_svg=False):
        self.participants = participants
        self.main_path = Path(main_path)
        self.stim_path = self.main_path.parent.parent / 'Stimuli/'
        self.data_path = self.main_path / 'Data_Plots/'
        self.plot_path = self.main_path / 'Plots/'
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        
        self.save_svg = save_svg

        self.fontsize = 25
        self.plotpad = 20
        self.markersize = 10
        self.median_linewidth = 3

        plt.rcParams.update({
            'font.size': 18,          # General font size
            'axes.titlesize': 20,     # Title font size
            'axes.labelsize': 30,     # X and Y label font size
            'xtick.labelsize': 20,    # X-tick label size
            'ytick.labelsize': 20,    # Y-tick label size
            'legend.fontsize': 18,     # Legend font size
            'axes.linewidth': 1.5,      # Axes line width
            'xtick.major.size': 10,    # X-tick major size
            'ytick.major.size': 10,    # Y-tick major size
            'xtick.major.width': 2,  # X-tick major width
            'ytick.major.width': 2  # Y-tick major width
        })

        self.hatch_patterns = {'Atd': '',
                               'Ign': '//',
                               'BL': '..'}
        
        self.color_palettes = {('male', 'res'): '#a6dba0',
                          ('male', 'unres'): '#66c2a5',
                          ('female', 'res'): '#c2a5cf',
                          ('female', 'unres'): '#8da0cb'}
        
    def sort_conditions_wo_attention(self, conditions, data):
        """Sorts data per stimulus types for plotting, excluding attention conditions.
        
        Parameters
        ----------
        conditions : list
            List of condition names
        data : dict
            Dictionary with condition names as keys and data as values
        
        Returns
        -------
        ax_names : list
            List of axis names for plotting
        sorted_conditions : list
            List of sorted conditions based on stimulus type
        data_combined : dict
            Dictionary with combined cycle data across attentional conditions
        """
        conds_single_old = [cond for cond in conditions if 'Single Speaker' in cond and 'Ignored' not in cond]
        conds_competing_old = [cond for cond in conditions if 'Competing Speaker' in cond]
        conditions_old = conds_single_old + conds_competing_old

        new_conditions = ['Single Speaker female resolved',
                    'Single Speaker female unresolved',
                    'Single Speaker male resolved',
                    'Single Speaker male unresolved',
                    'Competing Speaker female resolved',
                    'Competing Speaker female unresolved',
                    'Competing Speaker male resolved',
                    'Competing Speaker male unresolved']

        data_combined = {cond: [] for cond in new_conditions}
        for new_cond in new_conditions:
            keywords = new_cond.split()
            keywords[1:3] = [f' {k} ' for k in keywords[1:3]]
            keywords[0] = f'{keywords[0]} '
            keywords[3] = f' {keywords[3]}'

            for cond in conditions_old:
                if all(keyword in cond for keyword in keywords):
                    if isinstance(data[cond], Iterable) and not isinstance(data[cond], str):
                        data_combined[new_cond].extend(data[cond])
                    else:
                        data_combined[new_cond].append(data[cond])

        conds_single = [cond for cond in new_conditions if 'Single' in cond]
        conds_competing = [cond for cond in new_conditions if 'Competing' in cond]

        ax_names_single = [' '.join(name.split(' ')[2:4]) for name in conds_single]
        ax_names_competing = [' '.join(name.split(' ')[2:4]) for name in conds_competing]
        
        replacements = {'Attended': 'Atd', 'Ignored': 'Ign', 'resolved': 'res', 'unresolved': 'unres', 'Baseline': 'BL', 'female': 'F', 'male': 'M'}
        ax_names_single = [re.sub(r'\b(?:' + '|'.join(replacements.keys()) + r')\b', lambda m: replacements[m.group()], item) for item in ax_names_single]
        ax_names_competing = [re.sub(r'\b(?:' + '|'.join(replacements.keys()) + r')\b', lambda m: replacements[m.group()], item) for item in ax_names_competing]
        
        ax_names = [ax_names_single, ax_names_competing]
        sorted_conditions = [conds_single, conds_competing]

        return ax_names, sorted_conditions, data_combined
    
    def remove_outliers(self, bp_data, bp, paired=False):
        """Removes outliers from boxplot data.
        
        Parameters
        ----------
        bp_data : list
            List of boxplot data for each condition
        bp : dict
            Dictionary containing boxplot elements
        paired : bool, optional
            If True, assumes paired data and removes outliers accordingly.
            False by default, indicating independent data
        
        Returns
        -------
        clean_bp_data : list
            List of boxplot data with outliers removed
        related_param : bool
            Indicates whether the data is related (paired) or not
            If paired, returns True; otherwise, returns False
        """
        if paired:
            clean_bp_data = []
            outliers = []
        else:
            clean_bp_data = bp_data.copy()

        for i, box in enumerate(bp_data):
            flier_vals = bp['fliers'][i].get_ydata()
            if len(flier_vals) == 0:
                continue
            else:
                if paired:
                    # save index of outliers to list
                    outliers.append([i for i, val in enumerate(box) if val in flier_vals])
                else:
                    npbox = np.array(box)
                    clean_data = npbox[~np.isin(npbox, flier_vals)]
                    clean_bp_data[i] = list(clean_data)
        
        if paired:
            # remove the outliers
            outliers = list(set([item for sublist in outliers for item in sublist]))
            for i, box in enumerate(bp_data):
                clean_bp_data.append([val for j, val in enumerate(box) if j not in outliers])

        if paired: 
            related_param = True
        else:
            related_param = False

        return clean_bp_data, related_param
    
    def remove_participants_with_nones(self, data):
        """Remove participants from the data who have None values in any condition.
        
        Parameters
        ----------
        data : dict
            Dictionary with condition names as keys and data as values.
            Each value should be a list containing data for that condition.
            
        Returns
        data : dict
            Dictionary with None values removed for participants who have None in any condition.    
        """
        idx_list = {cond: [] for cond in data.keys()}
        for cond in data.keys():
            for i, val in enumerate(data[cond]):
                if val is None:
                    idx_list[cond].append(i)
                    
        all_idx = [item for sublist in idx_list.values() for item in sublist]
        participans_idx = np.unique(all_idx)

        # remove the participants indicated in participans_idx from the data
        for cond in data.keys():
            data[cond] = [val for i, val in enumerate(data[cond]) if i not in participans_idx]

        return data
    
    def determine_significant_combinations(self, data, positions=None, related=False):
        """Determine significant combinations of data using statistical tests.
        
        Parameters
        ----------
        data : list of lists
            A list of lists containing the data for each condition. Each sublist should contain the data
            for a specific condition.
        positions : list, optional
            A list of positions corresponding to the conditions in `data`. If provided, the significant combinations
            will include these positions. Default is None. Positions are instead determined from the indices of the data.
        related : bool, optional
            If True, the data is considered related (e.g., paired samples). If False, the data is considered independent.
            Default is False.   
        
        Returns
        -------
        sig_combinations : list
            A list of significant combinations, where each sublist comprises a tuple containing the condition indices (or the positions 
            if positions were given) and the p-value of the statistical test.
        """
        sig_combinations = []
        if positions is not None:
            combinations_positions = [(positions[i], positions[i + j]) for j in reversed(range(1, len(positions))) for i in range(len(positions) - j)]

        ls = list(range(1, len(data) + 1))
        combinations = [(ls[i], ls[i+j]) for j in reversed(ls) for i in range((len(ls) - j))]
        for c, combination in enumerate(combinations):
            data1 = data[combination[0] - 1]
            data2 = data[combination[1] - 1]

            # test for normality
            _, p_norm1 = sts.shapiro(data1)
            _, p_norm2 = sts.shapiro(data2)

            alpha = 0.05
            if related: # if the data is related, use the paired t-test
                if p_norm1 > alpha and p_norm2 > alpha:
                    stats = sts.ttest_rel(data1, data2, alternative='two-sided')
                else:
                    stats = sts.wilcoxon(data1, data2, alternative='two-sided')
            else: # if the data is not related, use the independent t-test
                if p_norm1 > alpha and p_norm2 > alpha: # fail to reject H0, data is Gaussian -> paired t-test
                    stats = sts.ttest_ind(data1, data2, alternative='two-sided')
                else: # reject H0, data is not Gaussian -> Mann-Whitney U test
                    stats = sts.mannwhitneyu(data1, data2, alternative='two-sided')

            p = stats.pvalue
            if positions is not None:
                sig_combinations.append([combinations_positions[c], p])
            else:
                sig_combinations.append([combination, p])
        
        return sig_combinations

    def flatten(self, nested):
        """Flatten nested list and store the shape for reconstruction.

        Parameters
        ----------
        nested : list
            A nested list structure to be flattened.
            
        Returns
        -------
        flat_list : list
            A flat list containing all elements from the nested list.
        shape : list
            A list representing the shape of the nested structure.
        """
        flat_list = []
        shape = []

        def _flatten(sublist):
            # Detect and preserve leaf node structure like [(a, b), val]
            if isinstance(sublist, list) and not (len(sublist) == 2 and isinstance(sublist[0], tuple)):
                current_shape = []
                for item in sublist:
                    result = _flatten(item)
                    current_shape.append(result)
                shape.append(current_shape)
                return current_shape
            else:
                flat_list.append(sublist)
                return None

        _flatten(nested)
        return flat_list, shape[-1]

    def reconstruct(self, flat_list, shape):
        """Reconstruct flat_list to nested structure using shape.

        Parameters
        ----------
        flat_list : list
            A flat list to be reconstructed.
        shape : list
            A list representing the shape of the nested structure.
        
        Returns
        -------
        reconstructed : list
            A nested list reconstructed from the flat list according to the provided shape.
        """
        it = iter(flat_list)
        
        def _reconstruct(s):
            if isinstance(s, list):
                return [_reconstruct(sub_s) for sub_s in s]
            else:
                return next(it)
        
        return _reconstruct(shape)

    def perform_fdr_correction(self, sig_combinations):
        """Perform FDR correction on significant combinations.
        
        Parameters
        ----------
        sig_combinations : list
            A list of significant combinations, where each combination is a list of [(condition indices), p-value].
        
        Returns
        -------
        sig_combinations : list
            A list of significant combinations with corrected p-values.
        """
        nested = any(isinstance(i, list) for i in sig_combinations)
        # perform FDR correction if there are significant combinations
        if len(sig_combinations) == 0:
            return sig_combinations
        # check if sig_combinations is a nested list
        else:
            if nested:
                sig_combinations, shape = self.flatten(sig_combinations)

            pvals = [sig_comb[1] for sig_comb in sig_combinations]
            rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

            for i, sig_comb in enumerate(sig_combinations):
                sig_comb[1] = pvals_corrected[i]

            if nested:
                sig_combinations = self.reconstruct(sig_combinations, shape)

            return sig_combinations
    
    def draw_significance(self, bp, sig_combinations, dv=0, distance=0.5, color='black'):
        """Draws significance bars on a boxplot.

        Parameters
        ----------
        bp : dict
            Dictionary containing boxplot elements.
        sig_combinations : list
            List of significant combinations, where each combination is a list of [(indices), p-value].
        dv : float, optional
            Vertical offset to top of whiskers, default is 0.
        distance : float, optional
            Distance between significance bars, default is 0.5.
        color : str, optional
            Color of the significance bars, default is 'black'.

        Returns
        -------
        None
        """

        # get y-positions of whiskers
        whisker_heights = []
        for whisker in bp['whiskers']:
            y_data = whisker.get_ydata()
            whisker_heights.append(y_data[1])
        
        bottom, top = min(whisker_heights), max(whisker_heights) + dv
        y_range = top - bottom

        for l, sublist in enumerate(sig_combinations):
            # remove all combinations with p > 0.05
            sc_level = [sig_comb for sig_comb in sublist if sig_comb[1] < 0.05]

            for i, sig_comb in enumerate(sc_level):
                x1 = sig_comb[0][0] - 1 + 0.02
                x2 = sig_comb[0][1] - 1 - 0.02
                level = len(sig_combinations) - l
                bar_height = (y_range * distance * level) + top
                bar_tips = bar_height - (y_range * 0.05)
                plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], c=color, lw=1.5)

                p = sig_comb[1]
                if p < 0.001:
                    sig_symbol = '***'
                elif p < 0.01:
                    sig_symbol = '**'
                elif p < 0.05:
                    sig_symbol = '*'
                
                text_height = bar_height + (y_range * 0.002)
                plt.text((x1 + x2) / 2, text_height, sig_symbol, ha='center', va='bottom', c=color, fontsize=self.fontsize)
        return
        
    def gaussian(self, x, a, mu, sigma):
        """Gaussian function for curve fitting.

        Parameters
        ----------
        x : array-like
            Input data points.
        a : float
            Amplitude of the Gaussian.
        mu : float
            Mean (center) of the Gaussian.
        sigma : float
            Standard deviation (width) of the Gaussian.

        Returns
        -------
        float
            Evaluated Gaussian function at x.
        """
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    def identify_peak_area(self):
        """Identifies the peak area of the Gaussian fit for single speaker conditions.
        This method loads the means of the Gaussian fits for single speaker conditions for the female voice data,
        fits a Gaussian to the data, and returns the parameters of the fitted Gaussian.

        Returns
        -------
        gauss_params : dict
            Dictionary containing Gaussian parameters (amplitude, mean, standard deviation) for each condition.
        """
        data = np.load(self.data_path / f'GA_means.npy', allow_pickle=True).item()
        conditions = data.keys()
        # modify the following line to filter conditions to include
        single_conds = [cond for cond in conditions if 'Single' in cond and 'Ignored' not in cond and ' female ' in cond]
        
        gauss_params = {}
        for cond in single_conds:
            x = data[cond][0, :]
            x_ms = x / 44100 * 1000
            y = data[cond][3, :]

            idx_gauss = np.where((x_ms >= -15) & (x_ms <= 15))[0]
            x = x[idx_gauss]
            y = y[idx_gauss]
            x_ms = x_ms[idx_gauss]

            # fit a Gaussian to the data
            popt, _ = curve_fit(self.gaussian, x, y, p0=[max(y), 0, 1])
            a_fit, mu_fit, sigma_fit = popt # peak height, peak position, peak width in frames
            gauss_params[cond] = (a_fit, mu_fit, sigma_fit) # in frames
        
        return gauss_params
    
    def determine_noise_level(self, env, lag, return_env_noise=False, fs=44100):
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
        l, h = l / 1000 * fs, h / 1000 * fs # frames
        idx_noise = np.where(((lag < -l) & (lag > -h)) | ((lag > l) & (lag < h)))[0]
        env_noise = env[idx_noise]
        noise_level = np.real(np.mean(env_noise))

        if return_env_noise:
            return noise_level, env_noise
        else:
            return noise_level

    def percentage_sig_peaks(self):
        """Generates barplots of the percentage of significant peaks for different conditions."""

        no_sig = np.load(self.data_path / f'no_sig_peaks.npy', allow_pickle=True).item()
        no_all = np.load(self.data_path / f'no_all_peaks.npy', allow_pickle=True).item()

        save_path = self.plot_path / f'Percentage_sig_peaks/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        conditions = list(no_sig.keys())
        conditions.sort()
        replacements = {'Attended': 'Atd', 'Ignored': 'Ign', 'resolved': 'res', 'unresolved': 'unres', 'Baseline': 'BL', 'female': 'F', 'male': 'M'}
        
        conds_single = [cond for cond in conditions if 'Single' in cond and 'Ignored' not in cond]
        ax_names_single = [' '.join(name.split(' ')[-4:-1]) for name in conds_single]
        ax_names_single = [re.sub(r'\b(?:' + '|'.join(replacements.keys()) + r')\b', lambda m: replacements[m.group()], item) for item in ax_names_single]

        competing = []
        ax_competing = []
        for g in ['male', 'female']:
            conds_competing = [cond for cond in conditions if 'Competing' in cond and f' {g} ' in cond]
            conds_competing = conds_competing[0:2] + conds_competing[4:6] + conds_competing[2:4]
            ax_competing_names = [' '.join(name.split(' ')[-4:-1]) for name in conds_competing]
            ax_competing_names = [re.sub(r'\b(?:' + '|'.join(replacements.keys()) + r')\b', lambda m: replacements[m.group()], item) for item in ax_competing_names]
            competing.append(conds_competing)
            ax_competing.append(ax_competing_names)

        ax_names = [ax_names_single] + ax_competing
        sorted_conditions = [conds_single] + competing

        ax_names, sorted_conditions, no_sig = self.sort_conditions_wo_attention(conditions, no_sig)
        _, _, no_all = self.sort_conditions_wo_attention(conditions, no_all)

        perc = {cond: None for cond in no_sig.keys()}
        for i, cond in enumerate(no_sig.keys()):
            perc[cond] = np.sum(no_sig[cond]) / np.sum(no_all[cond])

        for ax_name, cond_list in zip(ax_names, sorted_conditions):
            speaker_type = 'Single' if 'Single' in cond_list[0] else 'Comp'

            harms = []
            colors = []
            voices = []
            for i, name in enumerate(ax_name):
                voice = 'female' if 'F' in name else 'male'
                voices.append(voice)
                harm = 'res' if ' res' in name else 'unres'
                harms.append(harm)
                colors.append(self.color_palettes[(voice, harm)])

            fig, ax = plt.subplots(figsize=(7, 5))

            hist_data = [perc[cond]*100 for cond in cond_list]
            ax.bar(ax_name, hist_data, color=colors, alpha=1, linewidth=1.5, width=0.7)

            # uncomment the following lines to add number of significant peaks on top of the bars
            # numbers = [str(np.sum(no_sig[cond])) for cond in cond_list]
            # for x, text in enumerate(numbers):
            #     plt.text(x, 1.1, text, ha='center', va='bottom', fontsize=14)
            
            ax.tick_params(labelsize=self.fontsize)
            ax.set_xticks(np.arange(len(cond_list)))
            ax_name_plot = ['-'.join(name.split(' ')) for name in ax_name]
            ax.set_xticklabels(ax_name_plot, fontsize=self.fontsize, rotation=45)

            yborder = [0, 110]
            ax.set_ylim(yborder)
            ax.set_yticks(np.arange(0, yborder[1], 25))

            ax.set_ylabel('Percentage', labelpad=15, fontsize=self.fontsize)
            ax.set_title(f'{speaker_type} Speaker', pad=15, fontsize=self.fontsize)

            save_name = f'Barplot_{speaker_type}'

            sns.despine()
            plt.tight_layout()
            plt.savefig(save_path / f"{save_name}.png", dpi=1200)
            if self.save_svg:
                ax.title.set_visible(False)
                plt.savefig(save_path / f'{save_name}.svg', dpi=1200, format='svg', bbox_inches='tight', transparent=True)
            plt.close()

        return
    

    def single_speaker_magnitudes(self, paired_ttest=False, freq_scaling=False):
        """Generates boxplots of magnitudes for single speaker conditions and saves them as PNG and SVG files.
        
        Parameters
        ----------
        paired_ttest : bool, optional
            If True, performs paired t-tests on the data. Default is False.
        freq_scaling : bool, optional
            If True, scales the x-axis based on the mean frequency of the DP. Default is False.
        
        Returns
        -------
        None
        """
        filename = 'Mag_GAdelay_avgP'
        data = np.load(self.data_path / f'{filename}.npy', allow_pickle=True).item()
        
        save_path = self.plot_path / f'Boxplots_Mags_SingleSpeaker/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        conditions = data.keys()

        single_conditions = [cond for cond in conditions if 'Single' in cond and 'Ignored' not in cond]
        dp_freqs = []

        for cond in single_conditions:
            voice = 'female' if ' female ' in cond else 'male'
            harm = 'resolved' if ' resolved ' in cond else 'unresolved'
            
            if harm == 'resolved':
                nharm = 'n5' if voice == 'female' else 'n4'
            else:
                nharm = 'n12'

            freq_info = pd.read_csv(self.stim_path / f'pyin_mean_freq_Polarnacht_{voice}.csv')
            freq_info = dict(zip(freq_info['File'], freq_info['Mean Frequency [Hz]']))

            freq = []
            for key in freq_info.keys():
                if nharm in key and 'single' in key:
                    freq.append(freq_info[key])
            
            dp_freqs.append(np.mean(freq))
        
        # sort dp_freqs and single_conditions ascending
        dp_freqs, single_conditions = zip(*sorted(zip(dp_freqs, single_conditions), key=lambda x: x[0]))
        ax_names = [str(int(np.round(f, -1))) for f in dp_freqs]
        colors = []        
        for i, name in enumerate(single_conditions):
            voice = 'female' if ' female ' in name else 'male'
            harm = 'res' if ' resolved ' in name else 'unres'
            colors.append(self.color_palettes[(voice, harm)])
        
        bp_data = [data[cond] for cond in single_conditions]

        if freq_scaling:
            positions = [2, 10, 12, 20]
            fig, ax = plt.subplots(figsize=(11, 6))
            bp = ax.boxplot(bp_data, patch_artist=True, positions=(positions), showfliers=True, widths=1.5)
        else:
            fig, ax = plt.subplots(figsize=(6, 7))
            bp = ax.boxplot(bp_data, patch_artist=True, positions=(np.arange(len(single_conditions))), showfliers=True, widths=0.6)

        # apply styles
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(self.median_linewidth)
        
        for w in bp['whiskers']:
            w.set_color('black')
            w.set_linewidth(1.5)
        
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.5)

        for flier in bp['fliers']:
            flier.set_markerfacecolor('black')
            flier.set_markeredgecolor('none')
            flier.set_alpha(0.5)
            flier.set_markersize(self.markersize)
            flier.set_marker('d')

        clean_bp_data, related_param = self.remove_outliers(bp_data, bp, paired=paired_ttest)

        if paired_ttest:
            N = len(clean_bp_data[0])
            for i in range(len(clean_bp_data)):
                if len(clean_bp_data[i]) != N:
                    print(f"Warning: Condition {single_conditions[i]} has a different number of data points than the others.")
                    break
        else:
            N = [len(data) for data in clean_bp_data]

        sig_combs = []
        for c in [(1, 4), (2, 4), (1, 3)]:
            comp = [clean_bp_data[c[0]-1], clean_bp_data[c[1]-1]]
            sig_combs.append(self.determine_significant_combinations(comp, [c[0], c[1]], related=related_param))

        level3 = []
        for i in range(1, len(clean_bp_data)):
            level3 += self.determine_significant_combinations([clean_bp_data[i-1], clean_bp_data[i]], [i, i+1], related=related_param)

        sig_combs.append(level3)

        sig_combs_fdr = self.perform_fdr_correction(sig_combs)
        self.draw_significance(bp, sig_combs_fdr, dv=0.0003, distance=0.2)

        # clean look
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_size(20)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
        ax.tick_params(axis='y', labelsize=self.fontsize)

        ax.set_xticklabels(ax_names, fontsize=self.fontsize)    
        ax.set_xlabel(r'$\bar{f}_{DP}$ [Hz]', labelpad=self.plotpad, fontsize=self.fontsize)

        yborder = [-0.0005, 0.005]
        ax.set_ylim(yborder)
        ax.set_yticks(np.arange(0, yborder[1], 0.001))
        ax.set_ylabel(f'Amplitude', labelpad=self.plotpad, fontsize=self.fontsize)
        
        ax.set_title(f'Single Speaker', pad=30, fontsize=self.fontsize)
        
        sns.despine()
        plt.tight_layout()

        if freq_scaling:
            # save as png and svg
            plt.savefig(save_path / f"Box_Mag_SingleSpeaker_frequencies.png", dpi=1200)
            ax.title.set_visible(False)
            if self.save_svg:
                plt.savefig(save_path / f"Box_Mag_SingleSpeaker_frequencies.svg", dpi=1200, format='svg', bbox_inches='tight', transparent=True)
        else:   
            plt.savefig(save_path / f"Box_Mag_SingleSpeaker.png", dpi=1200)
            ax.title.set_visible(False)
            if self.save_svg:
                plt.savefig(save_path / f"Box_Mag_SingleSpeaker.svg", dpi=1200, format='svg', bbox_inches='tight', transparent=True)
            
        plt.close()

        return
    
    def single_speaker_delay(self):
        """Generates boxplots of delays for single speaker conditions and saves them as PNG and SVG files."""

        data = np.load(self.data_path / f'Delay_onlysig_avgP.npy', allow_pickle=True).item()

        related_param = False   
        save_path = self.plot_path / f'Boxplots_Delays_SingleSpeaker/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        conditions = data.keys()

        single_conditions = [cond for cond in conditions if 'Single' in cond and 'Ignored' not in cond]
        dp_freqs = []

        for cond in single_conditions:
            voice = 'female' if ' female ' in cond else 'male'
            harm = 'resolved' if ' resolved ' in cond else 'unresolved'
            
            if harm == 'resolved':
                nharm = 'n5' if voice == 'female' else 'n4'
            else:
                nharm = 'n12'

            freq_info = pd.read_csv(self.stim_path / f'pyin_mean_freq_Polarnacht_{voice}.csv')
            freq_info = dict(zip(freq_info['File'], freq_info['Mean Frequency [Hz]']))

            freq = []
            for key in freq_info.keys():
                if nharm in key and 'single' in key:
                    freq.append(freq_info[key])
            
            dp_freqs.append(np.mean(freq))
        
        # sort dp_freqs and single_conditions ascending
        dp_freqs, single_conditions = zip(*sorted(zip(dp_freqs, single_conditions), key=lambda x: x[0]))
        ax_names = [str(int(np.round(f, -1))) for f in dp_freqs]
        colors = []        
        for i, name in enumerate(single_conditions):
            voice = 'female' if ' female ' in name else 'male'
            harm = 'res' if ' resolved ' in name else 'unres'
            colors.append(self.color_palettes[(voice, harm)])
        
        bp_data = [data[cond] for cond in single_conditions]
        bp_data = [list(filter(lambda x: x is not None, item)) for item in bp_data] # remove None values if present

        fig, ax = plt.subplots(figsize=(6, 7))
        bp = ax.boxplot(bp_data, patch_artist=True, positions=(np.arange(len(single_conditions))), showfliers=True, widths=0.6)

        # apply styles
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(self.median_linewidth)
        
        for w in bp['whiskers']:
            w.set_color('black')
            w.set_linewidth(1.5)
        
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.5)

        for flier in bp['fliers']:
            flier.set_markerfacecolor('black')
            flier.set_markeredgecolor('none')
            flier.set_alpha(0.5)
            flier.set_markersize(self.markersize)
            flier.set_marker('d')

        clean_bp_data, related_param = self.remove_outliers(bp_data, bp, paired=related_param)

        sig_combs = []
        for c in [(1, 4), (2, 4), (1, 3)]:
            comp = [clean_bp_data[c[0]-1], clean_bp_data[c[1]-1]]
            sig_combs.append(self.determine_significant_combinations(comp, [c[0], c[1]], related=related_param))

        level3 = []
        for i in range(1, len(clean_bp_data)):
            level3 += self.determine_significant_combinations([clean_bp_data[i-1], clean_bp_data[i]], [i, i+1], related=related_param)

        sig_combs.append(level3)

        sig_combs_fdr = self.perform_fdr_correction(sig_combs)
        self.draw_significance(bp, sig_combs_fdr, dv=0, distance=0.12)

        # clean look
        ax.yaxis.offsetText.set_size(20)
        ax.tick_params(axis='y', labelsize=self.fontsize)

        ax.set_xticklabels(ax_names, fontsize=self.fontsize)    

        yborder = [-2.5, 7]
        ax.set_ylim(yborder)
        ax.set_yticks(np.arange(-2, 5, 1))
        ax.set_ylabel(f'Delay [ms]', labelpad=self.plotpad, fontsize=self.fontsize)
        ax.set_xlabel(r'$\bar{f}_{DP}$ [Hz]', labelpad=self.plotpad, fontsize=self.fontsize)

        sns.despine()
        plt.tight_layout()

        plt.savefig(save_path / f"Box_Delay_SingleSpeaker.png", dpi=1200)
        ax.title.set_visible(False)
        if self.save_svg:
            plt.savefig(save_path / f"Box_Delay_SingleSpeaker.svg", dpi=1200, format='svg', bbox_inches='tight', transparent=True)
            
        plt.close()

        return
    
    def GA_plots(self):
        """Generates cross-correlation plots for grand average data and saves them as PNG and SVG files."""

        data = np.load(self.data_path / f'GA_means.npy', allow_pickle=True).item()
        
        save_path = self.plot_path / f'Plots_GA/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gauss_params = self.identify_peak_area()
        peak_pos = [gauss_params[cond][1] for cond in gauss_params.keys()]
        peak_stds = [gauss_params[cond][2] for cond in gauss_params.keys()]

        print(f'\nMEAN: {np.mean(peak_pos) / 44100 * 1000} ± {np.mean(peak_stds) / 44100 * 1000} ms')
        print(f'MEDIAN: {np.median(peak_pos) / 44100 * 1000} ± {np.median(peak_stds) / 44100 * 1000} ms\n')

        expected_peak_pos = int(np.round(np.mean(peak_pos))) # in frames
        peak_border = int(np.round(np.mean(peak_stds))) # in frames

        # write to csv with columns 'expected_peak_pos' and 'peak_border'
        df = pd.DataFrame({'Expected peak position [frames]': [expected_peak_pos], 'Peak area border [frames]': [peak_border]})
        df.to_csv(self.data_path / f'GA_peak_pos.csv', index=False)

        snrs = {}
        for cond, mean in data.items():
            speaker_type = 'Single Speaker' if 'Single' in cond else 'Competing Speaker'
            voice = 'female' if ' female ' in cond else 'male'
            harm = 'res' if ' resolved ' in cond else 'unres'
            label = 'F' if voice == 'female' else 'M' 
            label = label + ' ' + harm
            attention = 'Attended video' if 'Baseline' in cond else 'Ignored' if 'Ignored' in cond else 'Attended'
            color = self.color_palettes[(voice, harm)]

            lag = mean[0, :]
            lag_ms = lag / 44100 * 1000
            env = mean[3, :]

            # find peak in ± peak_border
            idx_min = np.argmin(np.abs(lag - (expected_peak_pos - peak_border)))
            idx_max = np.argmin(np.abs(lag - (expected_peak_pos + peak_border)))
            env_range = env[idx_min:idx_max + 1]
            mag_peak, idx = np.max(env_range), np.argmax(env_range)
            idx = idx + idx_min
            lag_peak = lag[idx]

            noise_level, env_noise = self.determine_noise_level(env, lag, return_env_noise=True)

            if mag_peak > np.max(env_noise):
                sig_peak = True
            else:
                sig_peak = False

            lag_peak = int(lag_peak)
            lag_peak_ms = lag_peak / 44100 * 1000
            snrs[cond] = 20 * np.log10(mag_peak / noise_level)

            if speaker_type == 'Single Speaker':
                fig, ax = plt.subplots(figsize=(7, 7))
                yborder = [0, 0.0025]   
                stepsize = 0.001  
            else:
                fig, ax = plt.subplots(figsize=(7, 7))
                yborder = [0, 0.0015]
                stepsize = 0.0005
            xborder = 30

            ax.plot(lag_ms, env, color=color, linewidth=5, label=label)

            if sig_peak:
                if lag_peak_ms < 0:
                    lag_peak_ms = 0.0
                ax.plot([lag_peak_ms, lag_peak_ms], [yborder[0], mag_peak + 0.0002], c='black', linestyle='--', alpha=0.5, linewidth=2)
                ax.text(lag_peak_ms, mag_peak + 0.0003, f'{np.round(lag_peak_ms, 1)} ms', ha='center', fontsize=self.fontsize)

            # clean look
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -3))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.offsetText.set_size(20)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
            ax.tick_params(axis='y', labelsize=self.fontsize)
            
            ax.set_ylim(yborder)
            ax.set_ylabel('Env. Cross-corr.', labelpad=self.plotpad, fontsize=self.fontsize)
            ax.set_yticks(np.arange(yborder[0], yborder[1], stepsize))
            
            ticks = np.arange(-xborder + 10, xborder - 10 + 1, 10)
            ax.set_xlim(-xborder, xborder)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, fontsize=self.fontsize)
            ax.set_xlabel('Delay [ms]', labelpad=self.plotpad, fontsize=self.fontsize)
            
            if speaker_type == 'Single Speaker':
                ax.set_title(f'{voice.capitalize()} {harm}olved', pad=30, fontsize=self.fontsize)
            else:
                ax.set_title(f'{attention.capitalize()} {voice} {harm}olved', pad=30, fontsize=self.fontsize)

            sns.despine()
            plt.tight_layout()
            plt.savefig(save_path / f"GA_{speaker_type.split(' ')[0]}_{attention}_{voice}_{harm}.png", dpi=1200)
            if self.save_svg:
                plt.savefig(save_path / f"GA_{speaker_type.split(' ')[0]}_{attention}_{voice}_{harm}.svg", dpi=1200, format='svg', bbox_inches='tight', transparent=True)
            plt.close()

        # save snrs to csv
        snr_df = pd.DataFrame.from_dict(snrs, orient='index', columns=['SNR [dB]'])
        snr_df.index.name = 'Condition'
        snr_df.to_csv(self.data_path / f'GA_SNRs.csv')

        return
    
    def competing_speaker_magnitudes(self, paired_ttest=True):
        """Generates boxplots of magnitudes for competing speaker conditions and saves them as PNG and SVG files.

        Parameters
        ----------
        paired_ttest : bool, optional
            If True, performs paired t-tests on the data. Default is True.
        """

        filename = 'Mag_GAdelay_avgP'
        data = np.load(self.data_path / f'{filename}.npy', allow_pickle=True).item()
        
        save_path = self.plot_path / f'Boxplots_Mags_CompetingSpeaker/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        conditions = list(data.keys())
        conditions.sort()

        replacements = {'Attended': 'Atd', 'Ignored': 'Ign', 'resolved': 'res', 'unresolved': 'unres', 'Baseline': 'BL', 'female': 'F', 'male': 'M'}

        competing = []
        ax_competing = []
        for g in ['male', 'female']:
            for h in ['resolved', 'unresolved']:
                conds_competing = [cond for cond in conditions if 'Competing' in cond and f' {g} ' in cond and f' {h} ' in cond]
                conds_competing = [conds_competing[0], conds_competing[2], conds_competing[1]]
                ax_competing_names = [' '.join(name.split(' ')[-4:-1]) for name in conds_competing] 
                ax_competing_names = [re.sub(r'\b(?:' + '|'.join(replacements.keys()) + r')\b', lambda m: replacements[m.group()], item) for item in ax_competing_names]
                competing.append(conds_competing)
                ax_competing.append(ax_competing_names)
        
        ax_names = ax_competing
        sorted_conditions = competing

        sig_heights = {'Mres': (0.0004, 0.2), 'Munres': (-0.0002, 0.2), 'Fres': (-0.00001, 0.3), 'Funres': (0, 0)}

        for ax_name, cond_list in zip(ax_names, sorted_conditions):
            # replace Ign. with respective attention condition
            ax_name_sorted = ax_name.copy()

            voice = 'female' if 'F' in ax_name[0] else 'male'
            sub = ''.join(ax_name[0].split(' ')[1:3])
            
            if voice == 'female':
                ax_name_sorted[1] = 'Atd M ' + ax_name_sorted[1].split(' ')[-1]
            else:
                ax_name_sorted[1] = 'Atd F ' + ax_name_sorted[1].split(' ')[-1]

            ax_name_sorted, ax_name, cond_list = zip(*sorted(zip(ax_name_sorted, ax_name, cond_list), key=lambda x: x[0]))

            harms = []
            colors = []
            hatches = []
            for i, name in enumerate(ax_name):
                harms.append(name.split(' ')[-1])
                colors.append(self.color_palettes[(voice, harms[i])])
                hatches.append(self.hatch_patterns[name.split(' ')[0]])

            if len(np.unique(harms)) == 1:
                harm = harms[0] + 'olved'
            else:
                harm = ''

            data_dict = {cond: data[cond] for cond in cond_list}
            data_cut = self.remove_participants_with_nones(data_dict)
            bp_data = [data_cut[cond] for cond in cond_list]

            fig, ax = plt.subplots(figsize=(5, 5))
            bp = ax.boxplot(bp_data, patch_artist=True, positions=np.arange(len(cond_list)), showfliers=True, widths=0.6)

            # apply styles
            for patch, color, hatch in zip(bp['boxes'], colors, hatches):
                patch.set_facecolor(color)
                patch.set_hatch(hatch)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)

            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(self.median_linewidth)
            
            for w in bp['whiskers']:
                w.set_color('black')
                w.set_linewidth(1.5)

            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.5)

            for flier in bp['fliers']:
                flier.set_markerfacecolor('black')
                flier.set_markeredgecolor('none')
                flier.set_alpha(0.5)
                flier.set_markersize(self.markersize)
                flier.set_marker('d')

            clean_bp_data, related_param = self.remove_outliers(bp_data, bp, paired=paired_ttest)

            if paired_ttest:
                N = len(clean_bp_data[0])
                for i in range(len(clean_bp_data)):
                    if len(clean_bp_data[i]) != N:
                        print(f"Warning: Condition {cond_list[i]} has a different number of data points than the others.")
                        break
            else:
                N = [len(data) for data in clean_bp_data]

            # clean look
            replacements_final = {'Atd': 'Att.', 'BL': 'Att. V'}
            ax_name_short = [' '.join(name.split(' ')[0:1]) if i==2 else ' '.join(name.split(' ')[0:2]) for i, name in enumerate(ax_name_sorted)]
            ax_name_plot = [re.sub(r'\b(?:' + '|'.join(replacements_final.keys()) + r')\b', lambda m: replacements_final[m.group()], item) for item in ax_name_short]
            
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -3))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.offsetText.set_size(20)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
            ax.tick_params(axis='y', labelsize=self.fontsize)

            ax.set_xticks(np.arange(len(cond_list)))
            ax.set_xticklabels(ax_name_plot, fontsize=self.fontsize)
            ax.set_xlabel('Condition', labelpad=15, fontsize=self.fontsize)
                
            if voice == 'female' and harm == 'unresolved':
                yborder = [-0.0001, 0.0011]
                stepsize = 0.0005
            elif voice == 'female' and harm == 'resolved':
                yborder = [-0.0001, 0.0015]
                stepsize = 0.0005
            else:
                yborder = [-0.0001, 0.0034]
                stepsize = 0.0005

            ax.set_ylim(yborder)
            yticks = np.arange(0, yborder[1], stepsize)
            ax.set_yticks(yticks)

            ax.set_ylabel(f'Amplitude', labelpad=15, fontsize=self.fontsize)
            
            sig_combs = []
            for c in [(1, 3)]:
                comp = [clean_bp_data[c[0]-1], clean_bp_data[c[1]-1]]
                sig_combs.append(self.determine_significant_combinations(comp, [c[0], c[1]], related=related_param))

            level2 = []
            for i in range(1, len(clean_bp_data)):
                level2 += self.determine_significant_combinations([clean_bp_data[i-1], clean_bp_data[i]], [i, i+1], related=related_param)
            sig_combs.append(level2)

            sig_combs_fdr = self.perform_fdr_correction(sig_combs)
            self.draw_significance(bp, sig_combs_fdr, dv=sig_heights[sub][0], distance=sig_heights[sub][1])

            ax.set_title(f'CS {voice} {harm}', pad=30, fontsize=self.fontsize)

            # calculate ratio of amplitudes
            amp_ratio_FM = np.mean(clean_bp_data[0]) / np.mean(clean_bp_data[1])
            amp_ratio_FM_dB = 20 * np.log10(amp_ratio_FM)

            amp_ratio_FV = np.mean(clean_bp_data[0]) / np.mean(clean_bp_data[2])
            amp_ratio_FV_dB = 20 * np.log10(amp_ratio_FV)

            amp_ratio_MV = np.mean(clean_bp_data[1]) / np.mean(clean_bp_data[2])
            amp_ratio_MV_dB = 20 * np.log10(amp_ratio_MV)

            print(f'Ratio {voice} {harm}:\n') 
            print(f'{amp_ratio_FM} (Atd F/ Atd M) --- {amp_ratio_FM_dB} dB (Atd F/ Atd M)\n')
            print(f'{amp_ratio_FV} (Atd F/ Atd V) --- {amp_ratio_FV_dB} dB (Atd F/ Atd V)\n')
            print(f'{amp_ratio_MV} (Atd M/ Atd V) --- {amp_ratio_MV_dB} dB (Atd M/ Atd V)\n')
            print('--------------------------\n')

            sns.despine()
            plt.tight_layout()
            plt.savefig(save_path / f"Box_{voice}_{harm}.png", dpi=1200)
            if self.save_svg:
                ax.title.set_visible(False)
                plt.savefig(save_path / f"Box_{voice}_{harm}.svg", dpi=1200, format='svg', bbox_inches='tight', transparent=True)
            plt.close()

        return
    
    def competing_speaker_delay(self):
        """Generates boxplots of delays for competing speaker conditions and saves them as PNG and SVG files."""

        data = np.load(self.data_path / f'Delay_onlysig_avgP.npy', allow_pickle=True).item()

        save_path = self.plot_path / f'Boxplots_Delay_CompetingSpeaker/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        related_param = False
        
        conditions = list(data.keys())
        conditions.sort()

        replacements = {'Attended': 'Atd', 'Ignored': 'Ign', 'resolved': 'res', 'unresolved': 'unres', 'Baseline': 'BL', 'female': 'F', 'male': 'M'}
        competing = []
        ax_competing = []

        competing = []
        ax_competing = []
        for g in ['male', 'female']:
            for h in ['resolved', 'unresolved']:
                conds_competing = [cond for cond in conditions if 'Competing' in cond and f' {g} ' in cond and f' {h} ' in cond]
                conds_competing = [conds_competing[0], conds_competing[2], conds_competing[1]]
                ax_competing_names = [' '.join(name.split(' ')[-4:-1]) for name in conds_competing] 
                ax_competing_names = [re.sub(r'\b(?:' + '|'.join(replacements.keys()) + r')\b', lambda m: replacements[m.group()], item) for item in ax_competing_names]
                competing.append(conds_competing)
                ax_competing.append(ax_competing_names)
        
        ax_names = ax_competing
        sorted_conditions = competing

        for ax_name, cond_list in zip(ax_names, sorted_conditions):
            ax_name_sorted = ax_name.copy()
            voice = 'female' if 'F' in ax_name[0] else 'male'

            if voice == 'female':
                ax_name_sorted[1] = 'Atd M ' + ax_name_sorted[1].split(' ')[-1]
            else:
                ax_name_sorted[1] = 'Atd F ' + ax_name_sorted[1].split(' ')[-1]

            ax_name_sorted, ax_name, cond_list = zip(*sorted(zip(ax_name_sorted, ax_name, cond_list), key=lambda x: x[0]))

            harms = []
            colors = []
            hatches = []
            for i, name in enumerate(ax_name):
                harms.append(name.split(' ')[-1])
                colors.append(self.color_palettes[(voice, harms[i])])
                hatches.append(self.hatch_patterns[name.split(' ')[0]])

            if len(np.unique(harms)) == 1:
                harm = harms[0] + 'olved'
            else:
                harm = ''

            if related_param:
                data_dict = {cond: data[cond] for cond in cond_list}
                data_cut = self.remove_participants_with_nones(data_dict)
                bp_data = [data_cut[cond] for cond in cond_list]
            else:
                bp_data = [data[cond] for cond in cond_list]
                bp_data = [list(filter(lambda x: x is not None, item)) for item in bp_data] # remove None values if present

            fig, ax = plt.subplots(figsize=(5, 5))
            bp = ax.boxplot(bp_data, patch_artist=True, positions=np.arange(len(cond_list)), showfliers=True, widths=0.6)

            # apply styles
            for patch, color, hatch in zip(bp['boxes'], colors, hatches):
                patch.set_facecolor(color)
                patch.set_hatch(hatch)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)

            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(self.median_linewidth)
            
            for w in bp['whiskers']:
                w.set_color('black')
                w.set_linewidth(1.5)

            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.5)

            for flier in bp['fliers']:
                flier.set_markerfacecolor('black')
                flier.set_markeredgecolor('none')
                flier.set_alpha(0.5)
                flier.set_markersize(self.markersize)
                flier.set_marker('d')

            clean_bp_data, related_param = self.remove_outliers(bp_data, bp, paired=related_param)

            if related_param:
                N = len(clean_bp_data[0])
                for i in range(len(clean_bp_data)):
                    if len(clean_bp_data[i]) != N:
                        print(f"Warning: Condition {cond_list[i]} has a different number of data points than the others.")
                        break
            else:
                N = [len(data) for data in clean_bp_data]

            # clean look
            replacements_final = {'Atd': 'Att.', 'BL': 'Att. V'}
            ax_name_short = [' '.join(name.split(' ')[0:1]) if i==2 else ' '.join(name.split(' ')[0:2]) for i, name in enumerate(ax_name_sorted)]
            ax_name_plot = [re.sub(r'\b(?:' + '|'.join(replacements_final.keys()) + r')\b', lambda m: replacements_final[m.group()], item) for item in ax_name_short]

            ax.tick_params(axis='both', labelsize=self.fontsize)

            ax.set_xticks(np.arange(len(cond_list)))
            ax.set_xticklabels(ax_name_plot)
            ax.set_xlabel('Condition', labelpad=15, fontsize=self.fontsize)
            
            sig_combs = []
            for c in [(1, 3)]:
                comp = [clean_bp_data[c[0]-1], clean_bp_data[c[1]-1]]
                sig_combs.append(self.determine_significant_combinations(comp, [c[0], c[1]], related=related_param))

            level2 = []
            for i in range(1, len(clean_bp_data)):
                level2 += self.determine_significant_combinations([clean_bp_data[i-1], clean_bp_data[i]], [i, i+1], related=related_param)
            sig_combs.append(level2)

            sig_combs_fdr = self.perform_fdr_correction(sig_combs)
            self.draw_significance(bp, sig_combs_fdr, distance=0.4)

            yborder = [-2.5, 4.5]
            ax.set_ylim(yborder)
            ax.set_yticks(np.arange(-2, yborder[1], 1))
            ax.set_ylabel(f'Delay [ms]', labelpad=15, fontsize=self.fontsize)

            sns.despine()
            plt.tight_layout()
            plt.savefig(save_path / f"Box_Cycles{voice}_{harm}.png", dpi=1200)
            if self.save_svg:
                ax.title.set_visible(False)
                plt.savefig(save_path / f"Box_Cycles{voice}_{harm}.svg", dpi=1200, format='svg', bbox_inches='tight', transparent=True)
            plt.close()

        return
    
    def create_all_Figures(self):
        """Generates all figures for the analysis and saves them as PNG and SVG files."""
        self.percentage_sig_peaks()
        self.single_speaker_magnitudes()
        self.single_speaker_delay()
        self.GA_plots()
        self.competing_speaker_magnitudes()
        self.competing_speaker_delay()
        return

