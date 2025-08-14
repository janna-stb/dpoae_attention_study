import numpy as np
import librosa
from pathlib import Path
import os
import re
import soundfile as sf
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from collections.abc import Iterable
from plot_stimulus_correlations import stimuli_correlation
from butterworth_filter import butter_filter

class stimulus_generator:
    """Generates stimuli for speech-DPOAE measurements. Stimuli follow the specified harmonic numbers n1 and n2.
    Distortion product is calculated as 2*n1 - n2 and can be changed via distortion_product method.

    Parameters
    ----------
    n1 : list or int
        The lower harmonic number(s) for the stimuli.
    n2 : list or int
        The upper harmonic number(s) for the stimuli.
    voice : str
        The voice type of the source audio, setting the fundamental frequency range. 
        Can either be 'male' or 'female'.
    source_dir : str or Path
        The directory containing the source audio files.
    target_dir : str or Path
        The directory where the generated stimuli will be saved.
    fs : int, optional
        The sampling frequency of the audio files, by default 44100.
    """
    def __init__(self, n1, n2, voice, source_dir, target_dir, fs=44100):
        self.fs = fs
        self.n1 = np.array(n1)
        self.n2 = np.array(n2)
        if len(self.n1) != len(self.n2):
            raise ValueError("n1 and n2 must have the same length.")
        
        self.dp = self.distortion_product(self.n1, self.n2)
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        self.voice = voice
         
        if self.voice == 'female':
            self.f0_min, self.f0_max = 80, 300
        elif self.voice == 'male':
            self.f0_min, self.f0_max = 50, 200
        else:
            raise ValueError('Unknown voice.')

    #####################################################################
    def distortion_product(self, n1, n2):
        """Calculate the distortion product based on the provided harmonic numbers."""
        if isinstance(n1, Iterable) and isinstance(n2, Iterable):
            return 2 * np.array(n1) - np.array(n2)
        else:
            return 2 * n1 - n2

    def generate_sine_wave(self, frequency, duration, nrm=0.01):
        """Generate a sine wave of a given frequency and duration."""
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        wave = nrm * np.sin(2 * np.pi * frequency * t)
        return wave

    def save_wave(self, wave, filename):
        """Save the generated waveform to a file."""
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        sf.write(filename, wave, self.fs)

    def zscore_norm(self, data):
        """Normalize the data using z-score normalization."""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    
    ###################################################################
        
    def load_source_audio(self):
        """Load all audio files from the source directory."""
        self.source_files = sorted(list(self.source_dir.glob('*.wav')))
        if not self.source_files:
            raise FileNotFoundError("No .wav files found in the source directory.")
        return
    
    def determine_f0(self, audio_array, harm):
        """Estimate the fundamental frequency (f0) of the audio signal using the pyin algorithm.
        
        Parameters
        ----------
        audio_array : np.ndarray
            The audio signal from which to estimate the f0.
        harm : int
            The harmonic number for which to estimate the f0.
        
        Returns
        -------
        f0_mean : float
            The mean of the estimated f0 values.
        f0_std : float
            The standard deviation of the estimated f0 values.
        """
        # Parameters for pyin algorithm
        fr_length = 4000
        win_length = 3300
        hop_length = 1250

        # Estimate the fundamental frequency (f0)
        f0, voiced_flag, _ = librosa.pyin(audio_array, sr=self.fs, fmin=(harm*self.f0_min), fmax=(harm*self.f0_max), frame_length=fr_length, win_length=win_length, hop_length=hop_length)
        
        # Calculate mean and standard deviation of f0
        f0_mean = np.mean(f0[voiced_flag])
        f0_std = np.std(f0[voiced_flag])
        
        return f0_mean, f0_std
    
    #######################################################################
    
    def gen_fundamental_waveform(self, x, f0_mean, f0_std):
        """Generate the fundamental waveform of the audio signal x by bandpass filtering around the fundamental frequency of the signal.
        Filter bandwidth is defined as f0_mean +/- f0_std/2.

        Parameters
        ----------
        x : np.ndarray
            The audio signal from which to generate the fundamental waveform.
        f0_mean : float
            The mean of the estimated fundamental frequency.
        f0_std : float
            The standard deviation of the estimated fundamental frequency.
        
        Returns
        -------
        fund_wf : np.ndarray
            The generated fundamental waveform.
        """
        lowcut = np.round(f0_mean - f0_std/2, 1)
        highcut = np.round(f0_mean + f0_std/2, 1)

        fund_wf = butter_filter(x, [lowcut, highcut], 'bandpass', fs=self.fs)
        fund_wf = self.zscore_norm(fund_wf)

        return fund_wf

    def gen_stimuli_hilbert(self, n, fund_wf, nrm=0.01):
        """Generate the stimulus waveform for the nth harmonic overtone using the Hilbert transform of the fundamental waveform.
        
        Parameters
        ----------
        n : int
            The harmonic number for which to generate the stimulus waveform.
        fund_wf : np.ndarray
            The fundamental waveform of the audio signal.
        nrm : float, optional
            The normalization factor for the generated waveform, by default 0.01.

        Returns
        -------
        stim : np.ndarray
            The generated stimulus waveform for the nth harmonic overtone.
        """

        fund_wf_orig = fund_wf.copy()
        # remove nans from fund_wf and replace with zeros
        fund_wf = np.nan_to_num(fund_wf_orig)

        # as the hilbert transform expects a periodic signal, concatenate the waveform with itself
        fund_wf = np.concatenate((fund_wf[::-1], fund_wf))

        # apply the hilbert transform to get the amplitude and phase
        wf_hilbert = sig.hilbert(fund_wf)
        amp = np.abs(wf_hilbert[len(fund_wf_orig):])
        phase = np.unwrap(np.angle(wf_hilbert[len(fund_wf_orig):])) # halbe Länge, weil wir ja concatenated hatten

        stim = nrm * np.nan_to_num(amp * np.cos(phase * n))
        stim = stim - np.mean(stim)  # remove DC offset

        return stim
    
    ####################################################################
    def create_pure_tone_stimuli(self, f1, f2, duration=120):
        """Creates pure tone stimuli based on the specified frequencies.
        
        Parameters
        ----------
        f1 : float
            The frequency of the first pure tone in Hz.
        f2 : float
            The frequency of the second pure tone in Hz.
        duration : float, optional
            The duration of the stimuli in seconds, by default 120 seconds.
        """
        pt_dp = self.distortion_product(f1, f2)
        stim1 = self.generate_sine_wave(f1, duration)
        stim2 = self.generate_sine_wave(f2, duration)
        stimdp = self.generate_sine_wave(pt_dp, duration) 
        
        stereo_stim = np.vstack((stim1, stim2)).T

        self.save_wave(stereo_stim, self.target_dir.parent / f'stereo_pt.wav')
        self.save_wave(stimdp, self.target_dir.parent / f'dp_pt.wav')

        return
    
    def create_stimuli(self):
        """Create stimuli based on the source audio files and the specified harmonic numbers."""
        # Load source audio files
        self.load_source_audio()
        
        for audio_file in self.source_files:
            audio_name = audio_file.stem
            audio_array, _ = librosa.load(audio_file, sr=self.fs)
            audio_array = audio_array / np.max(np.abs(audio_array))  # normalize audio array
            #audio_array = audio_array - np.mean(audio_array)  # remove DC offset

            stim_columns = ['filename', 'f0_mean', 'f0_std', 'harm', 'harm_freq [Hz]', 'harm_std [Hz]']

            csv_stimuli = []
            f0_mean, f0_std = self.determine_f0(audio_array, 1)
            
            # Generate the fundamental waveform
            fund_wf = self.gen_fundamental_waveform(audio_array, f0_mean, f0_std)
            
            # Generate and save stimulus waveforms
            for harms in [self.n1, self.n2, self.dp]:
                if isinstance(harms, Iterable):
                    for n in harms:
                        wf = self.gen_stimuli_hilbert(n, fund_wf)
                        freq_stim, std_stim = self.determine_f0(wf, n)
                        filename = f"{self.target_dir}/mono_stimuli/{audio_name}_n{n}.wav" 
                        self.save_wave(wf, filename)
                        csv_stimuli.append([audio_name, f0_mean, f0_std, n, freq_stim, std_stim])

                else:
                    n = harms
                    wf = self.gen_stimuli_hilbert(n, fund_wf)
                    freq_stim, std_stim = self.determine_f0(wf, n)
                    filename = f"{self.target_dir}/mono_stimuli/{audio_name}_n{n}.wav" # das passt noch nicht
                    self.save_wave(wf, filename)
                    csv_stimuli.append([audio_name, f0_mean, f0_std, n, freq_stim, std_stim])

            self.determine_stimulus_correlations(audio_name) 
            csv_stimuli.append(['-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10])  # separator for different audio files   
        
            csv_stimuli_df = pd.DataFrame(csv_stimuli, columns=stim_columns)
            csv_path = self.target_dir / f'stimuli_info.csv'
            write_header = not os.path.exists(csv_path)
            csv_stimuli_df.to_csv(csv_path, mode='a', header=write_header, index=False)


    def determine_stimulus_correlations(self, audio_name):
        """Determine the correlations between the generated stimuli and the distortion product waveform."""
        sig_columns = ['filename', 'n1', 'n2', 'dp', 'sig_peak']
        csv_correlations = []

        stim_path = self.target_dir / 'mono_stimuli/'
        plot_path = self.target_dir / 'mono_stimuli/Plots/'
        
        # determine stimulus correlations
        if isinstance(self.n1, Iterable):
            for n1, n2, dp in zip(self.n1, self.n2, self.dp):
                stim_n1 = librosa.load(stim_path / f'{audio_name}_n{n1}.wav', sr=self.fs)[0]
                stim_n2 = librosa.load(stim_path / f'{audio_name}_n{n2}.wav', sr=self.fs)[0]
                stim_dp = librosa.load(stim_path / f'{audio_name}_n{dp}.wav', sr=self.fs)[0]

                sig_peak = stimuli_correlation(dp, stim_dp, stim_n1, stim_n2, filename=plot_path / f"corr_{audio_name}_n{n1}n{n2}_dp{dp}.png", fs=self.fs)

                csv_correlations.append([audio_name, n1, n2, dp, sig_peak])
        else:
            # if n1, n2, dp are not lists, just use them directly
            stim_n1 = librosa.load(stim_path / f'{audio_name}_n{self.n1}.wav', sr=self.fs)[0]
            stim_n2 = librosa.load(stim_path / f'{audio_name}_n{self.n2}.wav', sr=self.fs)[0]
            stim_dp = librosa.load(stim_path / f'{audio_name}_n{self.dp}.wav', sr=self.fs)[0]

            sig_peak = stimuli_correlation(dp, stim_dp, stim_n1, stim_n2, filename=plot_path / f"corr_{audio_name}_n{self.n1}n{self.n2}_dp{self.dp}.png", fs=self.fs)

            csv_correlations.append([audio_name, self.n1, self.n2, self.dp, sig_peak])

        sig_df = pd.DataFrame(csv_correlations, columns=sig_columns)
        csv_path = self.target_dir / f'stimuli_correlations.csv'
        write_header = not os.path.exists(csv_path)
        sig_df.to_csv(csv_path, mode='a', header=write_header, index=False)

        return sig_peak

    def create_stereo_stimuli(self):
        """Create stereo stimuli by stacking the mono stimuli."""
        mono_stim_path = self.target_dir / 'mono_stimuli/'
        out_path = self.target_dir / 'stereo_stimuli/'
        self.load_source_audio()
        for audio_file in self.source_files:
            audio_name = audio_file.stem
            if isinstance(self.n1, list):
                for n1, n2 in zip(self.n1, self.n2):
                    stim_n1 = librosa.load(mono_stim_path / f'{audio_name}_n{n1}.wav', sr=self.fs)[0]
                    stim_n2 = librosa.load(mono_stim_path / f'{audio_name}_n{n2}.wav', sr=self.fs)[0]
                    
                    # create stereo stimulus
                    stereo_stim = np.vstack((stim_n1, stim_n2)).T
                    filename = out_path / f'{audio_name}_n{n1}n{n2}.wav'
                    self.save_wave(stereo_stim, filename)

            else:
                stim_n1 = librosa.load(mono_stim_path / f'{audio_name}_n{self.n1}.wav', sr=self.fs)[0]
                stim_n2 = librosa.load(mono_stim_path / f'{audio_name}_n{self.n2}.wav', sr=self.fs)[0]
                
                # create stereo stimulus
                stereo_stim = np.vstack((stim_n1, stim_n2)).T
                filename = out_path / f'{audio_name}_n{self.n1}n{self.n2}.wav'
                self.save_wave(stereo_stim, filename)
        return
    

class multiband_generator:
    def __init__(self, stim_directory, bookA, bookB=None, fs=44100):
        """
        Initialize the multiband generator with the directory containing stimuli and the books to be combined.
        Parameters
        ----------
        stim_directory : str or Path
            The directory where the stimulus files are located.
        bookA : str
            The name of the folder containing the mono stimuli for book A.
            Typically, it should follow the format 'Name_Voice'.
        bookB : str
            The name of the folder containing the mono stimuli for book B.
            Typically, it should follow the format 'Name_Voice'.
            If not provided, only book A will be used.
        fs : int, optional
            The sampling frequency of the audio files, by default 44100.
        """

        self.stim_directory = Path(stim_directory)
        self.bookA = bookA
        self.bookB = bookB
        self.fs = fs

    def distorion_product(self, n1, n2):
        """
        Calculate the distortion product based on the provided harmonic numbers.
        Parameters
        ----------
        n1 : int or list of int
            The first harmonic number(s).
        n2 : int or list of int
            The second harmonic number(s).
        Returns
        -------
        np.ndarray
            The distortion product waveform.
        """
        if isinstance(n1, Iterable) and isinstance(n2, Iterable):
            return 2 * np.array(n1) - np.array(n2)
        else:
            return 2 * n1 - n2

    def save_wave(self, wave, filename):
        """Save the generated waveform to a file."""
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        sf.write(filename, wave, self.fs)

    def extract_chapter_number(self, filename):
        """Extract the chapter number from the filename."""
        base = re.sub(r'_n\d+$', '', filename)  # remove the n1 or n2 suffix
        match = re.findall(r'(?<!\d)(\d{2})(?!\d)', base)  # find two-digit numbers
        return match[0] if match else None

    def load_stim_files(self, book):
        """Load all audio files for the respective book from the stimulus directory."""
        mono_stim_path = self.stim_directory / book / 'mono_stimuli' 
        stim_files = sorted(list(mono_stim_path.glob('*.wav')))

        if not stim_files:
            raise FileNotFoundError("No .wav files found in the stimulus directory.")
        return stim_files
        
    def create_multiband_singlespeaker(self, all_n1, all_n2, book, nrm=0.01):
        """Create multiband stimuli by combining the mono stimuli from book A and book B.
        The resulting stimuli will be saved in a new directory named 'multiband_stimuli'.

        Parameters
        ----------
        all_n1 : list of int
            The harmonic numbers for the first band that will be combined.
        all_n2 : list of int
            The harmonic numbers for the second band that will be combined.
        book : str
            The name of the book for which the multiband stimuli will be created.
        nrm : float, optional
            The normalization factor for the generated waveform, by default 0.01.
        """
        # Create target directory for multiband stimuli
        target_dir = self.stim_directory / book / 'multiband_stimuli'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Load stimuli for book A only
        mono_stimuli = self.load_stim_files(book) 
        chapters = sorted(set(self.extract_chapter_number(file.stem) for file in mono_stimuli if 'single' in file.stem))

        for chapter in chapters:
            chapter_files = [file for file in mono_stimuli if self.extract_chapter_number(file.stem) == chapter and 'single' in file.stem]
            base_name = '_'.join(chapter_files[0].stem.split('_')[:3])
            harms_name = ''.join([f'n{n1}' for n1 in all_n1]) + '_' + ''.join([f'n{n2}' for n2 in all_n2])
            multiband_n1 = np.array([])
            multiband_n2 = np.array([])
            
            n1_files = [file for file in chapter_files if any(f'n{n1}' in file.stem for n1 in all_n1)]
            n2_files = [file for file in chapter_files if any(f'n{n2}' in file.stem for n2 in all_n2)]

            if len(n1_files) != len(n2_files):
                raise ValueError(f"Number of n1 and n2 files do not match for chapter {chapter} in book {book}.")
            
            for n1_file, n2_file in zip(n1_files, n2_files):
                # Load the audio files
                stim_n1, _ = librosa.load(n1_file, sr=self.fs)
                stim_n2, _ = librosa.load(n2_file, sr=self.fs)

                if multiband_n1.shape[0] == 0 and multiband_n2.shape[0] == 0:
                    multiband_n1 = stim_n1
                    multiband_n2 = stim_n2
                else:
                    multiband_n1 += stim_n1
                    multiband_n2 += stim_n2

            # Normalize the multiband stimuli
            multiband_n1 = nrm * multiband_n1 / np.max(np.abs(multiband_n1))
            multiband_n2 = nrm * multiband_n2 / np.max(np.abs(multiband_n2))

            # remove DC offset
            multiband_n1 = multiband_n1 - np.mean(multiband_n1)
            multiband_n2 = multiband_n2 - np.mean(multiband_n2)

            # save them as stereo stimuli
            multiband_stim = np.vstack((multiband_n1, multiband_n2)).T
            filename = target_dir / f'{base_name}_{harms_name}.wav'
            self.save_wave(multiband_stim, filename)

            # calculate correlation between dp waveforms and multiband stimuli
            dp = self.distorion_product(all_n1, all_n2)
            dp_files = [file for file in chapter_files if any(f'n{d}' in file.stem for d in dp)]

            sig_peak = []
            for ndp, dp_file in zip(dp, dp_files):
                wf_dp, _ = librosa.load(dp_file, sr=self.fs)
                sig_peak.append(stimuli_correlation(ndp, wf_dp, multiband_n1, multiband_n2, filename=target_dir / f"corr_{base_name}_{harms_name}_dp{ndp}.png", fs=self.fs))

            # Save the correlation results to a CSV file
            multiband_columns = ['filename', 'n1', 'n2', 'dp', 'sig_peak']
            multiband_data = np.array([[base_name for i in range(len(sig_peak))], ['n'.join(map(str, all_n1)) for i in range(len(sig_peak))], ['n'.join(map(str, all_n2)) for i in range(len(sig_peak))], list(dp), sig_peak]).T
            multiband_csv = pd.DataFrame(multiband_data, columns=multiband_columns)

            csv_path = self.stim_directory / book / f'multiband_correlations_{book}_{harms_name}.csv'
            write_header = not os.path.exists(csv_path)
            multiband_csv.to_csv(csv_path, mode='a', header=write_header, index=False)

        return


    def create_multiband_competingspeaker(self, bookA, bookB, A_n1, A_n2, B_n1, B_n2, nrm=0.01):
        """Create multiband stimuli by combining the mono stimuli from book A and book B.
        The resulting stimuli will be saved in a new directory named 'competing_stimuli_{bookA}_{bookB}'.

        Parameters
        ----------
        bookA : str
            The name of the folder containing the mono stimuli for book A.
        bookB : str
            The name of the folder containing the mono stimuli for book B.
        A_n1 : list of int
            The harmonic numbers for the first band in book A that will be combined.
        A_n2 : list of int
            The harmonic numbers for the second band in book A that will be combined.
        B_n1 : list of int
            The harmonic numbers for the first band in book B that will be combined.
        B_n2 : list of int
            The harmonic numbers for the second band in book B that will be combined.
        nrm : float, optional
            The normalization factor for the generated waveform, by default 0.01.
        """
        target_dir = self.stim_directory / f'competing_stimuli_{self.bookA}_{self.bookB}'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        stim_bookA = self.load_stim_files(bookA)
        stim_bookB = self.load_stim_files(bookB)

        # remove potential single speaker files
        stim_bookA = [file for file in stim_bookA if 'single' not in file.stem]
        stim_bookB = [file for file in stim_bookB if 'single' not in file.stem]

        chapters_A = sorted(set(self.extract_chapter_number(file.stem) for file in stim_bookA))
        chapters_B = sorted(set(self.extract_chapter_number(file.stem) for file in stim_bookB))
        
        if chapters_A != chapters_B:
            raise ValueError("Chapters in book A and book B do not match.")
        
        for chapter in chapters_A:
            chapter_files_A = [file for file in stim_bookA if self.extract_chapter_number(file.stem) == chapter]
            chapter_files_B = [file for file in stim_bookB if self.extract_chapter_number(file.stem) == chapter]

            base_name = 'Ch_' + chapter 
            harms_A = ''.join([f'n{n1}' for n1 in A_n1]) + '_' + ''.join([f'n{n2}' for n2 in A_n2])
            harms_B = ''.join([f'n{n1}' for n1 in B_n1]) + '_' + ''.join([f'n{n2}' for n2 in B_n2])
            
            multiband_n1 = np.array([])
            multiband_n2 = np.array([])

            n1_files_A = [file for file in chapter_files_A if any(f'n{n1}' in file.stem for n1 in A_n1)]
            n2_files_A = [file for file in chapter_files_A if any(f'n{n2}' in file.stem for n2 in A_n2)]
            n1_files_B = [file for file in chapter_files_B if any(f'n{n1}' in file.stem for n1 in B_n1)]
            n2_files_B = [file for file in chapter_files_B if any(f'n{n2}' in file.stem for n2 in B_n2)]

            if len(n1_files_A) != len(n2_files_A) or len(n1_files_B) != len(n2_files_B):
                raise ValueError(f"Number of n1 and n2 files do not match for chapter {chapter}.")

            for n1_file_A, n1_file_B, n2_file_A, n2_file_B in zip(n1_files_A, n1_files_B, n2_files_A, n2_files_B):
                # Load the audio files
                stim_n1_A, _ = librosa.load(n1_file_A, sr=self.fs)
                stim_n2_A, _ = librosa.load(n2_file_A, sr=self.fs)
                stim_n1_B, _ = librosa.load(n1_file_B, sr=self.fs)
                stim_n2_B, _ = librosa.load(n2_file_B, sr=self.fs)

                if len(stim_n1_A) > len(stim_n1_B):
                    stim_n1_B = np.pad(stim_n1_B, (0, len(stim_n1_A) - len(stim_n1_B)), 'constant')
                elif len(stim_n1_B) > len(stim_n1_A):
                    stim_n1_A = np.pad(stim_n1_A, (0, len(stim_n1_B) - len(stim_n1_A)), 'constant')

                if len(stim_n2_A) > len(stim_n2_B):
                    stim_n2_B = np.pad(stim_n2_B, (0, len(stim_n2_A) - len(stim_n2_B)), 'constant')
                elif len(stim_n2_B) > len(stim_n2_A):
                    stim_n2_A = np.pad(stim_n2_A, (0, len(stim_n2_B) - len(stim_n2_A)), 'constant')

                if multiband_n1.shape[0] == 0 and multiband_n2.shape[0] == 0:
                    multiband_n1 = stim_n1_A + stim_n1_B
                    multiband_n2 = stim_n2_A + stim_n2_B
                else:
                    multiband_n1 += stim_n1_A + stim_n1_B
                    multiband_n2 += stim_n2_A + stim_n2_B

            # Normalize the multiband stimuli
            multiband_n1 = nrm * multiband_n1 / np.max(np.abs(multiband_n1))
            multiband_n2 = nrm * multiband_n2 / np.max(np.abs(multiband_n2))

            # save them as stereo stimuli
            multiband_stim = np.vstack((multiband_n1, multiband_n2)).T
            filename = target_dir / f'{base_name}_{bookA}_{harms_A}_{bookB}_{harms_B}.wav'
            self.save_wave(multiband_stim, filename)

            # calculate correlation between dp waveforms and multiband stimuli
            dp_A = self.distorion_product(A_n1, A_n2)
            dp_B = self.distorion_product(B_n1, B_n2)
            dp_files_A = [file for file in chapter_files_A if any(f'n{d}' in file.stem for d in dp_A)]
            dp_files_B = [file for file in chapter_files_B if any(f'n{d}' in file.stem for d in dp_B)]    

            sig_peak = []
            for i, (dp_file_A, dp_file_B) in enumerate(zip(dp_files_A, dp_files_B)):
                wf_dp_A, _ = librosa.load(dp_file_A, sr=self.fs)
                wf_dp_B, _ = librosa.load(dp_file_B, sr=self.fs)

                res_A = stimuli_correlation(dp_A[i], wf_dp_A, multiband_n1, multiband_n2, filename=target_dir / f"corr_{base_name}_{bookA}_dp{dp_A[i]}.png", fs=self.fs)
                res_B = stimuli_correlation(dp_B[i], wf_dp_B, multiband_n1, multiband_n2, filename=target_dir / f"corr_{base_name}_{bookB}_dp{dp_B[i]}.png", fs=self.fs)

                sig_peak.append((res_A, res_B))

            # Save the correlation results to a CSV file
            multiband_columns = ['filename', 'book A', 'book B', 'n1 (book A)', 'n2 (book A)', 'dp (book A)', 'n1 (book B)', 'n2 (book B)', 'dp (book B)', 'sig_peak_A', 'sig_peak_B']
            multiband_data = np.array([[base_name for i in range(len(sig_peak))], [bookA for i in range(len(sig_peak))], [bookB for i in range(len(sig_peak))], ['n'.join(map(str, A_n1)) for i in range(len(sig_peak))], ['n'.join(map(str, A_n2)) for i in range(len(sig_peak))], list(dp_A), ['n'.join(map(str, B_n1)) for i in range(len(sig_peak))], ['n'.join(map(str, B_n2)) for i in range(len(sig_peak))], list(dp_B),[res[0] for res in sig_peak], [res[1] for res in sig_peak]]).T
            multiband_csv = pd.DataFrame(multiband_data, columns=multiband_columns)
            csv_path = self.stim_directory / f'multiband_correlations_{bookA}_{harms_A}_{bookB}_{harms_B}.csv'
            write_header = not os.path.exists(csv_path)
            multiband_csv.to_csv(csv_path, mode='a', header=write_header, index=False)

        return

    def mix_books(self, audio_path, bookA, bookB, nrm=0.02):
        """Mix the audio files from book A and book B into stereo stimuli.
        The resulting stimuli will be saved in a new directory named 'competing_audios_{bookA}_{bookB}'.

        Parameters
        ----------
        audio_path : str or Path
            The directory where the source audio files are located.
        bookA : str
            The name of the folder containing the audio files for book A.
        bookB : str
            The name of the folder containing the audio files for book B.
        nrm : float, optional
            The normalization factor for the generated waveform, by default 0.02.
        """
        audio_path = Path(audio_path)
        single_dir = audio_path.parent / 'single_audios/'
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)
        target_dir = audio_path.parent / f'competing_audios_{self.bookA}_{self.bookB}'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        path_bookA = audio_path / bookA 
        path_bookB = audio_path / bookB

        source_files_A = sorted(list(path_bookA.glob('*.wav')))
        source_files_B = sorted(list(path_bookB.glob('*.wav')))

        # extract potential single speaker files
        source_files_single = [file for file in source_files_A + source_files_B if 'single' in file.stem]
        
        # save single speaker files separately
        for file in source_files_single:
            single_audio = librosa.load(file, sr=self.fs)[0]
            single_audio = nrm * single_audio / np.max(np.abs(single_audio))
            self.save_wave(single_audio, single_dir / file.name)

        source_files_A = [file for file in source_files_A if file not in source_files_single]
        source_files_B = [file for file in source_files_B if file not in source_files_single]

        if len(source_files_A) != len(source_files_B):
            raise ValueError(f"Number of audio files in {bookA} and {bookB} do not match.")
        
        # mix remaining chapters together
        chapters_A = sorted(set(self.extract_chapter_number(file.stem) for file in source_files_A))
        chapters_B = sorted(set(self.extract_chapter_number(file.stem) for file in source_files_B))

        if chapters_A != chapters_B:
            raise ValueError("Chapters in book A and book B do not match.")
        
        for chapter in chapters_A:
            chapter_files_A = [file for file in source_files_A if self.extract_chapter_number(file.stem) == chapter]
            chapter_files_B = [file for file in source_files_B if self.extract_chapter_number(file.stem) == chapter]

            base_name = 'Ch_' + chapter 

            audio_A = librosa.load(chapter_files_A[0], sr=self.fs)[0]
            audio_B = librosa.load(chapter_files_B[0], sr=self.fs)[0]

            audio_A = nrm * audio_A / np.max(np.abs(audio_A))
            audio_B = nrm * audio_B / np.max(np.abs(audio_B))

            if len(audio_A) > len(audio_B):
                audio_B = np.pad(audio_B, (0, len(audio_A) - len(audio_B)), 'constant')
            elif len(audio_B) > len(audio_A):
                audio_A = np.pad(audio_A, (0, len(audio_B) - len(audio_A)), 'constant')

            mixed_audio = np.vstack((audio_A, audio_B)).T
            filename = target_dir / f'{base_name}_{bookA}_{bookB}.wav'
            self.save_wave(mixed_audio, filename)

        return
                                                                   
                                                       





