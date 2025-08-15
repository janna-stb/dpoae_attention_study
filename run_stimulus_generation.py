from stimulus_generation import stimulus_generator, multiband_generator

data_path = 'Path/to/your/data' # update this to the path where your speech signals are
books = ['Polarnacht_female', 'Polarnacht_male', 'Darum_female', 'Darum_male'] # folder names that contain .wav files
harms_male = [[6, 15], [8, 18]] # [n1, n2] for male voice
harms_female = [[7, 15], [9, 18]] # [n1, n2] for female voice
harms = [harms_female, harms_male, harms_female, harms_male]

#%%
for i, book in enumerate(books):
    print('--' * 30 + '\n')
    print(f'Creating stimuli for {book}...\n')
    sg = stimulus_generator(
        n1=harms[i][0],
        n2=harms[i][1],
        voice=book.split('_')[1],
        source_dir=data_path + f'source_audios/{book}',
        target_dir=data_path + f'stimuli/{book}',
        fs=44100
    )
    sg.create_stimuli()
    print('--' * 30 + '\n')
sg.create_pure_tone_stimuli(1000, 1200)  # create pure tone stimuli for all books
print('Stimuli creation completed.\n')
print('--' * 30 + '\n')
print('--' * 30 + '\n')

################################################################
#%%
print('Creating multiband stimuli...\n')

mg = multiband_generator(
    stim_directory=''Path/to/your/data' + '/stimuli', # update this to the path where your speech signals are and add subfolder 'stimuli'
    bookA='Polarnacht_female',
    bookB='Darum_male',
    fs=44100
)

mg.create_multiband_singlespeaker(harms_female[0], harms_female[1], mg.bookA)
mg.create_multiband_singlespeaker(harms_male[0], harms_male[1], mg.bookB)
mg.create_multiband_competingspeaker(mg.bookA, mg.bookB, harms_female[0], harms_female[1], harms_male[0], harms_male[1])
mg.mix_books(data_path + 'source_audios', 'Polarnacht_female', 'Darum_male')

mg_switch = multiband_generator(
    stim_directory=''Path/to/your/data' + '/stimuli',
    bookA='Polarnacht_male',
    bookB='Darum_female',
    fs=44100
)

mg_switch.create_multiband_singlespeaker(harms_male[0], harms_male[1], mg_switch.bookA)
mg_switch.create_multiband_singlespeaker(harms_female[0], harms_female[1], mg_switch.bookB)
mg_switch.create_multiband_competingspeaker(mg_switch.bookA, mg_switch.bookB, harms_male[0], harms_male[1], harms_female[0], harms_female[1])
mg_switch.mix_books(data_path + 'source_audios', 'Polarnacht_male', 'Darum_female')

print('Multiband stimuli creation completed.\n')
