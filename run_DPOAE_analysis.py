from DPOAE_analysis import ExperimentEvaluator

main_path = '/Path/To/Your/Data/'  # Update this to your actual data path

# create list of participants to be processed
# also works for subsample of participants
subjects = [f'P{str(p).zfill(2)}' for p in range(1, 41)]
exclude = ['P13', 'P20']
subjects = [subject for subject in subjects if subject not in exclude]

for sbj in subjects:
    eval = ExperimentEvaluator(sbj, main_path)
    print('--------------------------------------------------\n')
    print(f'Starting evaluation for {sbj} ...\n')
    eval.run_evaluation()
    print('--------------------------------------------------\n\n')
print('Finished!')
