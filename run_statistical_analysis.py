from statistical_analysis import CrossCorrelationProcessor

main_path = 'Path/To/Your/Data/'  # Update this to your actual data path

# create list of participants to be processed
# also works for subsample of participants
participants = [f'P{str(p).zfill(2)}' for p in range(1, 41)]
exclude = ['P13', 'P20'] # exclude participants that did not match inclusion criteria
participants = [participant for participant in participants if participant not in exclude]

print('---'*30)
print(f'Starting statistical analysis for participants {participants[0]} to {participants[-1]}...')
processor = CrossCorrelationProcessor(main_path, participants)

stats_data_path = processor.res_path + 'Data_Plots/'
processor.run_all_statistics(stats_data_path)

processor.evaluate_listening_effort(stats_data_path)
processor.evaluate_comprehension_questions(stats_data_path)
print('Finished statistical analysis!')
