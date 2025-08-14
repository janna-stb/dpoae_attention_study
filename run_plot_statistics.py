from plot_statistics import StatsPlotter

main_path = 'Path/To/Your/Data/'  # Update this to your actual data path. Needs to lead to the folder containing the statistical data.

# create list of participants to be processed
# also works for subsample of participants 
participants = [f'P{str(p).zfill(2)}' for p in range(1, 41)]
exclude = ['P13', 'P20']
participants = [participant for participant in participants if participant not in exclude]

print('---' * 30)
print('Creating plots for statistical data ...\n')
plotter = StatsPlotter(main_path, participants)
plotter.create_all_Figures()
print('Plots creation completed!')