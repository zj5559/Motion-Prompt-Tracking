import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot_extension_subset'
# dataset_name = 'lasot'
# dataset_name = 'tnl2k'


"""ostrack"""


trackers.extend(trackerlist(name='ostrack', parameter_name='MPT_MAE256_ep60', dataset_name=dataset_name,
                            run_ids=-1, display_name='OSTrack256_MPT'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='MPT_MAE384_ep60', dataset_name=dataset_name,
                            # run_ids=-1, display_name='OSTrack384_MPT'))


dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True,seq_eval=False, plot_types=('success', 'norm_prec', 'prec'))
