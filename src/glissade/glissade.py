import pandas as pd
import numpy as np
import re 
import pickle
import sys
import argparse
import warnings 
import matplotlib.pyplot as plt
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

def read_data(db_file : str, denovo_file : str):
  """
  Read in database search results and de novo results and join 
  results on scan ID. 

  Parameters
  ----------
  db_file: string containing the path to the database search results file. 
  denovo_file: string containing the path to the denovo results file. 

  Returns
  -------
  joined_df: a combined dataframe containing the de novo and database 
              search result for each scan
  """
  if 'psms.txt' in db_file:
    db_df = pd.read_csv(db_file, sep='\t')
    db_scans = [int(x.split('_')[2]) for x in db_df['PSMId']]
    db_df['scan'] = db_scans
  else:
    #FIXME handle other search results
    pass

  if '.mztab' in denovo_file:
    with open(denovo_file) as f_in:
      for skiprows, line in enumerate(f_in):
          if line.startswith("PSH"):
              break
    denovo_df = pd.read_csv(denovo_file, sep='\t', skiprows=skiprows)
    dn_scans = [int(x.split('scan=')[1].split('\t')[0]) for x in denovo_df['spectra_ref']]
    denovo_df['scan'] = dn_scans
    denovo_df = denovo_df.rename(columns={'search_engine_score[1]': 'denovo_score', 'sequence': 'denovo_peptide'})

  else:
    #FIXME handle other denovo result formats
    pass

  joined_df = pd.merge(db_df, denovo_df, on='scan', how='inner')
  joined_df.sort_values(by="denovo_score", ascending=False, inplace=True)

  return joined_df

def align_to_reference(results_df : pd.DataFrame, reference_file : str, database_fdr_threshold : float = 0.01):
  """
  Annotate the database search and denovo results based on whether they agree and whether the 
  de novo peptide is in the reference

  Parameters
  ----------
  results_df: a pandas dataframe containing the de novo and database 
              search result for each scan
  reference_file: a string containing the path to a reference FASTA
  database_fdr_threshold: optional float describing what FDR threshold to apply to database 
              search results (default 0.01)

  Returns
  -------
  results_df: a dataframe containing the de novo and database search result for each scan
              annotated based on agreement and whether each prediction is in the reference 
  """
  all_prots_string = ''
  with open(reference_file) as f_in:
      for line in f_in:
          if not line[0] == '>':
              all_prots_string += line[:-1].replace('I','L')
          else:
              all_prots_string += '$'
  in_tide = [x < database_fdr_threshold for x in results_df['q-value']]

  db_peps = [''.join([i for i in re.sub(r'\[.*?\]', '', x[2:-2]) if i.isalpha()]).replace('I','L') for x in results_df['peptide']]
  denovo_peps = [''.join([i for i in re.sub(r'\[.*?\]', '', x) if i.isalpha()]).replace('I','L') for x in results_df['denovo_peptide']]
  agrees = [x == y for x,y in zip(db_peps, denovo_peps)]

  in_reference = [x in all_prots_string for x in denovo_peps]

  results_df['in_reference'] = in_reference
  results_df['in_tide'] = in_tide
  results_df['agrees'] = agrees
  return results_df

def seperate_scores(labeled_df : pd.DataFrame, min_length : int = 8):
  """
  Extract lists of matched scores and external scores to run the FDR control procedure on.

  Parameters
  ----------
  results_df: A pandas dataframe with columns labeling whether each de novo prediction is for a 
              scan identified by database search, agrees with database search, and is in the reference.  
  min_length: Optional int specifying the minimum length of peptides to consider this should be large 
              enough to make random matches to the database unlikely (default 8). 

  Returns
  -------
  matched_scores: A list of de novo scores corresponding to matched peptides
  external_scores: A list of de novo scores corresponding to external peptides 
  external_peps: A list containing the peptides corresponding to external scores for reporting 
                 the final list of discoveries
  """
  matched_df = labeled_df[labeled_df['in_tide'] & labeled_df['agrees'] & (labeled_df['denovo_score'] > 0)]
  external_df = labeled_df[~labeled_df['in_tide'] & ~labeled_df['in_reference'] & (labeled_df['denovo_score'] > 0)]

  matched_df = matched_df[matched_df['denovo_peptide'].apply(lambda x: len(x) >= min_length)]
  external_df = external_df[external_df['denovo_peptide'].apply(lambda x: len(x) >= min_length)]

  matched_peps = matched_df.groupby('denovo_peptide')['denovo_score'].max()
  external_peps = external_df.groupby('denovo_peptide')['denovo_score'].max()

  matched_scores =  np.log(matched_peps.values)
  external_scores =  np.log(external_peps.values)
  return matched_scores, external_scores, list(external_peps.index)

def find_minima_sig(external_score_sample : np.array, matched_scores : np.array):
  """
  Find the crossover point x_0 in the distribution of external scores

  Parameters
  ----------
  external_score_sample: A list of external scores  
  matched_scores: A list of matched scores 

  Returns
  -------
  x_0: The score corresponding to the inferred crossover point in the external score distribution 
  """
  n_s = 50
  right_bin_edge = max(external_score_sample)
  left_bin_edge = list(sorted(matched_scores))[int(len(matched_scores)/10)]
  current_interval = list(sorted([i for i in external_score_sample if i >= left_bin_edge]))
  
  while True: 
    left_third = left_bin_edge + (right_bin_edge - left_bin_edge)/3
    right_third = left_bin_edge + 2*(right_bin_edge - left_bin_edge)/3

    num_left = 0
    num_middle = 0
    num_right = 0 

    for i in current_interval:
        if i <= left_third:
          num_left += 1
        elif i <= right_third:
          num_middle += 1
        else:
          num_right += 1
    
    left_sig = binomtest(num_left, len(current_interval), p=1/3, alternative='greater').pvalue
    mid_sig = binomtest(num_middle, len(current_interval), p=1/3, alternative='greater').pvalue
    right_sig = binomtest(num_right, len(current_interval), p=1/3, alternative='greater').pvalue

    if left_sig < 0.05 or mid_sig < 0.05: 
       index = int(min(n_s, len(current_interval)/3))
       left_bin_edge = current_interval[index]
       current_interval = current_interval[index:]
    elif right_sig < 0.05: 
       index = int(min(n_s, len(current_interval)/3))
       right_bin_edge = current_interval[-index]
       current_interval = current_interval[:-index]
    elif len(current_interval) < n_s:
       return current_interval[int(len(current_interval)/2)]
    else: 
       index = int(min(n_s, len(current_interval)/3))
       left_bin_edge = current_interval[index]
       right_bin_edge = current_interval[-index]
       current_interval = current_interval[index:-index]

def compute_windows(external_score_sample : np.array, x_0 : float, n_w : int, step_size : int, plot_figs = False):
  """
  Compute windows of equal probability mass between the crossover point x_0 and the max score  

  Parameters
  ----------
  external_score_sample: A sorted array of de novo scores for external peptides
  x_0: The crossover point 
  n_w: The number of samples each window will contain 
  step_size: The step size in number of samples between adjacent windows
  plot_figs: Optional flag for whether to plot the resulting windows (default False)

  Returns
  -------
  windows: A list of tuples containing the start and end score for each window
  """
  external_score_sample = [x for x in external_score_sample if x > 2*x_0]
  N_e = len(external_score_sample)
  
  windows = []
  for i in range(N_e - 1, n_w, -step_size):
    s_i = external_score_sample[i]

    window_start = None 
    window_end = None 
    best_width = np.inf

    for start in range(i-n_w, min(N_e - n_w,i)):
      width = external_score_sample[start + n_w] - external_score_sample[start]
      if width < best_width:
        best_width = width 
        window_start = external_score_sample[start]
        window_end = external_score_sample[start + n_w] 
    
    if (window_start,window_end) not in windows:
      windows.append((window_start,window_end))

    if s_i <= x_0:
      break
  
  if plot_figs:
    for i,(start,stop) in enumerate(windows):
      plt.plot([start,stop],[i,i])
    plt.show()

  return windows

   
def estimate_pi0(matched_scores : np.array, windows : list[tuple[float,float]], n_w : int, N_e : int, plot_figs = False):
  """
  Estimate the mixing parameter pi_0 using the empirical matched score distribution. 
  Samples from the matched_scores are subtracted from the external score distribution
  until all of the windows satisfy the condition that the density at x_0 is higher 
  than the density for all scores x > x_0. 

  Parameters
  ----------
  matched_scores: A sorted array of de novo scores for matched peptides
  windows: A list of tuples containing the start and end score for each equal density window
           in the external scores
  n_w: The number of external samples in each window 
  N_e: The total number of external scores
  plot_figs: Optional flag for whether to plot the resulting windows (default False)

  Returns
  -------
  est_pi_0: The inferred mixing paramater pi_0
  matched_sample: The corresponding sample from the matched score distribution
  """
  counts = np.array([n_w] * len(windows))
  widths = np.array([end - start for start,end in windows])
  matched_sample = []
  while min(counts) >= 1:
    if max([counts[i]/widths[i] - counts[-1]/widths[-1] for i in range(len(counts))]) <= 0:
      break
    sample = np.random.choice(matched_scores, 1)[0]
    matched_sample.append(sample)
    for i, (window_start, window_end) in enumerate(windows):
      if sample >= window_start and sample <= window_end:
        counts[i] -= 1
      elif sample > window_end:
        break
  matched_sample = sorted(matched_sample)

  if plot_figs:
    for i in range(len(counts)):
        plt.plot([windows[i][0], windows[i][1]], [counts[i]/widths[i], counts[i]/widths[i]])
    plt.show()

  est_pi_0 = 1 - len(matched_sample)/N_e
  return est_pi_0, matched_sample

def estimate_fdr(matched_sample, external_score_sample, grid):
  """
  Infer the FDR at each score threshold based on the list of external scores and a corresponding 
  matched sample obtained from estimating pi_0

  Parameters
  ----------
  matched_sample: A list of matched scores with size (1 - pi_0) * len(external_score_sample)  
  external_score_sample: The list of external scores
  grid: An array of score thresholds to calculate the corresponding FDR for 

  Returns
  -------
  qs: A list of q-values corresponding to each score threshold in the grid
  """
  qs = []
  denom = 1
  numer = 1
  for i in range(len(grid)-1, -1, -1):
    s_i = grid[i]
    while denom < len(external_score_sample) and external_score_sample[-denom] >= s_i:
      denom += 1
    while numer < len(matched_sample) and matched_sample[-numer] >= s_i:
      numer += 1
    if denom > 1:
      q = max((denom - numer) / (denom), 0)
    else:
      q = np.nan
    qs.append(q)

  return qs

def run_procedure(matched_scores, external_score_sample, grid, plot_figs = False):
  """
  Runs one iteration of the FDR control procedure for a given bootstrap sample of the external scores. 

  Parameters
  ----------
  matched_scores: A list of matched scores
  external_score_sample: A list containing a bootstrapped sample of the external scores
  grid: An array of score thresholds to calculate the corresponding FDR for 
  plot_figs: Optional flag for whether to plot the resulting windows (default False)

  Returns
  -------
  fdrs: A list of q-values corresponding to each score threshold in the grid
  est_pi_0: The estimated mixing parameter pi_0
  """
  external_score_sample = sorted(external_score_sample)
  
  x_0 = find_minima_sig(external_score_sample, matched_scores)
  #print(f"Crossover point identified at: {x_0}")

  if(plot_figs):
    plt.hist([x for x in external_score_sample if x > -5], bins=100)
    plt.xlabel('Log de novo score')
    plt.xlim([-.5,0])
    plt.ylabel('Density')
    plt.title('External scores')
    plt.show()

    plt.hist([x for x in matched_scores if x > -5], bins=100)
    plt.xlabel('Log de novo score')
    plt.xlim([-.5,0])
    plt.ylabel('Density')
    plt.title('Matched scores')
    plt.show()

  N_e = len(external_score_sample)
  n_w = int(len([x for x in external_score_sample if x >= x_0])/4)
  step_size = 2 #int(n_w/2)
  windows = compute_windows(external_score_sample, x_0, n_w, step_size, plot_figs)

  est_pi_0, matched_sample = estimate_pi0(matched_scores, windows, n_w, N_e, plot_figs)
  fdrs = estimate_fdr(matched_sample, external_score_sample, grid)

  return list(reversed(fdrs)), est_pi_0

def run_bootstraps(matched_scores, external_scores, n_bootstraps = 10, plot_figs = False):
  """
  Runs many iterations of the FDR control procedure for bootstrapped samples of the external scores

  Parameters
  ----------
  matched_scores: A list of matched scores
  external_scores: A list of external scores
  n_bootstraps: The number of bootstraps to perform 
  plot_figs: Optional flag for whether to plot the resulting windows (default False)

  Returns
  -------
  all_fdrs: A list of q-values corresponding to each score threshold in the grid for each of the bootstrap iterations
  grid: The array of score thresholds which the FDR was calculated on 
  all_pi0s: A list of the inferred pi_0 for each bootstrap
  """
  all_fdrs = []
  all_pi0s = []
  grid = np.linspace(min(external_scores), 0, 10000)
  for _ in range(n_bootstraps):
    external_score_sample = np.random.choice(external_scores, size=len(external_scores), replace=True)
    fdrs, est_pi_0 = run_procedure(matched_scores, external_score_sample, grid, plot_figs)
    all_fdrs.append(fdrs)
    all_pi0s.append(est_pi_0)
  print("Average estimated pi_0:", np.mean(all_pi0s))

  return all_fdrs, grid, all_pi0s

def annotate_results(external_peps, external_scores, fdrs, grid):
  """
  Assign the corresponding q-value to the score threshold at which each 
  external peptide would be accepted

  Parameters
  ----------
  external_peps: A list of external peptides sequences
  external_scores: A list of scores assigned to each external peptide
  fdrs: A list of q-values corresponding to each score threshold in the grid
  grid: The array of score thresholds which the FDR was calculated on 

  Returns
  -------
  peptides: A list of external peptides sequences sorted by score
  peptide_fdrs: The corresponding FDR for the score threshold at which each peptide is accepted
  scores: A sorted list of scores for the external peptides
  """
  fdrs = np.maximum(np.nanmean(fdrs, axis=0), 0)
  sidxs = np.argsort(external_scores)
  sorted_external_scores = external_scores[sidxs]
  sorted_external_peps = np.array(external_peps)[sidxs]
  cur_grid_idx = 0
  peptide_fdrs = []
  peptides = []
  scores = []
  for score,pep in zip(sorted_external_scores, sorted_external_peps):
      while score > grid[cur_grid_idx]:
        cur_grid_idx += 1
      peptide_fdrs.append(fdrs[cur_grid_idx])
      peptides.append(pep)
      scores.append(score)
  
  return peptides, peptide_fdrs, scores

def compute_fdr_transform(fdrs):
  """
  Assign to each peptide the lowest q-value corresponding to a score threshold at which that peptide 
  would be accepted ie. the min FDR for all scores greater than or equal to it. 

  Parameters
  ----------
  fdrs: A list of FDRs sorted by their corresponding score 

  Returns
  -------
  transformed_fdrs: The list of FDRs after applying the transformation
  """
  transformed_fdrs = []
  cur_min = 1
  for i in range(len(fdrs)):
    cur_min = max(0, min(cur_min, fdrs[i]))
    transformed_fdrs.append(cur_min)
  return transformed_fdrs

def write_results(peptides, peptide_fdrs, scores):
  """
  Write results to a file

  Parameters
  ----------
  peptides: A list of external peptides sequences sorted by score
  peptide_fdrs: The corresponding FDR for the score threshold at which each peptide is accepted
  scores: A sorted list of scores for the external peptides
  """
  res = pd.DataFrame({"Peptide":peptides, "Score":scores, "q-value":peptide_fdrs})
  res.sort_values(by='Score', ascending=False, inplace=True)
  res.to_csv('peptide.tsv', sep='\t', index=False)

def main():
  parser=argparse.ArgumentParser()
  parser.add_argument("denovo_results")
  parser.add_argument("database_results")
  parser.add_argument("fasta_file")
  parser.add_argument("-n", "--n_bootstraps", type= int, default= 100, required=False, help="Number of bootstrap samples to perform")
  args = parser.parse_args(args=sys.argv[1:])
  
  denovo_results = args.denovo_results
  database_results = args.database_results
  fasta_file = args.fasta_file
  n_bootstraps = args.n_bootstraps

  print('Reading search results and aligning to reference...')
  joined_df = read_data(database_results, denovo_results)
  labeled_df = align_to_reference(joined_df, fasta_file)
  matched_scores, external_scores, external_peps = seperate_scores(labeled_df)
  print(f"Total matched scores: {len(matched_scores)}")
  
  print(f"Performing FDR control on {len(external_scores)} external peptides from de novo sequencing")
  fdrs, grid, _ = run_bootstraps(matched_scores, external_scores, n_bootstraps = n_bootstraps, plot_figs=False)
  peptides, peptide_fdrs, scores = annotate_results(external_peps, external_scores, fdrs, grid)
  peptide_fdrs = compute_fdr_transform(peptide_fdrs)
  write_results(peptides, peptide_fdrs, scores)
