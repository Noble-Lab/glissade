import pandas as pd
import numpy as np
import re 
import pickle
import sys
import argparse
import warnings 
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from preprocessing import read_data, align_to_reference, seperate_scores
from pava import alpha_minimize, mixture_sanity_check, _ecdf_on_grid

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

def run_procedure(alt, x_mix, peps, n_boots = 250):
    x0 = -0.04
    x0 = find_minima_sig(x_mix, alt)
    print("Inferred x0:", x0)

    alpha_lo0, alpha_hi0 = 0.01, 0.99 
    alpha_hat, H_hat, grid_x, info = alpha_minimize(x_mix, alt, x0, alpha_tol=2e-3, tol_mode='bootstrap',  
                                                    B=n_boots, per_check_delta=0.05,  max_checks=30, random_state=1966, 
                                                    initial_bracket=(alpha_lo0, alpha_hi0), verbose=False)
    
    print("Inferred pi_0:", 1-alpha_hat)
    
    Gm = _ecdf_on_grid(np.sort(alt), grid_x)
    Fn = _ecdf_on_grid(np.sort(x_mix), grid_x)
    res = mixture_sanity_check(Fn, Gm, H_hat, alpha_hat, grid_x, mode='bootstrap', delta=0.05, B=400, rng=np.random.default_rng(7))
    print(f"\n[mixture check] D_ks={res['D_ks']:.4g}  crit={res['crit']:.4g}  "
        f"{'(PASS)' if res['pass_test'] else '(FAIL)'}"
        f"{'' if res.get('p_value') is None else f'  p≈{res['p_value']:.3f}'}\n")
    

    emp_correct = alt

    fdrs = []
    scores = []
    ordered_peps = []
    total = 0
    num_correct = 1
    for score,pep in zip(x_mix[::-1], peps[::-1]):
        total += 1

        while num_correct < len(emp_correct) and score <= emp_correct[-num_correct]:
            num_correct += 1
        
        true_count_hat = ((num_correct-1) / len(emp_correct)) * ((alpha_hat) * len(x_mix))

        scores.append(score)
        ordered_peps.append(pep)
        fdr = (total-true_count_hat) / (total)
        if fdr < 0:
            fdr += 10
        fdrs.append(fdr)
    
    return fdrs[::-1], ordered_peps[::-1], scores[::-1]

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
  fdrs, peps, scores = run_procedure(matched_scores, external_scores, external_peps, n_boots = 250)
  fdrs = compute_fdr_transform(fdrs)
  write_results(peps, fdrs, scores)
  
  # fdrs, grid, _ = run_bootstraps(matched_scores, external_scores, n_bootstraps = n_bootstraps, plot_figs=False)
  # peptides, peptide_fdrs, scores = annotate_results(external_peps, external_scores, fdrs, grid)
  # peptide_fdrs = compute_fdr_transform(peptide_fdrs)
  # write_results(peptides, peptide_fdrs, scores)
