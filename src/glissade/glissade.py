import pandas as pd
import numpy as np
import re 
import pickle
import sys
import argparse
import warnings 
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from glissade.preprocessing import read_data, align_to_reference, seperate_scores
from glissade.pava3 import alpha_minimize, tool_tail_decision

def run_procedure(emp_correct, x_mix, peps, n_boots = 1000):
    alpha_hat, x0_hat, H_hat, G_hat, grid_x, fit_info = alpha_minimize(x_mix, emp_correct, alpha_tol=0.01, B=n_boots, deltas=(0.01, 0.005), cdf_delta=0.01, flat_delta=0.1, min_mix_tail=200, min_alt_tail=100, mix_tail_quantile_cap=0.9, max_checks=50, random_state=1966, x0_tol=0.01, shape_weight_gamma=6.0)
    if not fit_info.get("success", False):
        raise RuntimeError(f"PAVA3 fit failed: {fit_info}")
    alpha_hat, H_hat, _ = tool_tail_decision('casanovo', alpha_hat, x0_hat, H_hat, G_hat, grid_x, x_mix, emp_correct, B=2000, random_state=1966)
    print("Inferred pi0:", 1-alpha_hat)
    
    fdrs = []
    scores = []
    ordered_peps = []
    total = 0
    num_correct = 1
    for score,pep in zip(x_mix, peps):
        total += 1

        while num_correct < len(emp_correct) and score <= emp_correct[num_correct]:
            num_correct += 1
        
        true_count_hat = ((num_correct) / len(emp_correct)) * ((alpha_hat) * len(x_mix))

        scores.append(score)
        ordered_peps.append(pep)
        fdr = (total-true_count_hat) / (total)
        if fdr < 0:
            fdr = np.inf
        fdrs.append(fdr)
        # print(pep, score, fdr)
    
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
  res.to_csv('glissade_discoveries.tsv', sep='\t', index=False)
   
def main():
  parser=argparse.ArgumentParser()
  parser.add_argument("denovo_results")
  parser.add_argument("database_psm_results")
  parser.add_argument("database_peptide_results")
  parser.add_argument("fasta_file")
  parser.add_argument("-n", "--n_bootstraps", type= int, default= 1000, required=False, help="Number of bootstrap samples to perform")
  args = parser.parse_args(args=sys.argv[1:])
  
  denovo_results = args.denovo_results
  database_psm_results = args.database_psm_results
  database_peptide_results = args.database_peptide_results
  fasta_file = args.fasta_file
  n_bootstraps = args.n_bootstraps

  print('Reading search results and aligning to reference...')
  joined_df = read_data(database_psm_results, database_peptide_results, denovo_results)
  labeled_df = align_to_reference(joined_df, fasta_file)
  matched_scores, external_scores, external_peps = seperate_scores(labeled_df)
  print(f"Total matched scores: {len(matched_scores)}")
  
  print(f"Performing FDR control on {len(external_scores)} external peptides from de novo sequencing")
  fdrs, peps, scores = run_procedure(matched_scores, external_scores, external_peps, n_boots = n_bootstraps)
  fdrs = compute_fdr_transform(fdrs)
  write_results(peps, fdrs, scores)

