import pandas as pd
import numpy as np
import re 
import pickle
import sys
import argparse
import warnings 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def read_data(psm_file : str, peptide_file, denovo_file : str, database_fdr_threshold = 0.01):
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
  if 'percolator' in psm_file:
    db_df = pd.read_csv(psm_file, sep='\t')
    db_scans = [int(x.split('_')[2]) for x in db_df['PSMId']]
    db_df['scan'] = db_scans
  elif False:
      #FIXME handle other search results
      pass
  else:
      print('Unsupported database search file format')
      exit()
  
  if 'percolator' in peptide_file:
      pep_df = pd.read_csv(peptide_file, sep='\t')
      peps = [x for x in pep_df['peptide']]
      qs = [float(x) for x in pep_df['q-value']]
  elif False:
      #FIXME handle other search results
      pass
  else:
      print('Unsupported database search file format')
      exit()

  pep2q = {}
  for pep,q in zip(peps,qs):
     pep2q[pep] = q

  if '.mztab' in denovo_file:
    with open(denovo_file) as f_in:
      for skiprows, line in enumerate(f_in):
          if line.startswith("PSH"):
              break
    denovo_df = pd.read_csv(denovo_file, sep='\t', skiprows=skiprows)
    dn_scans = [int(x.split('scan=')[1].split('\t')[0]) for x in denovo_df['spectra_ref']]
    denovo_df['scan'] = dn_scans
    denovo_df = denovo_df.rename(columns={'search_engine_score[1]': 'denovo_score', 'sequence': 'denovo_peptide'})
  elif '.csv' in denovo_file:
    denovo_df = pd.read_csv(denovo_file)
    denovo_df = denovo_df.rename(columns={'scan_number': 'scan', 'predictions': 'denovo_peptide'})
    denovo_df['denovo_score'] = [np.exp(x) for x in denovo_df['log_probs']]
  elif '.tab' in denovo_file:
    denovo_df = pd.read_csv(denovo_file, sep='\t')
    dn_scans = [int(x.split(':')[1]) for x in denovo_df['scan']]
    denovo_df['scan'] = dn_scans
    denovo_df['denovo_peptide'] = [str(x).replace(',','').replace('mod','') for x in denovo_df['output_seq']]
    denovo_df['denovo_score'] = [np.exp(x) for x in denovo_df['output_score']]
  else:
     print('Unsupported de novo file format')
     exit()

  joined_df = pd.merge(db_df, denovo_df, on='scan', how='inner')
  joined_df.sort_values(by="denovo_score", ascending=False, inplace=True)

  in_tide = [pep2q[pep] < database_fdr_threshold for pep in joined_df['peptide']]
  joined_df['in_tide'] = in_tide

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

  db_peps = [''.join([i for i in re.sub(r'\[.*?\]', '', x[2:-2]) if i.isalpha()]).replace('I','L') for x in results_df['peptide']]
  denovo_peps = [''.join([i for i in re.sub(r'\[.*?\]', '', x) if i.isalpha()]).replace('I','L') for x in results_df['denovo_peptide']]
  agrees = [x == y for x,y in zip(db_peps, denovo_peps)]

  in_reference = [x in all_prots_string for x in denovo_peps]

  results_df['in_reference'] = in_reference
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
  matched_scores.sort()
  external_scores =  np.log(external_peps.values)
  sidxs = np.argsort(external_scores)[::-1]
  return matched_scores[::-1], np.array(external_scores)[sidxs], np.array(external_peps.index)[sidxs]