import os
import pandas as pd
import numpy as np
import re
import pickle
import sys
import argparse
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def read_data(db_file : str, denovo_file : str):
  """
  Read in database search results and de novo results and join
  results on (file_stem, scan) to support multi-file analyses.

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
    # Use the filename column for file identity (supports multi-file analyses)
    db_df['file_stem'] = db_df['filename'].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
  else:
    #FIXME handle other search results
    pass

  if '.mztab' in denovo_file:
    # Build ms_run -> file stem mapping from MTD header
    run_map = {}
    with open(denovo_file) as f_in:
      for skiprows, line in enumerate(f_in):
          if line.startswith("PSH"):
              break
          m = re.match(r'MTD\s+(ms_run\[\d+\])-location\s+(.*)', line.strip())
          if m:
              run_map[m.group(1)] = os.path.splitext(os.path.basename(m.group(2).replace('file://', '').strip()))[0]
    denovo_df = pd.read_csv(denovo_file, sep='\t', skiprows=skiprows)
    def _parse_scan(ref):
      ref = str(ref) if ref is not None else ''
      m = re.search(r'scan=(\d+)', ref)
      return int(m.group(1)) if m else None
    def _parse_stem(ref):
      ref = str(ref) if ref is not None else ''
      m = re.match(r'(ms_run\[\d+\])', ref)
      return run_map.get(m.group(1), '') if m else ''
    dn_scans = [_parse_scan(x) for x in denovo_df['spectra_ref']]
    dn_stems = [_parse_stem(x) for x in denovo_df['spectra_ref']]
    denovo_df['scan'] = dn_scans
    denovo_df['file_stem'] = dn_stems
    n_before = len(denovo_df)
    denovo_df = denovo_df.dropna(subset=['scan'])
    denovo_df['scan'] = denovo_df['scan'].astype(int)
    if len(denovo_df) < n_before:
      print(f"  Dropped {n_before - len(denovo_df)} Casanovo rows with unparseable spectra_ref")
    denovo_df = denovo_df.rename(columns={'search_engine_score[1]': 'denovo_score', 'sequence': 'denovo_peptide'})

  else:
    #FIXME handle other denovo result formats
    pass

  print(f"  Percolator PSMs: {len(db_df)}")
  print(f"  Casanovo PSMs:   {len(denovo_df)}")

  joined_df = pd.merge(db_df, denovo_df, on=['file_stem', 'scan'], how='inner')
  joined_df.sort_values(by="denovo_score", ascending=False, inplace=True)

  print(f"  PSMs after inner join on (file, scan): {len(joined_df)}")
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
  n_in_tide = sum(in_tide)
  print(f"  FDR threshold: {database_fdr_threshold}")
  n_total = len(results_df)
  pct = f"{100*n_in_tide/n_total:.1f}%" if n_total > 0 else "N/A"
  print(f"  Percolator PSMs passing FDR: {n_in_tide} / {n_total} ({pct})")

  db_peps = [''.join([i for i in re.sub(r'\[.*?\]', '', x[2:-2]) if i.isalpha()]).replace('I','L') for x in results_df['peptide']]
  denovo_peps = [''.join([i for i in re.sub(r'\[.*?\]', '', x) if i.isalpha()]).replace('I','L') for x in results_df['denovo_peptide']]
  agrees = [x == y for x,y in zip(db_peps, denovo_peps)]
  n_agrees = sum(agrees)
  n_both = sum(a and b for a, b in zip(in_tide, agrees))
  pct_agrees = f"{100*n_agrees/n_total:.1f}%" if n_total > 0 else "N/A"
  pct_both = f"{100*n_both/n_in_tide:.1f}%" if n_in_tide > 0 else "N/A"
  print(f"  PSMs where Casanovo and Percolator agree: {n_agrees} / {n_total} ({pct_agrees})")
  print(f"  PSMs passing FDR and agreeing: {n_both} / {n_in_tide} ({pct_both} of FDR-passing)")

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

  n_matched_raw = len(matched_df)
  n_external_raw = len(external_df)
  print(f"  Matched PSMs (in_tide & agrees & score>0): {n_matched_raw}")
  print(f"  External PSMs (not in_tide, not in_reference, score>0): {n_external_raw}")

  matched_df = matched_df[matched_df['denovo_peptide'].apply(lambda x: len(x) >= min_length)]
  external_df = external_df[external_df['denovo_peptide'].apply(lambda x: len(x) >= min_length)]

  n_matched_filt = len(matched_df)
  n_external_filt = len(external_df)
  pct_m = f"{100*n_matched_filt/n_matched_raw:.1f}%" if n_matched_raw > 0 else "N/A"
  pct_e = f"{100*n_external_filt/n_external_raw:.1f}%" if n_external_raw > 0 else "N/A"
  print(f"  Matched PSMs after min_length={min_length} filter: {n_matched_filt} / {n_matched_raw} ({pct_m})")
  print(f"  External PSMs after min_length={min_length} filter: {n_external_filt} / {n_external_raw} ({pct_e})")

  matched_peps = matched_df.groupby('denovo_peptide')['denovo_score'].max()
  external_peps = external_df.groupby('denovo_peptide')['denovo_score'].max()

  pct_um = f"{100*len(matched_peps)/n_matched_filt:.1f}%" if n_matched_filt > 0 else "N/A"
  pct_ue = f"{100*len(external_peps)/n_external_filt:.1f}%" if n_external_filt > 0 else "N/A"
  print(f"  Unique matched peptide sequences: {len(matched_peps)} / {n_matched_filt} ({pct_um})")
  print(f"  Unique external peptide sequences: {len(external_peps)} / {n_external_filt} ({pct_ue})")

  matched_scores =  np.log(matched_peps.values)
  external_scores =  np.log(external_peps.values)
  return matched_scores, external_scores, list(external_peps.index)