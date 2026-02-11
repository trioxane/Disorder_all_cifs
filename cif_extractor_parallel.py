#!/usr/bin/env python3
"""
Script to extract key information from COD database CIF files using pymatgen.
Processes all COD database CIF files in a specified folder in parallel and outputs results to CSV.
"""

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import sys
import csv
from pathlib import Path
import argparse
import warnings
import time
from multiprocessing import Manager, cpu_count
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from functools import partial

from disorder.disorder import Disorder

warnings.filterwarnings('ignore')


def extract_cif_info(cif_file):
    """
    Extract comprehensive information from a CIF file.
    
    Args:
        cif_file: Path to the CIF file
        
    Returns:
        dict: Dictionary containing extracted information
    """
    
    try:
        parser = CifParser(cif_file, occupancy_tolerance=1.0)
        structure = parser.parse_structures()[0]
        occupancies_rescaling = 0
    except Exception as e:
        occupancies_rescaling = 1
        parser = CifParser(cif_file, occupancy_tolerance=1.1)
        structure = parser.parse_structures()[0]
    
    # Get raw CIF data
    cif_data = parser.as_dict()
    
    # Get the first (and typically only) entry key
    entry_key = list(cif_data.keys())[0]
    cif_dict = cif_data[entry_key]
    
    info = {}
    
    # 1. Composition
    info['composition'] = structure.composition.formula
    
    # 2. Element set (chemical system)
    info['chemical_system'] = '-'.join(sorted(set(structure.symbol_set)))
    
    # 3. Number of sites
    info['num_sites'] = len(structure.sites)
    
    # 4. Space group
    info['space_group_number'] = structure.get_space_group_info()[1]
    
    # 5. Partially occupied positions
    partial_occ_site_species = set()
    partial_occ_elements_set = set()
    for site in structure.sites:
        # Check if site has multiple species or occupancy < 1
        if len(site.species) > 1 or any(occ < 0.99 for occ in site.species.values()):
            for element, occ in site.species.items():
                if occ < 0.99:
                    partial_occ_site_species.add(str(element))
                    partial_occ_elements_set.add(element.symbol)

    info['partial_occ_site_species'] = sorted(partial_occ_site_species) if partial_occ_site_species else None
    info['partial_occ_elements_set'] = sorted(partial_occ_elements_set) if partial_occ_elements_set else None
    info['occupancies_rescaling'] = occupancies_rescaling
    
    # 6. Temperature (look for common CIF tags)
    temp_tags = ['_diffrn_ambient_temperature', '_cell_measurement_temperature']
    info['temperature'] = None
    for tag in temp_tags:
        if tag in cif_dict:
            info['temperature'] = cif_dict[tag]
            break
    
    # 7. Pressure (look for common CIF tags)
    pressure_tags = ['_diffrn_ambient_pressure', '_cell_measurement_pressure']
    info['pressure'] = None
    for tag in pressure_tags:
        if tag in cif_dict:
            info['pressure'] = cif_dict[tag]
            break
    
    # 8. Chemical formula tags (structural)
    composition_tags = ['_chemical_formula_structural', '_chemical_formula_moiety', '_chemical_formula_iupac']
    info['chemical_formula_structural'] = None
    for tag in composition_tags:
        if tag in cif_dict:
            info['chemical_formula_structural'] = cif_dict[tag]
            break
    
    # 9. Chemical formula tags (sum)
    sum_tags = ['_chemical_formula_sum', '_cod_original_formula_sum']
    info['chemical_formula_sum'] = None
    for tag in sum_tags:
        if tag in cif_dict:
            info['chemical_formula_sum'] = cif_dict[tag]
            break
    
    # 10. Publication year
    year_tags = ['_journal_year']
    info['journal_year'] = None
    for tag in year_tags:
        if tag in cif_dict:
            info['journal_year'] = cif_dict[tag]
            break
    
    # 11. R-factor
    rf_tags = ['_refine_ls_R_factor_all', '_refine_ls_R_factor_gt', '_refine_ls_R_factor_obs']
    info['Rf'] = None
    for tag in rf_tags:
        if tag in cif_dict:
            info['Rf'] = cif_dict[tag]
            break

    # 12. Cell volume
    rf_tags = ['_cell_volume']
    info['V'] = None
    for tag in rf_tags:
        if tag in cif_dict:
            info['V'] = cif_dict[tag]
            break
    
    # 13. C-C and C-H bond detection
    info['CC_present'] = 0
    info['CH_present'] = 0
    
    # Check if structure contains C and/or H
    elements = structure.symbol_set
    has_carbon = 'C' in elements
    has_hydrogen = 'H' in elements
    
    if has_carbon:
        # Get all C and H sites
        c_sites = [site for site in structure.sites if site.is_ordered and site.specie.symbol == 'C']
        h_sites = [site for site in structure.sites if site.is_ordered and site.specie.symbol == 'H']
        
        # Check C-C distances (threshold: 1.6 Angstrom)
        if len(c_sites) > 1:
            for i, site1 in enumerate(c_sites):
                for site2 in c_sites[i+1:]:
                    distance = site1.distance(site2)
                    if distance < 1.7:
                        info['CC_present'] = 1
                        break
                if info['CC_present'] == 1:
                    break
        
        # Check C-H distances (threshold: 1.1 Angstrom)
        if has_hydrogen and len(h_sites) > 0:
            for c_site in c_sites:
                for h_site in h_sites:
                    distance = c_site.distance(h_site)
                    if distance < 1.2:
                        info['CH_present'] = 1
                        break
                if info['CH_present'] == 1:
                    break
    
    return info


def process_single_cif(cif_file):
    """
    Process a single CIF file (to be used in parallel processing).
    
    Args:
        cif_file: Path to the CIF file
        
    Returns:
        dict: Row dictionary for CSV
    """
    try:
        info = extract_cif_info(cif_file)
        
        try:
            disorder = Disorder(cif_file, radius_file='data/all_radii.csv')
            # making classification
            disorder_class = '_'.join(sorted(set(disorder.classify()['orbit_disorder'])))
            disorder_class_error = ''
        except Exception as e:
            disorder_class = ''
            disorder_class_error = str(e)               
        
        # Prepare row for CSV
        row = {
            'filename': cif_file.name,
            'status': 1,
            'occupancies_rescaling': info['occupancies_rescaling'],
            'composition': info['composition'],
            'chemical_system': info['chemical_system'],
            'num_sites': info['num_sites'],
            'space_group_number': info['space_group_number'],
            'partial_occ_site_species': ', '.join(info['partial_occ_site_species']) if info['partial_occ_site_species'] else '',
            'partial_occ_elements_set': '-'.join(info['partial_occ_elements_set']) if info['partial_occ_elements_set'] else '',
            'temperature': info['temperature'] if info['temperature'] else '',
            'pressure': info['pressure'] if info['pressure'] else '',
            'chemical_formula_structural': info['chemical_formula_structural'] if info['chemical_formula_structural'] else '',
            'chemical_formula_sum': info['chemical_formula_sum'] if info['chemical_formula_sum'] else '',
            'journal_year': info['journal_year'] if info['journal_year'] else '',
            'Rf': info['Rf'] if info['Rf'] else '',
            'V': info['V'] if info['V'] else '',
            'CC_present': info['CC_present'],
            'CH_present': info['CH_present'],
            'error': '',
            'disorder_class_error': disorder_class_error,
            'disorder_class': disorder_class,
        }
        success = True
        
    except Exception as e:
        row = {
            'filename': cif_file.name,
            'status': 0,
            'occupancies_rescaling': '',
            'composition': '',
            'chemical_system': '',
            'num_sites': '',
            'space_group_number': '',
            'partial_occ_site_species': '',
            'partial_occ_elements_set': '',
            'temperature': '',
            'pressure': '',
            'chemical_formula_structural': '',
            'chemical_formula_sum': '',
            'journal_year': '',
            'Rf': '',
            'V': '',
            'CC_present': '',
            'CH_present': '',
            'error': str(e),
            'disorder_class_error': '',
            'disorder_class': '',
        }
        success = False
    
    return row, success


def process_folder(folder_path, output_file, n_processes=None, verbose=False, timeout=30):
    """
    Process all CIF files in a folder in parallel and save results to CSV.
    
    Args:
        folder_path: Path to folder containing CIF files
        output_file: Path to output CSV file
        n_processes: Number of parallel processes (default: CPU count)
        verbose: If True, print verbose output
        timeout: Timeout in seconds for each file (default: 30)
    """
    folder = Path(folder_path)
    
    # Find all CIF files
    cif_files = list(folder.glob('**/*cif'))  # Process all files in all subfolders if present of the target folder
    
    if not cif_files:
        print(f"Warning: No CIF files found in '{folder_path}'", flush=True)
        sys.exit(0)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = 4
    
    print(f"Found {len(cif_files)} CIF file(s) in '{folder_path}'", flush=True)
    print(f"Using {n_processes} parallel processes", flush=True)
    print(f"Timeout per file: {timeout} seconds", flush=True)
    print(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Start timing
    start_time = time.time()
    
    # Process files in parallel with timeout
    results = []
    successful = 0
    failed = 0
    timed_out = 0
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_cif, cif_file): cif_file 
                         for cif_file in cif_files}
        
        for i, future in enumerate(as_completed(future_to_file), 1):
            cif_file = future_to_file[future]
            try:
                row, success = future.result(timeout=timeout)
                results.append(row)
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except FuturesTimeoutError:
                # Handle timeout - create error row
                timed_out += 1
                row = {
                    'filename': cif_file.name,
                    'status': 0,
                    'occupancies_rescaling': '',
                    'composition': '',
                    'chemical_system': '',
                    'num_sites': '',
                    'space_group_number': '',
                    'partial_occ_site_species': '',
                    'partial_occ_elements_set': '',
                    'temperature': '',
                    'pressure': '',
                    'chemical_formula_structural': '',
                    'chemical_formula_sum': '',
                    'journal_year': '',
                    'Rf': '',
                    'V': '',
                    'CC_present': '',
                    'CH_present': '',
                    'error': f'Processing timeout ({timeout}s)',
                    'disorder_class_error': '',
                    'disorder_class': '',
                }
                results.append(row)
                
            except Exception as e:
                # Handle any other exception
                failed += 1
                row = {
                    'filename': cif_file.name,
                    'status': 0,
                    'occupancies_rescaling': '',
                    'composition': '',
                    'chemical_system': '',
                    'num_sites': '',
                    'space_group_number': '',
                    'partial_occ_site_species': '',
                    'partial_occ_elements_set': '',
                    'temperature': '',
                    'pressure': '',
                    'chemical_formula_structural': '',
                    'chemical_formula_sum': '',
                    'journal_year': '',
                    'Rf': '',
                    'V': '',
                    'CC_present': '',
                    'CH_present': '',
                    'error': str(e),
                    'disorder_class_error': '',
                    'disorder_class': '',
                }
                results.append(row)
            
            if i % 1000 == 0:
                print(f"Processed {i} files ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    
    # Final timing report
    total_time = time.time() - start_time
    
    # Save results to CSV
    output_path = Path(output_file)
    fieldnames = [
        'filename', 'status', 'occupancies_rescaling', 'composition', 'chemical_system',
        'num_sites', 'space_group_number',
        'partial_occ_site_species', 'partial_occ_elements_set',
        'temperature', 'pressure', 
        'chemical_formula_structural', 'chemical_formula_sum',
        'journal_year', 'Rf', 'V', 'CC_present', 'CH_present',
        'error', 'disorder_class_error', 'disorder_class'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Processing complete!", flush=True)
    print(f"Total time: {total_time:.1f} sec", flush=True)
    print(f"Successfully processed: {successful}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Timed out: {timed_out}", flush=True)
    if len(cif_files) > 0:
        print(f"Average time per file: {total_time/len(cif_files):.2f}s", flush=True)
        print(f"Effective files/second: {len(cif_files)/total_time:.2f}", flush=True)
    print(f"Results saved to: {output_path.absolute()}", flush=True)
    print(f"{'='*60}", flush=True)
    
    return results


def main():
    """Main function to process CIF files."""
    parser = argparse.ArgumentParser(
        description='Extract information from COD CIF files in a folder and save to CSV (parallel processing).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s /path/to/cif/folder
  %(prog)s /path/to/cif/folder -o results.csv
  %(prog)s /path/to/cif/folder -o results.csv -n 16
  %(prog)s /path/to/cif/folder -o results.csv -n 16 -t 60
  %(prog)s /path/to/cif/folder -o results.csv -n 16 -t 30 -v
        '''
    )
    
    parser.add_argument(
        'folder',
        help='Path to folder containing CIF files'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='cif_analysis.csv',
        help='Output CSV file name (default: cif_analysis.csv)'
    )
    
    parser.add_argument(
        '-n', '--n-processes',
        type=int,
        default=None,
        help=f'Number of parallel processes (default: 4)'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=30,
        help='Timeout in seconds for each CIF file (default: 30)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    process_folder(args.folder, args.output, args.n_processes, args.verbose, args.timeout)


if __name__ == "__main__":
    main()
