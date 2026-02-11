import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.groups import SpaceGroup

from typing import Union
import re
import warnings


class Read_CIF:
    """
    A class to read cif file using pymatgen's CifParser internally.
    
    Provides the same interface as the original implementation:
    - read_formula: read sum formula from cif
    - read_id: reads collection code from cif
    - cell: read cell info: a,b,c,alpha,beta,gamma
    - z: read number of formula units Z
    - space_group: reads space group number
    - symmetry(): reads symmetry operations from cif
    - orbits(): reads orbit information from cif
    - positions(): produces the dataframe with the list of sites
    - return_errors(): returns list of errors
    - add_error(): adds an error to the list
    """
    
    def __init__(self, file: str, occ_tol: float = 1.05):
        self.error = []
        self.occ_tol = occ_tol
        self._file = file
        self._cif_data = None
        self._structure = None
        self._cif_dict = None
        
        try:
            # Use pymatgen's CifParser for robust parsing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser = CifParser(file, occupancy_tolerance=occ_tol)
                
                # Get the raw CIF dictionary (preserves all tags)
                self._cif_dict = parser.as_dict()
                
                # Get the first (usually only) data block
                self._block_key = list(self._cif_dict.keys())[0]
                self._cif_data = self._cif_dict[self._block_key]
                
                # Try to get structure (may fail for some CIFs)
                try:
                    structures = parser.get_structures(primitive=False)
                    if structures:
                        self._structure = structures[0]
                except Exception as e:
                    self.error.append(f'Structure generation failed: {str(e)}')
                    
        except Exception as e:
            self.error.append(f'CIF parsing failed: {str(e)}')
    
    def _get_cif_value(self, key, default=None):
        """Helper to safely get a value from the CIF dictionary."""
        if self._cif_data is None:
            return default
        return self._cif_data.get(key, default)
    
    def _parse_number(self, value):
        """Parse a CIF number, handling parenthetical uncertainties like '5.232(3)'."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        # Remove parenthetical uncertainty
        s = str(value).split('(')[0].strip()
        if s in ['.', '-.', '?', '']:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    @property
    def read_formula(self):
        """Reading sum formula"""
        formula = self._get_cif_value('_chemical_formula_sum', '')
        if isinstance(formula, list):
            formula = formula[0] if formula else ''
        # Clean up quotes
        formula = str(formula).replace('"', '').replace("'", '').strip()
        return formula
    
    @property
    def read_id(self):
        """Reading database collection code"""
        # Try various ID fields
        for key in ['_database_code_ICSD', '_cod_database_code', '_database_code_amcsd']:
            val = self._get_cif_value(key)
            if val is not None:
                if isinstance(val, list):
                    val = val[0]
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        return None
    
    @property
    def cell(self):
        """Reading cell info. Returns lattice periods and angles."""
        if self._structure is not None:
            lattice = self._structure.lattice
            return {
                'abc': [lattice.a, lattice.b, lattice.c],
                'angles': [lattice.alpha, lattice.beta, lattice.gamma]
            }
        
        # Fallback to parsing CIF directly
        a = self._parse_number(self._get_cif_value('_cell_length_a'))
        b = self._parse_number(self._get_cif_value('_cell_length_b'))
        c = self._parse_number(self._get_cif_value('_cell_length_c'))
        alpha = self._parse_number(self._get_cif_value('_cell_angle_alpha'))
        beta = self._parse_number(self._get_cif_value('_cell_angle_beta'))
        gamma = self._parse_number(self._get_cif_value('_cell_angle_gamma'))
        
        if a == 0 and b == 0 and c == 0:
            self.error.append('No unit cell in CIF')
            return None
            
        return {
            'abc': [a, b, c],
            'angles': [alpha, beta, gamma]
        }

    @property    
    def z(self):
        """Read number of formula units Z"""
        val = self._get_cif_value('_cell_formula_units_Z')
        if val is None:
            self.error.append('No Z in CIF')
            return None
        if isinstance(val, list):
            val = val[0]
        try:
            return int(float(str(val).split('(')[0]))
        except (ValueError, TypeError):
            self.error.append('No Z in CIF')
            return None

    @property    
    def space_group(self):
        """Read space group number"""
        # Try to get from structure first
        if self._structure is not None:
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(self._structure)
                return sga.get_space_group_number()
            except:
                pass
        
        # Try direct CIF fields
        for key in ['_space_group_IT_number', '_symmetry_Int_Tables_number']:
            val = self._get_cif_value(key)
            if val is not None:
                if isinstance(val, list):
                    val = val[0]
                try:
                    return int(float(str(val).split('(')[0]))
                except (ValueError, TypeError):
                    pass
        
        # Try to derive from H-M symbol
        for key in ['_symmetry_space_group_name_H-M', '_space_group_name_H-M_alt']:
            val = self._get_cif_value(key)
            if val is not None:
                if isinstance(val, list):
                    val = val[0]
                try:
                    # Clean up the symbol
                    symbol = str(val).replace("'", "").replace('"', '').strip()
                    sg = SpaceGroup(symbol)
                    return sg.int_number
                except:
                    pass
        
        self.error.append('No space group number in CIF')
        return None
   
    def symmetry(self):
        """Reading symmetry operations from CIF.
        
        Returns:
            list: Symmetry operations as [x_op, y_op, z_op] strings
                  e.g., [['x', 'y', 'z'], ['-x', '-y', 'z'], ...]
                  Returns None if no symmetry operations found
        """
        symops = []
        
        # Try various symmetry operation field names
        symop_keys = [
            '_space_group_symop_operation_xyz',
            '_space_group_symop.operation_xyz', 
            '_symmetry_equiv_pos_as_xyz',
        ]
        
        for key in symop_keys:
            val = self._get_cif_value(key)
            if val is not None:
                if not isinstance(val, list):
                    val = [val]
                for op_str in val:
                    parsed = self._parse_symmetry_operation(str(op_str))
                    if parsed:
                        symops.append(parsed)
                if symops:
                    return symops
        
        # Fallback: try to get from structure's space group
        if self._structure is not None:
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(self._structure)
                for op in sga.get_symmetry_operations():
                    # Convert rotation matrix and translation to string representation
                    rot = op.rotation_matrix
                    trans = op.translation_vector
                    xyz_ops = self._matrix_to_xyz_string(rot, trans)
                    if xyz_ops:
                        symops.append(xyz_ops)
                if symops:
                    return symops
            except:
                pass
        
        self.error.append('No symops in CIF')
        return None
    
    def _parse_symmetry_operation(self, symop_str):
        """Parse symmetry operation string into x, y, z components."""
        try:
            symop_str = symop_str.strip().replace(' ', '')
            components = symop_str.split(',')
            if len(components) == 3:
                return [c.strip() for c in components]
        except:
            pass
        return None
    
    def _matrix_to_xyz_string(self, rot, trans):
        """Convert rotation matrix and translation to xyz string representation."""
        try:
            xyz = ['x', 'y', 'z']
            result = []
            for i in range(3):
                terms = []
                for j in range(3):
                    coef = rot[i, j]
                    if abs(coef) > 0.5:
                        sign = '' if coef > 0 and not terms else ('+' if coef > 0 else '-')
                        terms.append(f"{sign}{xyz[j]}" if abs(coef) == 1 else f"{sign}{coef:.0f}*{xyz[j]}")
                # Add translation
                t = trans[i]
                if abs(t) > 1e-6:
                    # Convert to fraction
                    from fractions import Fraction
                    frac = Fraction(t).limit_denominator(12)
                    if frac.numerator != 0:
                        sign = '+' if frac > 0 else ''
                        terms.append(f"{sign}{frac}")
                result.append(''.join(terms) if terms else '0')
            return result
        except:
            return None

    def orbits(self):
        """Reading atomic unique positions with respect to symmetry operations (orbits).
        
        Returns:
            DataFrame with columns: atom_site_label, atom_site_type_symbol,
            atom_site_symmetry_multiplicity, atom_site_Wyckoff_symbol,
            atom_site_fract_x, atom_site_fract_y, atom_site_fract_z, atom_site_occupancy
        """
        if self._cif_data is None:
            self.error.append('No structure in CIF (ReadCIF.orbits)')
            return None
        
        # Get atom site data from CIF
        labels = self._get_cif_value('_atom_site_label')
        if labels is None:
            self.error.append('No structure in CIF (ReadCIF.orbits)')
            return None
        
        if not isinstance(labels, list):
            labels = [labels]
        
        n_atoms = len(labels)
        
        # Get fractional coordinates
        fract_x = self._get_cif_value('_atom_site_fract_x', [0.0] * n_atoms)
        fract_y = self._get_cif_value('_atom_site_fract_y', [0.0] * n_atoms)
        fract_z = self._get_cif_value('_atom_site_fract_z', [0.0] * n_atoms)
        
        if not isinstance(fract_x, list):
            fract_x = [fract_x]
        if not isinstance(fract_y, list):
            fract_y = [fract_y]
        if not isinstance(fract_z, list):
            fract_z = [fract_z]
        
        # Get optional fields
        type_symbols = self._get_cif_value('_atom_site_type_symbol')
        multiplicities = self._get_cif_value('_atom_site_symmetry_multiplicity')
        wyckoffs = self._get_cif_value('_atom_site_Wyckoff_symbol')
        occupancies = self._get_cif_value('_atom_site_occupancy')
        
        # Build DataFrame
        rows = []
        for i in range(n_atoms):
            label = labels[i]
            
            # Type symbol
            if type_symbols is not None:
                ts_list = type_symbols if isinstance(type_symbols, list) else [type_symbols]
                type_sym = ts_list[i] if i < len(ts_list) else self._extract_element_from_label(label)
            else:
                type_sym = self._extract_element_from_label(label)
            
            # Multiplicity
            if multiplicities is not None:
                mult_list = multiplicities if isinstance(multiplicities, list) else [multiplicities]
                mult = int(mult_list[i]) if i < len(mult_list) else 1
            else:
                mult = 1
            
            # Wyckoff symbol
            if wyckoffs is not None:
                wyck_list = wyckoffs if isinstance(wyckoffs, list) else [wyckoffs]
                wyck = wyck_list[i] if i < len(wyck_list) else '?'
            else:
                wyck = '?'
            
            # Occupancy
            if occupancies is not None:
                occ_list = occupancies if isinstance(occupancies, list) else [occupancies]
                occ = self._parse_number(occ_list[i]) if i < len(occ_list) else 1.0
            else:
                occ = 1.0
            
            if occ > self.occ_tol:
                self.error.append('Occupancies outside the tolerance')
            
            rows.append({
                'atom_site_label': label,
                'atom_site_type_symbol': type_sym,
                'atom_site_symmetry_multiplicity': mult,
                'atom_site_Wyckoff_symbol': wyck,
                'atom_site_fract_x': self._parse_number(fract_x[i]),
                'atom_site_fract_y': self._parse_number(fract_y[i]),
                'atom_site_fract_z': self._parse_number(fract_z[i]),
                'atom_site_occupancy': occ
            })
        
        columns = ['atom_site_label', 'atom_site_type_symbol', 'atom_site_symmetry_multiplicity',
                   'atom_site_Wyckoff_symbol', 'atom_site_fract_x', 'atom_site_fract_y',
                   'atom_site_fract_z', 'atom_site_occupancy']
        
        return pd.DataFrame(rows, columns=columns)

    def _extract_element_from_label(self, label):
        """Extract element symbol from atom site label."""
        two_letter_elements = {
            'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc',
            'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
            'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
            'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
            'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        }
        
        clean_label = str(label).strip()
        if not clean_label:
            return 'X'
        
        if len(clean_label) >= 2:
            first_two = clean_label[0].upper() + clean_label[1].lower()
            if first_two in two_letter_elements:
                return first_two
        
        if clean_label[0].isalpha():
            return clean_label[0].upper()
        
        return 'X'

    def positions(self, orbits: Union[pd.DataFrame, None], symops: list, pystruct: bool=False,
                  merge_sites: bool=False, merge_tol: float=1e-2, r: int=3, dist_tol: float=1e-2):
        """
        Function creating positions from orbits and symmetry operations.
        
        Input: 
            orbits - pandas dataframe describing orbits (output of self.orbits())
            symops - list[str] list of symmetry operations from CIF
            pystruct - bool, whether to output pymatgen structure object
            merge_sites - bool, whether to merge closely located sites
            merge_tol - float, distance tolerance for merging sites (angstroms)
            r - int, number of digits to retain for coordinates
            dist_tol - float, distance tolerance to determine if two points are the same
        """
        
        def pbc(n):
            if n == -0.0:
                n = 0.0
            elif n < 0:
                n = n + 1.0
            elif n >= 1.0:
                n = n - 1.0
            return n
        
        def distance(x1, x2, lattice, space_num):
            if space_num < 16 or (space_num > 142 and space_num < 195):
                x1c = np.dot(x1, lattice)
                x2c = np.dot(x2, lattice)
                x1images = np.zeros((27, 3))
                x2images = np.zeros((27, 3))

                for i in range(3):
                    vec = np.array([[0, lattice[0][i], -lattice[0][i], lattice[1][i], -lattice[1][i], lattice[2][i], -lattice[2][i],
                        lattice[0][i]+lattice[1][i], -lattice[0][i]-lattice[1][i], lattice[0][i]-lattice[1][i], -lattice[0][i]+lattice[1][i],
                        lattice[2][i]+lattice[1][i], -lattice[2][i]-lattice[1][i], lattice[2][i]-lattice[1][i], -lattice[2][i]+lattice[1][i],
                        lattice[0][i]+lattice[2][i], -lattice[0][i]-lattice[2][i], lattice[0][i]-lattice[2][i], -lattice[0][i]+lattice[2][i],
                        lattice[0][i]+lattice[1][i]+lattice[2][i], -lattice[0][i]-lattice[1][i]-lattice[2][i],
                        -lattice[0][i]+lattice[1][i]+lattice[2][i], lattice[0][i]-lattice[1][i]+lattice[2][i], lattice[0][i]+lattice[1][i]-lattice[2][i],
                        -lattice[0][i]-lattice[1][i]+lattice[2][i], -lattice[0][i]+lattice[1][i]-lattice[2][i], lattice[0][i]-lattice[1][i]-lattice[2][i]]])

                    x1images[:, i] = x1c[i] * np.ones(27) + vec
                    x2images[:, i] = x2c[i] * np.ones(27) + vec

                d = cdist(x1images, x2images, 'euclidean')
                dist = np.min(d)
            else:
                d = np.zeros(3)
                for i in range(3):
                    if x1[i] - x2[i] > 0.5:
                        d[i] = x1[i] - x2[i] - 1.0
                    elif x1[i] - x2[i] < -0.5:
                        d[i] = x1[i] - x2[i] + 1.0
                    else:
                        d[i] = x1[i] - x2[i]
                dc = np.dot(d, lattice)
                dist = np.sqrt(dc[0]**2 + dc[1]**2 + dc[2]**2)

            return dist         

        def check_present(new, new_sites, lattice, space_num, tol=dist_tol):
            switch = False
            for xyz in new_sites:
                truth_string = []
                for i in [0, 1, 2, 3, 7]:
                    if new[i] == xyz[i]:
                        truth_string.append(True)
                    else:
                        truth_string.append(False)
                r2 = distance([new[4], new[5], new[6]], [xyz[4], xyz[5], xyz[6]], lattice, space_num)
                if r2 < tol:
                    truth_string.append(True)
                else:
                    truth_string.append(False)
                if set(truth_string) == {True}:
                    switch = True
            return switch
        
        def check_present_x(new, coords, lattice, space_num, tol=dist_tol):
            switch = False
            for xyz in coords:
                r2 = distance(new, xyz, lattice, space_num)
                if r2 < tol:
                    switch = True
            return switch

        def correct_rounding(orb, lattice, space_num, r=r, tol=dist_tol):
            columns = orb.columns.values
            site_label = orb.iloc[0]['atom_site_label']
            site_type = orb.iloc[0]['atom_site_type_symbol']
            multi = orb.iloc[0]['atom_site_symmetry_multiplicity']
            Wyckoff = orb.iloc[0]['atom_site_Wyckoff_symbol']
            occup = orb.iloc[0]['atom_site_occupancy']
            
            new_coord = []
            for i in range(len(orb)):
                coord = np.zeros(3)
                coord[0] = pbc(round(orb.iloc[i]['atom_site_fract_x'], r))
                coord[1] = pbc(round(orb.iloc[i]['atom_site_fract_y'], r))
                coord[2] = pbc(round(orb.iloc[i]['atom_site_fract_z'], r))
                
                new = [site_label, site_type, multi, Wyckoff, coord[0], coord[1], coord[2], occup]
                if not check_present(new, new_coord, lattice, space_num, tol=tol):
                    new_coord.append(new)
           
            new_orb = pd.DataFrame(new_coord, columns=columns) 
            return new_orb
        
        columns = ['atom_site_label', 'atom_site_type_symbol', 'atom_site_symmetry_multiplicity', 'atom_site_Wyckoff_symbol',
             'atom_site_fract_x', 'atom_site_fract_y', 'atom_site_fract_z', 'atom_site_occupancy']
        new_sites = []
        coords = []
        error_orbits = []
        
        space_num = self.space_group
        if space_num is None:
            space_num = 1  # Default to P1 if unknown
            
        lattice = Lattice.from_parameters(
            a=self.cell['abc'][0], b=self.cell['abc'][1], c=self.cell['abc'][2],
            alpha=self.cell['angles'][0], beta=self.cell['angles'][1], gamma=self.cell['angles'][2]
        )
        
        for i in range(len(orbits)):
            site_label = orbits.iloc[i]['atom_site_label']
            site_type = orbits.iloc[i]['atom_site_type_symbol']
            multi = orbits.iloc[i]['atom_site_symmetry_multiplicity']
            Wyckoff = orbits.iloc[i]['atom_site_Wyckoff_symbol']
            occup = round(orbits.iloc[i]['atom_site_occupancy'], 3)

            x = pbc(round(orbits.iloc[i]['atom_site_fract_x'], 6))
            y = pbc(round(orbits.iloc[i]['atom_site_fract_y'], 6))
            z = pbc(round(orbits.iloc[i]['atom_site_fract_z'], 6))
            new = [site_label, site_type, multi, Wyckoff, x, y, z, occup]
            coords.append([x, y, z])
            new_sites.append(new)
            calc_multi = 1
            for sym in symops:
                new_x = pbc(round(eval(sym[0]), 6))
                new_y = pbc(round(eval(sym[1]), 6))
                new_z = pbc(round(eval(sym[2]), 6))
                new = [site_label, site_type, multi, Wyckoff, new_x, new_y, new_z, occup]
                if not check_present(new, new_sites, lattice.matrix, space_num, tol=dist_tol):
                    new_sites.append(new)
                    calc_multi += 1
            if calc_multi != multi:
                self.error.append(f'Multiplicity problem, nominal = {multi}, calculated = {calc_multi}, orbit_label: {site_label}')
                error_orbits.append(site_label)
        
        positions = pd.DataFrame(new_sites, columns=columns)
        print(positions.head(2))
        
        if len(error_orbits) > 0:
            for label in error_orbits:
                orb = positions.loc[positions['atom_site_label'] == label]
                ind = orb.index.values
                new_orb = correct_rounding(orb, lattice.matrix, space_num, r=3, tol=dist_tol)
                positions.drop(index=ind, inplace=True)
                positions = pd.concat([positions, new_orb])
                positions.reset_index(drop=True, inplace=True)

        if merge_sites == False and pystruct == False:
            return positions
        elif merge_sites == False and pystruct == True:
            species = []
            for i, sp in enumerate(positions['atom_site_type_symbol'].values):
                if positions['atom_site_occupancy'].values[i] < self.occ_tol and positions['atom_site_occupancy'].values[i] > 1.0:
                    positions['atom_site_occupancy'].values[i] = 1.0
                comp = {sp: positions['atom_site_occupancy'].values[i]}
                species.append(Composition(comp))

            x = (np.array([positions['atom_site_fract_x'].values])).T
            y = (np.array([positions['atom_site_fract_y'].values])).T
            z = (np.array([positions['atom_site_fract_z'].values])).T

            coords = np.concatenate([x, y, z], axis=1)
            struct = Structure(lattice=lattice, species=species,
                               coords=coords, labels=list(positions['atom_site_label'].values))
            return positions, struct
        
        elif merge_sites == True:
            species = []
            coords = []
            new_sites = []
            for i in range(len(positions)):
                linei = positions.iloc[i]
                comp = {linei['atom_site_type_symbol']: linei['atom_site_occupancy']}
                lab = linei['atom_site_label']
                occ = linei['atom_site_occupancy']
                if i < len(positions) - 1:
                    for j in range(i + 1, len(positions)):
                        linej = positions.iloc[j]
                        r2 = distance(
                            [linei['atom_site_fract_x'], linei['atom_site_fract_y'], linei['atom_site_fract_z']],
                            [linej['atom_site_fract_x'], linej['atom_site_fract_y'], linej['atom_site_fract_z']],
                            lattice.matrix, space_num
                        )
                        if r2 < merge_tol:
                            if linej['atom_site_type_symbol'] not in comp.keys():
                                comp[linej['atom_site_type_symbol']] = linej['atom_site_occupancy']
                            else:
                                comp[linej['atom_site_type_symbol']] += linej['atom_site_occupancy']
                            lab += linej['atom_site_label']
                            occ += linej['atom_site_occupancy']
                            if linei['atom_site_symmetry_multiplicity'] != linej['atom_site_symmetry_multiplicity']:
                                self.add_error('multiplicity of merged sites are different, labels are:' +
                                             linei['atom_site_label'] + ' and ' + linej['atom_site_label'])
                            if linei['atom_site_Wyckoff_symbol'] != linej['atom_site_Wyckoff_symbol']:
                                self.add_error('Wyckoff symbols of merged sites are different, labels are:' +
                                             linei['atom_site_label'] + ' and ' + linej['atom_site_label'])
                
                new = [linei['atom_site_fract_x'], linei['atom_site_fract_y'], linei['atom_site_fract_z']]
                if not check_present_x(new, coords, lattice.matrix, space_num, tol=merge_tol):
                    coords.append(new)
                    if occ > 1.0 and occ <= self.occ_tol:
                        self.add_error(f'occupancy of {lab} is {occ}, rescaled to 1.0')
                        for key, val in comp.items():
                            val = round(float(val) / float(occ), 4)
                            comp[key] = val
                        occ = 1.0
                    if occ > 1.05:
                        self.add_error(f'occupancy of {lab} is {occ}')
                    species.append(Composition(comp))
                    new_sites.append([lab, comp, linei['atom_site_symmetry_multiplicity'],
                                    linei['atom_site_Wyckoff_symbol'], new[0], new[1], new[2], occ])
            
            columns = ['atom_site_label', 'atom_site_type_symbol', 'atom_site_symmetry_multiplicity', 'atom_site_Wyckoff_symbol',
                'atom_site_fract_x', 'atom_site_fract_y', 'atom_site_fract_z', 'atom_site_occupancy']
            positions = pd.DataFrame(new_sites, columns=columns)

            if pystruct == True:
                try:
                    struct = Structure(lattice=lattice, species=species,
                                       coords=coords, labels=list(positions['atom_site_label'].values))
                except:
                    self.add_error('pymatgen Structure failed')
                    struct = None

                return positions, struct
            else:
                return positions

    def return_errors(self):
        return self.error
    
    def add_error(self, err):
        self.error.append(err)
        return
    
    # Keep the old method for backward compatibility
    def _float_brackets(self, ss):
        """Transforms numbers from string of format x.y(z) into float x.yz"""
        return self._parse_number(ss)
