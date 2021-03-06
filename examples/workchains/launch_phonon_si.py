# from aiida import load_dbenv, is_dbenv_loaded
# if not is_dbenv_loaded():
#    load_dbenv()

from aiida.orm import CalculationFactory, DataFactory, WorkflowFactory
from aiida.work.run import run, submit
from aiida.orm.data.structure import StructureData
from aiida.orm.data.base import Str, Float, Bool, Int

KpointsData = DataFactory("array.kpoints")
ParameterData = DataFactory('parameter')


# Define structure
import numpy as np

# Silicon structure
a = 5.404
cell = [[a, 0, 0],
        [0, a, 0],
        [0, 0, a]]

symbols=['Si'] * 8
scaled_positions = [(0.875,  0.875,  0.875),
                    (0.875,  0.375,  0.375),
                    (0.375,  0.875,  0.375),
                    (0.375,  0.375,  0.875),
                    (0.125,  0.125,  0.125),
                    (0.125,  0.625,  0.625),
                    (0.625,  0.125,  0.625),
                    (0.625,  0.625,  0.125)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])


# Machine
machine_dict = {'resources': {'num_machines': 1,
                              'parallel_env': 'smp',
                              'tot_num_mpiprocs': 1},
                'max_wallclock_seconds': 3600 * 10,
                'queue_name': 'iqtc04.q',
                'import_sys_environment': False
                }


# PHONOPY settings
ph_settings = ParameterData(dict={'supercell': [[2, 0, 0],
                                                [0, 2, 0],
                                                [0, 0, 2]],
                                  'primitive': [[0.0, 0.5, 0.5],
                                                [0.5, 0.0, 0.5],
                                                [0.5, 0.5, 0.0]],
                                  'distance': 0.01,
                                  'mesh': [20, 20, 20],
                                  'symmetry_precision': 1e-5,
                                  # Uncomment to use remote phonopy to calculate the Force constants
                                  # 'code_fc': 'phonopy@stern_outside'
                                  # 'machine': machine_dict
                                  })

# code_to_use = 'VASP'
#code_to_use = 'QE'
code_to_use = 'LAMMPS'

# VASP SPECIFIC
if code_to_use == 'VASP':
    incar_dict = {
        'NELMIN' : 5,
        'NELM'   : 100,
        'ENCUT'  : 400,
        'ALGO'   : 38,
        'ISMEAR' : 0,
        'SIGMA'  : 0.01,
        'GGA'    : 'PS'
    }

    settings_dict = {'code': {'optimize': 'vasp@stern_in',
                              'forces': 'vasp@stern_in'},
                     'parameters': incar_dict,
                     'kpoints_density': 0.5,  # k-point density,
                     'pseudos_family': 'pbe_test_family',
                     'family_folder': '/Users/abel/VASP/test_paw/',
                     'machine': machine_dict
                     }

    # pseudos = ParameterData(dict=potcar.as_dict())
    es_settings = ParameterData(dict=settings_dict)



# QE SPECIFIC
if code_to_use == 'QE':
    parameters_dict = {
        'SYSTEM': {'ecutwfc': 30.,
                   'ecutrho': 200.,},
        'ELECTRONS': {'conv_thr': 1.e-6,}
    }

    settings_dict = {'code': {'optimize': 'pw6@boston_in',
                              'forces': 'pw6@boston_in'},
                     'parameters': parameters_dict,
                     'kpoints_density': 0.5,  # k-point density
                     'pseudos_family': 'pbe_test_family',
                     'machine': machine_dict
                     }

    es_settings = ParameterData(dict=settings_dict)


# LAMMPS SPECIFIC
if code_to_use == 'LAMMPS':
    # Silicon(C) Tersoff
    tersoff_si = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}

    potential ={'pair_style': 'tersoff',
                'data': tersoff_si}

    parameters = {'units': 'metal',
                  'relax': {
                      'type': 'tri',  # iso/aniso/tri
                       'pressure': 0.0,  # bars
                       'vmax': 0.000001,  # Angstrom^3
                  },
                  'minimize': {
                      'energy_tolerance': 1.0e-25,  # eV
                      'force_tolerance': 1.0e-25,  # eV angstrom
                      'max_evaluations': 1000000,
                      'max_iterations': 500000
                  },
                  'lammps_version': '28 Jun 2014'
                  }

    settings_dict = {'code': {'optimize': 'lammps_optimize@iqtc',
                              'forces': 'lammps_force@iqtc'},
                     'parameters': parameters,
                     'potential': potential,
                     'machine': machine_dict
                     }

    es_settings = ParameterData(dict=settings_dict)

PhononPhonopy = WorkflowFactory('phonopy.phonon')

# Chose how to run the calculation
run_by_deamon = False
if not run_by_deamon:
    result = run(PhononPhonopy,
                 structure=structure,
                 es_settings=es_settings,
                 ph_settings=ph_settings,
                 # Optional settings
                 pressure=Float(0.0),
                 optimize=Bool(False),
                 use_nac=Bool(False),
                 )

    print (result)
else:
    future = submit(PhononPhonopy,
                    structure=structure,
                    es_settings=es_settings,
                    ph_settings=ph_settings,
                    # Optional settings
                    pressure=Float(0),
                    optimize=Bool(False),
                    use_nac=Bool(False),
                    )

    print future
    print('Running workchain with pk={}'.format(future.pid))