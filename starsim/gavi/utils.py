import numpy as np
import numba as nb # For faster computations
import random
import warnings
import functools
import starsim as ss
import pandas as pd
from starsim import version as ssv
from starsim.settings import options as sso # To set options
from gavi import interventions as ssi
import sciris as sc

__all__ = ['parse_age_range', 'set_seed', 'peak_coverage_filter', 'outbreak_detection_trigger', 'ebola_detection_action']

nbbool  = nb.bool_
nbint   = ss.nbint
nbfloat = ss.nbfloat

cache = sso.numba_cache # Turning this off can help switching parallelization options

def parse_age_range(x):
    if "+" in x:  # Handle "95+"
        age_lower = float(x.split("+")[0])
        age_upper = np.inf
    elif "-" in x:  # Handle "5-9"
        age_lower = float(x.split("-")[0])
        age_upper = float(x.split("-")[1])
    else:  # Handle "5 to 9"
        age_lower = float(x.split("to")[0])
        age_upper = float(x.split("to")[1])
    return age_lower, age_upper

def check_version(expected, die=False, verbose=True):
    '''
    Get current git information and optionally write it to disk. The expected
    version string may optionally start with '>=' or '<=' (== is implied otherwise),
    but other operators (e.g. ~=) are not supported. Note that e.g. '>' is interpreted
    to mean '>='.

    Args:
        expected (str): expected version information
        die (bool): whether or not to raise an exception if the check fails

    **Example**::

        cv.check_version('>=1.7.0', die=True) # Will raise an exception if an older version is used
    '''
    if expected.startswith('>'):
        valid = 1
    elif expected.startswith('<'):
        valid = -1
    else:
        valid = 0 # Assume == is the only valid comparison
    expected = expected.lstrip('<=>') # Remove comparator information
    version = ssv.__version__
    compare = sc.compareversions(version, expected) # Returns -1, 0, or 1
    relation = ['older', '', 'newer'][compare+1] # Picks the right string
    if relation: # Versions mismatch, print warning or raise error
        string = f'Note: Covasim is {relation} than expected ({version} vs. {expected})'
        if die and compare != valid:
            raise ValueError(string)
        elif verbose:
            print(string)
    return compare

def load(*args, **kwargs):
    '''
    Convenience method for sc.loadobj() and equivalent to cv.Sim.load() or
    cv.Scenarios.load().

    Args:
        filename (str): file to load
        args (list): passed to sc.loadobj()
        kwargs (dict): passed to sc.loadobj()

    Returns:
        Loaded object

    **Examples**::

        sim = cv.load('calib.sim') # Equivalent to cv.Sim.load('calib.sim')
        scens = cv.load(filename='school-closures.scens', folder='schools')
    '''
    obj = sc.loadobj(*args, **kwargs)
    if hasattr(obj, 'version'):
        v_curr = ssv.__version__
        v_obj = obj.version
        cmp = check_version(v_obj, verbose=False)
        if cmp != 0:
            print(f'Note: you have Covasim v{v_curr}, but are loading an object from v{v_obj}')
    return obj

def save(*args, **kwargs):
    '''
    Convenience method for sc.saveobj() and equivalent to ss.Sim.save().

    Args:
        filename (str): file to save to
        obj (object): object to save
        args (list): passed to sc.saveobj()
        kwargs (dict): passed to sc.saveobj()

    Returns:
        Filename the object is saved to

    **Examples**::

        ss.save('calib.sim', sim) # Equivalent to sim.save('calib.sim')
        ss.save(filename='school-closures.scens', folder='schools', obj=scens)
    '''
    filepath = sc.saveobj(*args, **kwargs)
    return filepath

def git_info(filename=None, check=False, comments=None, old_info=None, die=False, indent=2, verbose=True, frame=2, **kwargs):
    '''
    Get current git information and optionally write it to disk. Simplest usage
    is ss.git_info(__file__)

    Args:
        filename  (str): name of the file to write to or read from
        check    (bool): whether or not to compare two git versions
        comments (dict): additional comments to include in the file
        old_info (dict): dictionary of information to check against
        die      (bool): whether or not to raise an exception if the check fails
        indent    (int): how many indents to use when writing the file to disk
        verbose  (bool): detail to print
        frame     (int): how many frames back to look for caller info
        kwargs   (dict): passed to sc.loadjson() (if check=True) or sc.savejson() (if check=False)

    **Examples**::

        ss.git_info() # Return information
        ss.git_info(__file__) # Writes to disk
        ss.git_info('stisim_version.gitinfo') # Writes to disk
        ss.git_info('stisim_version.gitinfo', check=True) # Checks that current version matches saved file
    '''

    # Handle the case where __file__ is supplied as the argument
    if isinstance(filename, str) and filename.endswith('.py'):
        filename = filename.replace('.py', '.gitinfo')

    # Get git info
    calling_file = sc.makefilepath(sc.getcaller(frame=frame, tostring=False)['filename'])
    ss_info = {'version':ssv.__version__}
    ss_info.update(sc.gitinfo(__file__, verbose=False))
    caller_info = sc.gitinfo(calling_file, verbose=False)
    caller_info['filename'] = calling_file
    info = {'starsim':ss_info, 'called_by':caller_info}
    if comments:
        info['comments'] = comments

    # Just get information and optionally write to disk
    if not check:
        if filename is not None:
            output = sc.savejson(filename, info, indent=indent, **kwargs)
        else:
            output = info
        return output

    # Check if versions match, and optionally raise an error
    else:
        if filename is not None:
            old_info = sc.loadjson(filename, **kwargs)
        old_ss_info = old_info['starsim'] if 'starsim' in old_info else old_info
        if ss_info != old_ss_info: # pragma: no cover
            string = f'Git information differs: {ss_info} vs. {old_ss_info}'
            if die:
                raise ValueError(string)
            elif verbose:
                print(string)
        return

def set_seed(seed=None):
    '''
    Reset the random seed -- complicated because of Numba, which requires special
    syntax to reset the seed. This function also resets Python's built-in random
    number generated.

    Args:
        seed (int): the random seed
    '''

    @nb.njit((nbint,), cache=cache)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed) # If None, reinitializes it
    if seed is None: # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed) # Finally, reset Python's built-in random number generator, just in case (used by SynthPops)

    return

def result_df(sim):
    '''
    Helper function for making dataframe
    '''
    resdict = sim.export_results(for_json=False)
    result_df = pd.DataFrame.from_dict(resdict)
    return result_df

def peak_coverage_filter(people, vac_peak_coverage: str) -> np.array:
    """
    Account for peak coverage

    - Coverage among 18+ (including 18)

    Args:
        people:
        vac_peak_coverage:

    Returns:

    """
    coverage = [float(x) / 100 for x in vac_peak_coverage.split("_")]
    p_vaccinated = np.full(len(people), fill_value=np.nan)
    p_vaccinated[(people.age >= 18)] = coverage[0]
    eligible = ss.binomial_arr(p_vaccinated)
    #people.vaccine_eligible = eligible
    return eligible

def peak_coverage_filter_measles(people, vac_peak_coverage: str) -> np.array:
    """
    Account for peak coverage

    - Coverage among 6months+

    Args:
        people:
        vac_peak_coverage:

    Returns:

    """
    coverage = [float(x) / 100 for x in vac_peak_coverage.split("_")]
    p_vaccinated = np.full(len(people), fill_value=np.nan)
    p_vaccinated[(people.age >= 0.5)] = coverage[0]
    eligible = ss.binomial_arr(p_vaccinated)
    #people.vaccine_eligible = eligible
    return eligible

def peak_coverage_filter_meningitis(people, vac_peak_coverage: str) -> np.array:
    """
    Account for peak coverage

    - Coverage among 9months+

    Args:
        people:
        vac_peak_coverage:

    Returns:

    """
    coverage = [float(x) / 100 for x in vac_peak_coverage.split("_")]
    p_vaccinated = np.full(len(people), fill_value=np.nan)
    p_vaccinated[(people.age >= 0.75)] = coverage[0]
    eligible = ss.binomial_arr(p_vaccinated)
    #people.vaccine_eligible = eligible
    return eligible

def peak_coverage_filter_yellowfever(people, vac_peak_coverage: str) -> np.array:
    """
    Account for peak coverage

    - Coverage among 9months+

    Args:
        people:
        vac_peak_coverage:

    Returns:

    """
    coverage = [float(x) / 100 for x in vac_peak_coverage.split("_")]
    p_vaccinated = np.full(len(people), fill_value=np.nan)
    p_vaccinated[(people.age >= 0.75)] = coverage[0]
    eligible = ss.binomial_arr(p_vaccinated)
    #people.vaccine_eligible = eligible
    return eligible


def set_intervention_attributes(sim, intervention_name, **kwargs):
    '''
        Workaround for updating intervention parameters during simulation
    '''
    iv = sim.get_intervention(intervention_name)
    for attr, value in kwargs.items():
        assert hasattr(iv, attr), "set_intervention_attributes() should only be used to change existing attributes"  # avoid silent errors if the attr is misspelled
        setattr(iv, attr, value)


def outbreak_detection_trigger(sim, disease, size=0):
    """
    True if a case has been detected
    """
    if disease == 'yellow_fever':
        return sim.diseases[disease].severe.sum() > size
    elif disease == 'meningitis':
        if sim.ti < 7:
            return False
        start_ti = sim.ti - 7
        end_ti = sim.ti
        diagnoses = sum((sim.diseases[disease].ti_diagnosed > start_ti) & (sim.diseases[disease].ti_diagnosed <= end_ti))
        return diagnoses >= size
    else:
        return sim.diseases[disease].diagnosed.sum() > size


#ToDO: update this if ever seeding later than ti=0
def check_ti_trigger(sim, t):
    """
    True if sim.ti == t
    """
    return sim.ti == t

def ebola_detection_action(sim, disease, response_time, num_doses=10, symp_prob=0.02):
    '''
        Function to schedule outbreak response measure upon detection/declaration for Ebola, yellow fever, measles
    '''
    vacc_day = sim.ti + response_time
    ## ADD VACCINATION

    schedule = sim.get_intervention(ssi.EventSchedule, first=True)
    schedule[vacc_day].append(functools.partial(set_intervention_attributes, intervention_name="vacc_rollout", num_doses=num_doses))
    schedule[sim.ti+1].append(functools.partial(set_intervention_attributes, intervention_name="symp_testing", symp_prob=symp_prob, vac_symp_prob=symp_prob))
    sim.results[disease+'-outbreak_detection'] = np.array([sim.ti])
    return

def meningitis_detection_action(sim, disease, response_time, num_doses=10):
    '''
        Function to schedule outbreak response measure upon detection/declaration for meningitis
    '''
    vacc_day = sim.ti + response_time
    ## ADD VACCINATION

    schedule = sim.get_intervention(ssi.EventSchedule, first=True)
    schedule[vacc_day].append(functools.partial(set_intervention_attributes, intervention_name="vacc_rollout", num_doses=num_doses))
    sim.results[disease+'-outbreak_detection'] = np.array([sim.ti])
    return

def cholera_detection_action(sim, disease, response_time, num_doses=10, symp_prob=0.02, new_wash=0.2, new_hygiene=0.24):
    '''
        Function to schedule outbreak response measure upon detection/declaration for cholera
    '''
    vacc_day = sim.ti + response_time
    ## ADD VACCINATION

    schedule = sim.get_intervention(ssi.EventSchedule, first=True)
    schedule[vacc_day].append(functools.partial(set_intervention_attributes, intervention_name="vacc_rollout", num_doses=num_doses))
    schedule[sim.ti + 1].append(functools.partial(set_intervention_attributes, intervention_name="symp_testing", symp_prob=symp_prob,vac_symp_prob=symp_prob))
    schedule[70].append(functools.partial(update_wash_hygiene, disease=disease, new_wash=new_wash, new_hygiene=new_hygiene))
    sim.results[disease+'-outbreak_detection'] = np.array([sim.ti])
    return

def update_wash_hygiene(sim, disease, new_wash, new_hygiene):
    '''
        Small function to update WASH parameters for cholera mid-simulation
    '''
    sim.diseases[disease].pars['WASH_factor'] = new_wash
    sim.diseases[disease].pars['hygiene_factor'] = new_hygiene
    return

def ebola_forced_detection_action(sim, disease, symp_prob=0.02, sensitivity=0.87):
    '''
        Function to handle scenarios where detection is defined to occur on a specific time step
    '''
    infectious_uid = ss.true(sim.diseases[disease].infected)
    infectious_choice = infectious_uid[0]
    sim.diseases[disease].tested[infectious_choice] = True
    sim.diseases[disease].ti_tested[infectious_choice] = sim.ti  # Only keep the last time they tested
    sim.diseases[disease].diagnosed[infectious_choice] = True
    sim.diseases[disease].ti_diagnosed[infectious_choice] = sim.ti
    sim.diseases[disease].ti_pos_test[infectious_choice] = sim.ti
    schedule = sim.get_intervention(ssi.EventSchedule, first=True)
    schedule[sim.ti+1].append(functools.partial(set_intervention_attributes, intervention_name="symp_testing", symp_prob=symp_prob, vac_symp_prob=symp_prob, sensitivity=sensitivity))
    return

def warn(msg, category=None, verbose=None, die=None):
    ''' Helper function to handle warnings -- not for the user '''

    # Handle inputs
    warnopt = sso.warnings if not die else 'error'
    if category is None:
        category = RuntimeWarning
    if verbose is None:
        verbose = sso.verbose

    # Handle the different options
    if warnopt in ['error', 'errors']: # Include alias since hard to remember
        raise category(msg)
    elif warnopt == 'warn':
        msg = '\n' + msg
        warnings.warn(msg, category=category, stacklevel=2)
    elif warnopt == 'print':
        if verbose:
            msg = 'Warning: ' + msg
            print(msg)
    elif warnopt == 'ignore':
        pass
    else:
        options = ['error', 'warn', 'print', 'ignore']
        errormsg = f'Could not understand "{warnopt}": should be one of {options}'
        raise ValueError(errormsg)

    return