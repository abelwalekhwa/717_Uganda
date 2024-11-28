import starsim as ss
from gavi.measles import Measles, MeaslesVaccine
from gavi.networks import RandomNetwork, HouseholdNetwork, generate_household_clusters
from gavi import utils as ssu
from gavi import interventions as ssi
from gavi import analyzers as ssa
import pandas as pd
import functools
import numpy as np

data_dir = ss.root / "data"


def run_Measles(seed, beta, test_prob=0.03, response_time=7, daily_vacc=20, initial=10, forced_detect=np.nan,
                popsize=10000, **kwargs):
    '''
    Define population and networks
    Parameters:
        seed = random integer seed used to define initial conditions and RNG during simulation,
        beta = probability of transmission for each interaction with infectious agent,
        test_prob = daily testing probability for symptomatic agents,
        response_time = time in days between outbreak declaration (after one detected case) and ORI initiation,
        daily_vacc = number of vaccine doses delivered per day while ORI is active,
        initial = number of infected agents used to seed outbreak at first timestep,
        forced_detect = optional scenario parameter, defines day at which an infectious agent is detected to force outbreak declaration,
        popsize = number of agents in model population.
    '''
    ssu.set_seed(seed)
    pop_size = popsize
    mean_contacts = 3.5
    community_beta = beta / 1
    n_contacts_dist = ss.poisson(mean_contacts)
    #  mixing_H = pd.read_csv(ss.root / data_dir/ "mixing_H.csv", index_col="Age group")
    reference_ages = pd.read_csv(ss.root / data_dir/ "reference_ages_measles.csv") #, index_col="age").squeeze("columns")
    #  reference_ages=reference_ages, households=households) people = ss.People(pop_size, networks=[HouseholdNetwork(
    #  clusters=household_clusters), RandomNetwork(n_contacts=n_contacts_dist, layer_beta=0.012)])
    people = ss.People(pop_size, networks=[RandomNetwork(n_contacts=n_contacts_dist, layer_beta=1)])
    # Overwrite ages:
    cdf = np.cumsum(reference_ages['value'].values)
    cdf = cdf / cdf[-1]
    values = np.random.rand(pop_size)
    value_bins = np.searchsorted(cdf, values)
    ages_integers = reference_ages['age'].values[value_bins]
    ages = [age if age != 0 else np.random.uniform(low=0, high=1, size=1)[0] for age in ages_integers]
    people.age.values = np.array(ages)

    interventions = []

    # Add intervention schedule
    schedule = ssi.EventSchedule()
    interventions.append(schedule)

    if ~np.isnan(forced_detect):
        # Parameters to use if defining the day when an outbreak is detected
        forced_detect = int(forced_detect)
        use_test_prob = 0
        use_sensitivity = 0
    else:
        use_test_prob = test_prob
        use_sensitivity = 0.87

    # Add testing intervention
    symp_testing_intervention = ssi.test_prob_quarantine(
        disease='measles',
        symp_prob=use_test_prob,
        symp_quar_prob=0,  # Optimistically assume anyone in quarantine will test immediately
        sensitivity=use_sensitivity,
        test_delay_mean=1,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        quarantine_compliance=1,  # Everyone quarantines
        vac_symp_prob=use_test_prob,
        label="symp_testing",
    )

    interventions.append(symp_testing_intervention)

    # Add vaccination
    # Grab first and second dose peak coverage in kwargs
    if not kwargs.__contains__('first_dose_peak_coverage'):
        first_dose_peak_coverage = 100 # Assume 100% if
    else:
        first_dose_peak_coverage = kwargs['first_dose_peak_coverage']
    if not kwargs.__contains__('second_dose_peak_coverage'):
        second_dose_peak_coverage_rel2_first = 100
        second_dose_peak_coverage = 100
    else:
        second_dose_peak_coverage = kwargs['second_dose_peak_coverage']
        second_dose_peak_coverage_rel2_first = (second_dose_peak_coverage / first_dose_peak_coverage) * 100

    # Initialize with vaccinated people
    if not kwargs.__contains__('baseline_vax_coverage'):
        baseline_vax_coverage = 0
    else:
        baseline_vax_coverage = kwargs['baseline_vax_coverage']

    vac_peak_coverage = str(first_dose_peak_coverage)  # capacity constraints on coverages for population 6months+
    vaccine_eligible = ssu.peak_coverage_filter_measles(people, vac_peak_coverage) # All of these will receive the first dose.
    # Prioritize children <5. Doesn't affect simulation at the moment because of equal infection probability.
    sequence = ssi.get_vaccine_sequence_grouped(people, 'measles', "0.75+", vaccine_eligible)

    vac_used = MeaslesVaccine.measles_vacc()

    infants_immune_age = baseline_vax_coverage/100 * 0.97 + (1-baseline_vax_coverage/100)*3.78
    # Setting dynamic sequence to false for random vaccinating order
    interventions.append(
        ssi.TimedVaccinationProgram_Measles(vaccine=vac_used, sequence=sequence, num_doses=0, label="vacc_rollout",
                                            dynamic_sequence=False, infants_immune_age=infants_immune_age))

    if ~np.isnan(forced_detect):
        # methods to force case detection and outbreak declaration on a specified timestep
        assert isinstance(forced_detect, int)
        forced_detection = ssi.DynamicTrigger(condition=functools.partial(ssu.check_ti_trigger, t=forced_detect),
                                              action=functools.partial(ssu.ebola_forced_detection_action,
                                                                       disease='measles',
                                                                       sensitivity={"symptomatic": 0.87},
                                                                       symp_prob=test_prob),
                                              once_only=True, label='forced_detection')
        interventions.append(forced_detection)

    detection = ssi.DynamicTrigger(condition=functools.partial(ssu.outbreak_detection_trigger, disease='measles', size=1),
                                   action=functools.partial(ssu.ebola_detection_action, disease='measles',
                                                            response_time=response_time,
                                                            num_doses=daily_vacc, symp_prob=0.2),
                                   once_only=True, label='detection_trigger')
    interventions.append(detection)
    # add analyzers for post-simulation analysis
    analyzers = [ssa.TrackOutbreakDur, ssa.UpdateResults]
    analyzers.append(ssa.VaccinatedKnownContacts)

    # define key outbreak simulation parameters for measles model
    sim = ss.Sim(
        pars={'start': 2020, 'end': 2021, 'dt': 1 / 365, 'interventions': interventions, 'analyzers': analyzers,
              'remove_dead': False, 'rand_seed': seed},
        people=people,
        diseases=Measles({'initial': initial, 'baseline_vax_coverage': baseline_vax_coverage/100,
                          'sequence': sequence, 'second_dose_peak_coverage_rel2_first': second_dose_peak_coverage_rel2_first/100,
                          'beta': {'randomnetwork': community_beta},
                          'iso_factor': {'randomnetwork': 0.7},
                          'quar_period': {'randomnetwork': 0}})
    )
    sim.initialize()
    sim.results['measles-outbreak_detection'] = np.array([np.nan])
    sim.results['r_eff'] = np.array([np.nan])

    return sim
