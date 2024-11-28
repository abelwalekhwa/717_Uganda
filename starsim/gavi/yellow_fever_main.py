import starsim as ss
from gavi.yellow_fever import Yellow_Fever, Yellow_FeverVaccine
from gavi import utils as ssu
from gavi import interventions as ssi
from gavi import analyzers as ssa
import pandas as pd
import numpy as np
import functools

data_dir = ss.root / "data"

def run_Yellow_Fever(seed, beta, test_prob=0.03, response_time=7, daily_vacc=20, initial=10, forced_detect=np.nan, popsize=10000, **kwargs):
    '''
    Define population and networks
    Parameters:
        seed = random integer seed used to define initial conditions and RNG during simulation,
        beta = probability of transmission for each interaction between mosquito and infectious agent or vice-versa,
        test_prob = daily testing probability for symptomatic agents,
        response_time = time in days between outbreak declaration (after one detected case) and ORI initiation,
        daily_vacc = number of vaccine doses delivered per day while ORI is active,
        initial = number of infected agents used to seed outbreak at first timestep,
        forced_detect = optional scenario parameter, defines day at which an infectious agent is detected to force outbreak declaration,
        popsize = number of agents in model population.
    '''
    ssu.set_seed(seed)
    pop_size = popsize
    people = ss.People(pop_size)
    #people = ss.People(pop_size, networks=[RandomNetwork(n_contacts=n_contacts_dist, layer_beta=1)])
    reference_ages = pd.read_csv(ss.root / data_dir/ "reference_ages_YF.csv")
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

    # Add testing intervention

    if ~np.isnan(forced_detect):
        # Parameters to use if defining the day when an outbreak is detected
        forced_detect = int(forced_detect)
        use_test_prob = 0
        use_sensitivity = 0
    else:
        use_test_prob = test_prob
        use_sensitivity = 0.87

    symp_testing_intervention = ssi.test_prob_quarantine(
        disease='yellow_fever',
        symp_prob=use_test_prob,
        symp_quar_prob=0,  # Optimistically assume anyone in quarantine will test immediately
        sensitivity=use_sensitivity,
        test_delay_mean=1,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        quarantine_compliance=0.8,  # Everyone quarantines
        vac_symp_prob=use_test_prob,
        label="symp_testing",
    )

    interventions.append(symp_testing_intervention)

    # Add vaccination
    vac_peak_coverage = "100"  # capacity constraints on coverages for population 1+
    vaccine_eligible = ssu.peak_coverage_filter_yellowfever(people, vac_peak_coverage)
    sequence = ssi.get_vaccine_sequence_grouped(people, 'yellow_fever', "0.75+", vaccine_eligible)

    vac_used = Yellow_FeverVaccine.yellow_fever_vacc()

    interventions.append(ssi.TimedVaccinationProgram(vaccine=vac_used, sequence=sequence, num_doses=0, label="vacc_rollout",
                                    dynamic_sequence=True))

    if ~np.isnan(forced_detect):
        # methods to force case detection and outbreak declaration on a specified timestep
        forced_detection = ssi.DynamicTrigger(condition=functools.partial(ssu.check_ti_trigger, t=forced_detect),
                                       action=functools.partial(ssu.ebola_forced_detection_action, disease='yellow_fever',
                                                                symp_prob=test_prob, sensitivity={"symptomatic": 0.87}), once_only=True, label='forced_detection')
        interventions.append(forced_detection)

    detection = ssi.DynamicTrigger(condition=functools.partial(ssu.outbreak_detection_trigger, disease='yellow_fever'),
                                  action=functools.partial(ssu.ebola_detection_action, disease='yellow_fever', response_time=response_time,
                                                           num_doses=daily_vacc, symp_prob=0.2), once_only=True, label='detection_trigger')
    interventions.append(detection)

    # Initialize with vaccinated people
    if not kwargs.__contains__('baseline_vax_coverage'):
        baseline_vax_coverage = 0
    else:
        baseline_vax_coverage = kwargs['baseline_vax_coverage']

    if not kwargs.__contains__('beta_mosquito2human_modifier'):
        print('The Yellow Fever Model requires a mosquito2human modifier!')
    else:
        beta_human2mosquito = beta
        beta_mosquito2human = beta_human2mosquito * kwargs['beta_mosquito2human_modifier']

    # add analyzers for post-simulation analysis
    analyzers = []
    analyzers.append(ssa.TrackOutbreakDur)
    analyzers.append(ssa.UpdateResults)
    analyzers.append(ssa.AgeOfDeath)
    analyzers.append(ssa.DurationOfSymptoms)

    # define key outbreak simulation parameters for yellow fever model
    sim = ss.Sim(
        pars={'start': 2020, 'end': 2021, 'dt': 1/365, 'interventions': interventions, 'analyzers': analyzers, 'remove_dead': False, 'rand_seed': seed},
        people=people,
        diseases=Yellow_Fever({'initial': initial,
                               'beta_human2mosquito': beta_human2mosquito,
                               'beta_mosquito2human': beta_mosquito2human,
                               'baseline_vax_coverage': baseline_vax_coverage/100,
                               'sequence': sequence})
    )
    sim.initialize()
    sim.results['yellow_fever-outbreak_detection'] = np.array([np.nan])
    sim.results['r_eff'] = np.array([np.nan])

    return sim
