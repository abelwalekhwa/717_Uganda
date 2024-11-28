import starsim as ss
from gavi.cholera import Cholera, CholeraVaccine
from gavi.networks import RandomNetwork, HouseholdNetwork, generate_household_clusters
from gavi import utils as ssu
from gavi import interventions as ssi
from gavi import analyzers as ssa
import pandas as pd
import numpy as np
import functools

data_dir = ss.root / "data"

def run_Cholera(seed, beta, env_beta, test_prob=0.03, response_time=7, daily_vacc=20, initial=10, forced_detect=np.nan, popsize=50000):
    '''
    Define population and networks
    Parameters:
        seed = random integer seed used to define initial conditions and RNG during simulation,
        beta = probability of transmission for each household interaction with infectious agent,
        env_beta = probability of infection for each susceptible agent due to environmental cholera (proportional to concentration, following dose-response relationship),
        test_prob = daily testing probability for symptomatic agents,
        response_time = time in days between outbreak declaration (after one detected case) and ORI initiation,
        daily_vacc = number of vaccine doses delivered per day while ORI is active,
        initial = number of infected agents used to seed outbreak at first timestep,
        forced_detect = optional scenario parameter, defines day at which an infectious agent is detected to force outbreak declaration,
        popsize = number of agents in model population.
    '''
    ssu.set_seed(seed)
    pop_size = popsize
    mean_contacts = 9, # average of all non-household contacts in LMICs from Prem et al.
    community_beta = beta / 2.9 # Richterman et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6188541/) found that household contacts had on average 2.9 times higher odds of transmitting cholera than non-household contacts. This aligns with findings from Sugimoto et al. (https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0003314)
    n_contacts_dist = ss.poisson(mean_contacts)
    mixing_H = pd.read_csv(ss.root / data_dir/ "mixing_H_cholera.csv", index_col="Age group") # average household contact matrix of all LMICs from Prem et al.
    reference_ages = pd.read_csv(ss.root / data_dir/ "reference_ages_cholera.csv", index_col="age").squeeze("columns") # average age distribution of LMICs from UNWPP
    households = pd.read_csv(ss.root / data_dir/ "households_cholera.csv", index_col="size").squeeze("columns") # average household distribution of LMICs from UN Household Compositions 2022 https://population.un.org/household/#/countries/840
    household_clusters, ages = generate_household_clusters(n_people=pop_size, mixing=mixing_H, reference_ages=reference_ages, households=households)
    people = ss.People(pop_size, networks=[HouseholdNetwork(clusters=household_clusters), RandomNetwork(n_contacts=n_contacts_dist, layer_beta=0.018)])
    people.age.values = ages

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
        disease='cholera',
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
    vaccine_eligible = ssu.peak_coverage_filter(people, vac_peak_coverage)
    sequence = ssi.get_vaccine_sequence_grouped(people, 'cholera', "1+", vaccine_eligible)

    vac_used = CholeraVaccine.cholera_vacc()

    interventions.append(ssi.TimedVaccinationProgram(vaccine=vac_used, sequence=sequence, num_doses=0, label="vacc_rollout",
                                    dynamic_sequence=True))

    if ~np.isnan(forced_detect):
        # methods to force case detection and outbreak declaration on a specified timestep
        forced_detection = ssi.DynamicTrigger(condition=functools.partial(ssu.check_ti_trigger, t=forced_detect),
                                       action=functools.partial(ssu.ebola_forced_detection_action, disease='cholera',
                                                                symp_prob=test_prob, sensitivity={"symptomatic": 0.87}), once_only=True, label='forced_detection')
        interventions.append(forced_detection)

    detection = ssi.DynamicTrigger(condition=functools.partial(ssu.outbreak_detection_trigger, disease='cholera'),
                                  action=functools.partial(ssu.cholera_detection_action, disease='cholera', response_time=response_time,
                                                           num_doses=daily_vacc, symp_prob=0.08), once_only=True, label='detection_trigger')
    interventions.append(detection)

    #add analyzers for post-simulation analysis
    analyzers = []
    analyzers.append(ssa.TrackOutbreakDur)
    analyzers.append(ssa.UpdateResults)
    analyzers.append(ssa.AgeOfDeath)
    analyzers.append(ssa.DurationOfSymptoms)

    # define key outbreak simulation parameters for cholera model
    sim = ss.Sim(
        pars={'start': 2020, 'end': 2021, 'dt': 1/365, 'interventions': interventions, 'analyzers': analyzers, 'remove_dead': False, 'rand_seed': seed},
        people=people,
        diseases=Cholera({'initial':initial,'beta_direct':{'householdnetwork': beta, 'randomnetwork': community_beta},'beta_environment_mult': env_beta,
                        'iso_factor': {'householdnetwork': 0.8, 'randomnetwork': 0.1},
                        'quar_factor': {'householdnetwork': 0.8, 'randomnetwork': 0.8},
                        'quar_period': {'householdnetwork': 21, 'randomnetwork': 21}})
    )
    sim.initialize()
    sim.results['cholera-outbreak_detection'] = np.array([np.nan])

    return sim
