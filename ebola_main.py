import starsim as ss
from starsim.gavi.ebola import Ebola, EbolaVaccine
from starsim.gavi.networks import RandomNetwork, HouseholdNetwork, generate_household_clusters
from starsim.gavi import utils as ssu
from starsim.gavi import interventions as ssi
from starsim.gavi import analyzers as ssa
import pandas as pd
import numpy as np
import functools

data_dir = ss.root / "data"

def run_Ebola(seed, beta, test_prob=0.03, response_time=7, daily_vacc=20, initial=10, forced_detect=np.nan, popsize=10000):
    '''
    Define population and networks
    Parameters:
        seed = random integer seed used to define initial conditions and RNG during simulation,
        beta = probability of transmission for each household interaction with infectious agent,
        test_prob = daily testing probability for symptomatic agents,
        response_time = time in days between outbreak declaration (after one detected case) and ORI initiation,
        daily_vacc = number of vaccine doses delivered per day while ORI is active,
        initial = number of infected agents used to seed outbreak at first timestep,
        forced_detect = optional scenario parameter, defines day at which an infectious agent is detected to force outbreak declaration,
        popsize = number of agents in model population.
    '''
    ssu.set_seed(seed)
    pop_size = popsize
    mean_contacts = 14,
    community_beta = beta / 4.6 # based on adult family members being at increased risk https://academic.oup.com/jid/article/179/Supplement_1/S87/882673
    n_contacts_dist = ss.poisson(mean_contacts)
    mixing_H = pd.read_csv(ss.root / data_dir/ "mixing_H_Ebola.csv", index_col="Age group")
    reference_ages = pd.read_csv(ss.root / data_dir/ "reference_ages_Ebola.csv", index_col="age").squeeze("columns")
    households = pd.read_csv(ss.root / data_dir/ "households_Ebola.csv", index_col="size").squeeze("columns")
    household_clusters, ages = generate_household_clusters(n_people=pop_size, mixing=mixing_H, reference_ages=reference_ages, households=households)
    people = ss.People(pop_size, networks=[HouseholdNetwork(clusters=household_clusters), RandomNetwork(n_contacts=n_contacts_dist, layer_beta=0.018)])
    people.age.values = ages

    interventions = []
    # Add intervention schedule
    schedule = ssi.EventSchedule()
    interventions.append(schedule)

    # Add testing intervention

    trace_testing_intervention = ssi.test_prob_quarantine(
        disease='ebola',
        symp_prob=0,
        symp_quar_prob=0,  # Optimistically assume anyone in quarantine will test immediately
        sensitivity=0.99,
        test_delay_mean=2,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        quarantine_compliance=0.9,  # Everyone quarantines
        vac_symp_prob=0,
        exclude=people.uid,  # Nobody tests by default
        label="trace_testing",
    )
    if ~np.isnan(forced_detect):
        # Parameters to use if defining the day when an outbreak is detected
        forced_detect = int(forced_detect)
        use_test_prob = 0
        use_sensitivity = 0
    else:
        use_test_prob = test_prob
        use_sensitivity = 0.87
    symp_testing_intervention = ssi.test_prob_quarantine(
        disease='ebola',
        symp_prob=use_test_prob,
        symp_quar_prob=0,  # Optimistically assume anyone in quarantine will test immediately
        sensitivity=use_sensitivity,
        test_delay_mean=1,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        quarantine_compliance=0.8,  # Everyone quarantines
        vac_symp_prob=use_test_prob,
        label="symp_testing",
    )

    interventions.append(trace_testing_intervention)
    interventions.append(symp_testing_intervention)

    #Add some contact tracing
    contact_tracing = ssi.contact_tracing(disease='ebola', trace_probs={'householdnetwork': 0.95, 'randomnetwork': 0.25},
                                         trace_time={'householdnetwork': 1, 'randomnetwork': 2}, start_day=0, quar_period=21,
                                         test_schedule=(6, 20), capacity=25, testing_intervention=trace_testing_intervention,
                                         label='ebola_tracing')
    interventions.append(contact_tracing)

    # Add vaccination
    vac_peak_coverage = "85"  # capacity constraints on coverages for population 18+
    vaccine_eligible = ssu.peak_coverage_filter(people, vac_peak_coverage)
    sequence = ssi.get_vaccine_sequence_grouped(people, 'ebola', "18+", vaccine_eligible)

    vac_used = EbolaVaccine.ervebo()

    interventions.append(ssi.TimedVaccinationProgram(vaccine=vac_used, sequence=sequence, num_doses=0, label="vacc_rollout",
                                    dynamic_sequence=True))

    if ~np.isnan(forced_detect):
        # methods to force case detection and outbreak declaration on a specified timestep
        forced_detection = ssi.DynamicTrigger(condition=functools.partial(ssu.check_ti_trigger, t=forced_detect),
                                       action=functools.partial(ssu.ebola_forced_detection_action, disease='ebola',
                                                                symp_prob=test_prob, sensitivity={"symptomatic": 0.87}), once_only=True, label='forced_detection')
        interventions.append(forced_detection)

    detection = ssi.DynamicTrigger(condition=functools.partial(ssu.outbreak_detection_trigger, disease='ebola'),
                                  action=functools.partial(ssu.ebola_detection_action, disease='ebola', response_time=response_time,
                                                           num_doses=daily_vacc, symp_prob=0.2), once_only=True, label='detection_trigger')
    interventions.append(detection)

    # add analyzers for post-simulation analysis
    analyzers = []
    analyzers.append(ssa.TrackOutbreakDur)
    analyzers.append(ssa.UpdateResults)
    analyzers.append(ssa.VaccinatedKnownContacts)
    analyzers.append(ssa.AgeOfDeath)
    analyzers.append(ssa.DurationOfSymptoms)
    analyzers.append(ssa.CurrentlyHospitalised)
    analyzers.append(ssa.ProportionChildrenInfected)
    analyzers.append(ssa.SafelyBuried)
    analyzers.append(ssa.CalcDALYsCosts)

    # define key outbreak simulation parameters for Ebola model
    sim = ss.Sim(
        pars={'start': 2020, 'end': 2021, 'dt': 1/365, 'interventions': interventions, 'analyzers': analyzers, 'remove_dead': False, 'rand_seed': seed},
        people=people,
        diseases=Ebola({'initial':initial,'beta':{'householdnetwork': beta, 'randomnetwork': community_beta},
                        'iso_factor': {'householdnetwork': 0.6, 'randomnetwork': 0.9},
                        'quar_factor': {'householdnetwork': 0.8, 'randomnetwork': 0.9},
                        'sev_factor': {'householdnetwork': 2.2, 'randomnetwork': 2.2},
                        'unburied_factor': {'householdnetwork': 2.1, 'randomnetwork': 1.6},
                        'quar_period': {'householdnetwork': 21, 'randomnetwork': 21}})
    )
    sim.initialize()
    sim.results['ebola-outbreak_detection'] = np.array([np.nan])

    return sim
