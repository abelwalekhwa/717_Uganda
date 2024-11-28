import starsim as ss
from gavi.meningitis import Meningitis, MeningitisVaccine
from gavi.networks import RandomNetwork, HouseholdNetwork, generate_household_clusters
from gavi import utils as ssu
from gavi import interventions as ssi
from gavi import analyzers as ssa
import pandas as pd
import functools
import numpy as np

data_dir = ss.root / "data"


def run_Meningitis(seed, beta, test_prob=0.03, response_time=7, daily_vacc=20, initial_prev=0.02, vacc_used='polysaccharide', vacc_ages='1-29',
                popsize=10000, **kwargs):
    '''
    Define population and networks
    Parameters:
        seed = random integer seed used to define initial conditions and RNG during simulation,
        beta = probability of transmission for each interaction with infectious agent,
        test_prob = daily testing probability for symptomatic agents,
        response_time = time in days between outbreak declaration (after one detected case) and ORI initiation,
        daily_vacc = number of vaccine doses delivered per day while ORI is active,
        initial_prev = number of infected agents used to seed outbreak at first timestep,
        vacc_used = define the type of vaccine used by ORI, either polysaccharide or conjugate,
        vacc_ages = defines the age range targeted for vacciantion by ORI,
        popsize = number of agents in model population.
    '''
    ssu.set_seed(seed)
    pop_size = popsize

    mean_contacts = 11/4
    community_beta = beta

    n_contacts_dist = ss.poisson(mean_contacts)
    mixing_H = pd.read_csv(ss.root / data_dir / "mixing_H_meningitis.csv", index_col="Age group")  # average household contact matrix of all LMICs from Prem et al.
    reference_ages = pd.read_csv(ss.root / data_dir / "reference_ages_meningitis.csv", index_col="age").squeeze("columns")  # average age distribution of LMICs from UNWPP
    households = pd.read_csv(ss.root / data_dir / "households_meningitis.csv", index_col="size").squeeze("columns")  # average household distribution of LMICs from UN Household Compositions 2022 https://population.un.org/household/#/countries/840
    household_clusters, ages = generate_household_clusters(n_people=pop_size, mixing=mixing_H,
                                                           reference_ages=reference_ages, households=households)
    people = ss.People(pop_size, networks=[HouseholdNetwork(clusters=household_clusters),
                                           RandomNetwork(n_contacts=n_contacts_dist, layer_beta=0.2)]) # household transmission occurs at approximately 5 times the rate of non-household (https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(16)30244-3/fulltext)
    people.age.values = ages

    # Calculated to fit age based carriage prevalence meta analysis by Cooper et al. [figures in the SM] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6625194/)
    age_sus_carriage = {"0-1": 0.4, "1-2": 0.46/5, "2-3": 0.52/5, "3-4": 0.58/5, "4-5": 0.65/5, "5-6": 0.7/5, "6-7": 0.76/5,
                              "7-8": 0.84/5, "8-9": 0.9/5, "9-14": 1.0, "14-19": 0.9/5, "19-29": 0.69/5, "29+": 0.48/10}
    # Based on risk assessment by age (0-9) in Figure 2 by Rivero-Calle et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9026321/)
    # Older ages informed by incidence of IMD by age from CDC (https://www.cdc.gov/meningococcal/surveillance/surveillance-data.html)
    age_sus_imd = {"0-1": 1.0/5, "1-2": 0.6/5, "2-3": 0.38/5, "3-4": 0.28/5, "4-5": 0.2/5, "5-6": 0.18/5, "6-7": 0.16/5, "7-8": 0.13/5,
                   "8-9": 0.12/5, "9-14": 0.12/10, "14-19": 0.42/10, "19-29": 0.38/10, "29+": 0.32/30}
    #age_sus_imd = {age: 1.0 for age in age_sus_carriage.keys()}

    interventions = []

    # Add intervention schedule
    schedule = ssi.EventSchedule()
    interventions.append(schedule)

    # Add testing intervention
    symp_testing_intervention = ssi.test_prob_quarantine(
        disease='meningitis',
        symp_prob=0,
        symp_quar_prob=0,  # Optimistically assume anyone in quarantine will test immediately
        sensitivity=0,
        test_delay_mean=1,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        quarantine_compliance=1,  # Everyone quarantines
        vac_symp_prob=0,
        label="symp_testing",
    )

    interventions.append(symp_testing_intervention)

    # Add vaccination
    first_dose_peak_coverage = 100 # Assume 100%

    vac_peak_coverage = str(first_dose_peak_coverage)  # capacity constraints on coverages for population 6months+
    vaccine_eligible = ssu.peak_coverage_filter_meningitis(people, vac_peak_coverage) # All of these will receive the first dose.

    sequence = ssi.get_vaccine_sequence_grouped_priority_layer(people, 'meningitis', vacc_ages, vaccine_eligible)

    if vacc_used == 'conjugate':
        vac_used = MeningitisVaccine.meningitis_conjugate_vacc()
    elif vacc_used == 'polysaccharide':
        vac_used = MeningitisVaccine.meningitis_polysacc_vacc()
    else:
        print('Unkown vaccine specified, using polysaccharide as default')
        vac_used = MeningitisVaccine.meningitis_polysacc_vacc()

    # Setting dynamic sequence to false for random vaccinating order
    interventions.append(
        ssi.TimedVaccinationProgram(vaccine=vac_used, sequence=sequence, num_doses=0, label="vacc_rollout", dynamic_sequence=False))

    detection = ssi.DynamicTrigger(condition=functools.partial(ssu.outbreak_detection_trigger, disease='meningitis', size=5),
                                   action=functools.partial(ssu.meningitis_detection_action, disease='meningitis',
                                                            response_time=response_time,
                                                            num_doses=daily_vacc),
                                   once_only=True, label='detection_trigger')
    interventions.append(detection)
    # add analyzers for post-simulation analysis
    analyzers = [ssa.TrackOutbreakDur, ssa.UpdateResults]
    analyzers.append(ssa.AgeOfDeath)
    analyzers.append(ssa.DurationOfSymptoms)
    analyzers.append(ssa.InfectionsByAge)

    # define key outbreak simulation parameters for meningitis model
    sim = ss.Sim(
        pars={'start': 2020, 'end': 2020, 'dt': 1 / 365, 'interventions': interventions, 'analyzers': analyzers,
              'remove_dead': False, 'rand_seed': seed},
        people=people,
        diseases=Meningitis({'init_carr_prev': initial_prev, 'age_sus_carriage': age_sus_carriage, 'age_sus_imd': age_sus_imd,
                          'beta': {'householdnetwork': beta, 'randomnetwork': community_beta},
                          'iso_factor': {'householdnetwork': 0.8, 'randomnetwork': 0.5},
                        'quar_period': {'householdnetwork': 0, 'randomnetwork': 0}})
    )
    sim.initialize()
    sim.results['meningitis-outbreak_detection'] = np.array([np.nan])
    sim.results['r_eff'] = np.array([np.nan])

    return sim
