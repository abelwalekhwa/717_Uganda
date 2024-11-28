# FUNCTIONS TO RUN MODELS ON CLUSTER
import pandas as pd
from celery import Celery
import starsim as ss
from gavi import utils
import sciris as sc
from gavi import multisim as ssm
from celery.signals import after_setup_task_logger
import logging
from gavi.ebola_main import run_Ebola
import numpy as np
from gavi.measles_main import run_Measles
from gavi.cholera_main import run_Cholera
from gavi.meningitis_main import run_Meningitis
from gavi.yellow_fever_main import run_Yellow_Fever
utils.git_info = lambda: None  # Disable this function to increase performance slightly


import os

broker = os.getenv("GAVI_OB_REDIS_URL", "redis://127.0.0.1:6379")

# Create celery app
celery = Celery("gavi-outbreaks")
celery.conf.broker_url = broker
celery.conf.result_backend = broker
celery.conf.task_default_queue = "gavi-outbreaks"
celery.conf.accept_content = ["pickle", "json"]
celery.conf.task_serializer = "pickle"
celery.conf.result_serializer = "pickle"
celery.conf.worker_prefetch_multiplier = 1
celery.conf.task_acks_late = True  # Allow other servers to pick up tasks in case they are faster
celery.conf.worker_max_tasks_per_child = 5
celery.conf.worker_max_memory_per_child = 3000000


# Quieter tasks
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.setLevel(logging.WARNING)

@celery.task()
def run_calibration(seed, beta, response_time=7, daily_vacc=20, env_beta=0.5, initial_prev=0.02, vacc_used='conjugate', vacc_ages='1-29', test_prob=0.03, forced_detect=np.nan, initial=10, disease='ebola', return_sim=False, stopping_func=None, **kwargs):
    """
    Run the calibration sim (use run_sim for non-calibration runs)

    Args:
        beta:
        seed:
        stopping_func: Pass in stopping function. This will be passed to the Sim and used to check the calibration

    Returns:

    """
    if disease.lower() == 'ebola':
        sim = run_Ebola(seed, beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'measles':
        sim = run_Measles(seed, beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'cholera':
        sim = run_Cholera(seed, beta, env_beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'yellow_fever':
        sim = run_Yellow_Fever(seed, beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'meningitis':
        sim = run_Meningitis(seed, beta, test_prob, response_time, daily_vacc, initial_prev, vacc_used, vacc_ages, **kwargs)
    else:
        print(disease + ' is not a supported disease yet.')
        return
    sim.pars["stopping_func"] = stopping_func  # Insert the stopping function

    sim.run()

    accepted_calibration = True

    if not sim.results_ready:
        # If the calibration was terminated via the stopping function, then we need
        # to perform the finalization and cleanup manually so that we can write the output
        accepted_calibration = False
        sim.finalize()

        sim.tivec = sim.tivec[0 : sim.ti + 1]
        for key, result in sim.results.items():
            if isinstance(result, ssm.MultiSimResult):
                result.values = result.values[0 : sim.ti + 1]
            elif isinstance(result, ss.Result):
                result = result[0: sim.ti + 1]
            elif key in sim.diseases:
                for k, res in result.items():
                    if isinstance(res, ssm.MultiSimResult):
                        res.values = res.values[0: sim.ti + 1]
                    elif isinstance(res, ss.Result):
                        res = result[0: sim.ti + 1]
            else:
                res = np.zeros(len(sim.tivec))
                res[-1] = result[0]
                result = res
                        
    df_res = ssm.export_results(sim)
    df = pd.DataFrame.from_dict(df_res)
    if disease == 'measles' or disease == 'yellow_fever':
        df["r_eff"] = sim.results.r_eff

    summary = {}
    summary["beta"] = beta
    summary["seed"] = seed
    summary["response_time"] = response_time
    summary["daily_vacc"] = daily_vacc

    if disease=='measles':
        summary["baseline_vax_coverage"] = kwargs["baseline_vax_coverage"]
    elif disease == 'cholera':
        summary["env_beta"] = env_beta
        summary["average_death_age"] = sim.results["average_death_age"][-1]
        summary["average_symp_dur"] = sim.results["average_symp_dur"][-1]

    elif disease == 'yellow_fever':
        summary["beta_mosquito2human"] = beta * kwargs["beta_mosquito2human_modifier"]
        summary["cum_severe"] = sim.results[disease + "-cum_severe"][-1]
    elif disease == 'meningitis':
        summary["initial_prev"] = initial_prev
        summary["average_death_age"] = sim.results["average_death_age"][250]
        summary["average_symp_dur"] = sim.results["average_symp_dur"][250]
        summary["final_prev"] = sim.results[disease + "-carriage_prevalence"][250]
        age_bins = ["0-4", "4-9", "10-19", "20-29", "30+"]
        infection_cat = ['all', 'symp', 'asymp']
        for cat in infection_cat:
            for age_range in age_bins:
                summary[cat + '_inf_' + age_range] = sim.results[cat + '_inf_' + age_range][250]
    elif disease.lower() == 'ebola':
        summary["average_death_age"] = sim.results["average_death_age"]
        summary["average_symp_dur"] = sim.results["average_symp_dur"]


    summary["outbreak_dur"] = sim.results[disease + "-outbreak_dur"][-1]
    summary["detect_time"] = sim.results[disease + "-outbreak_detection"][-1]

    final_day_quantities = ["cum_diagnoses",
            "cum_infections",
            "cum_deaths"]

    for quantity in final_day_quantities:
        if disease == 'meningitis':
            if "dalys" in quantity or "costs" in quantity:
                summary[quantity] = df.iloc[250][quantity]
            else:
                summary[quantity] = df.iloc[250][disease.lower() + "-" + quantity]
        else:
            if "dalys" in quantity or "costs" in quantity:
                summary[quantity] = df.iloc[-1][quantity]
            else:
                summary[quantity] = df.iloc[-1][disease.lower() + "-" + quantity]
    summary["accepted_calibration"] = accepted_calibration
    
    if return_sim:
        return df, summary, sim
    else:
        return df, summary


def stop_calibration_Ebola(sim, response_time, daily_vacc):
    time_ranges = [4, 7, 10, 14]
    vacc_ranges = [5, 10, 20, 35]
    case_data = [[5, np.nan, 130, 11],
                 [np.nan, np.nan, np.nan, np.nan],
                 [1, 12, 23, np.nan],
                 [54, np.nan, np.nan, np.nan]]
    death_data = [[5, np.nan, 55, 9],
                  [np.nan, np.nan, np.nan, np.nan],
                  [1, 6, 12, np.nan],
                  [33, np.nan, np.nan, np.nan]]
    duration_data = [[90, np.nan, 184, 68],
                  [np.nan, np.nan, np.nan, np.nan],
                  [66, 98, 155, np.nan],
                  [111, np.nan, np.nan, np.nan]]
    # Choose data to match against based on lattice point
    rt_ind = time_ranges.index(response_time)
    dv_ind = vacc_ranges.index(daily_vacc)
    cases = case_data[rt_ind][dv_ind]
    deaths = death_data[rt_ind][dv_ind]
    duration = duration_data[rt_ind][dv_ind]

    # Skip any check if there is no data for lattice point
    if np.isnan(cases):
        return False

    if ~np.isnan(sim.results['ebola-outbreak_detection']):
        skip_until = sim.results['ebola-outbreak_detection'] + duration - 14
    else:
        skip_until = 120

    if sim.ti < skip_until:
        return False  # Run for at least 30 days to allow time for the epidemic to start

    # CHECK TOTAL CASES
    model_cases = sim.diseases['ebola'].results['cum_diagnoses'].values[sim.ti-1]
    model_deaths = sim.diseases['ebola'].results['cum_deaths'][sim.ti-1]

    if cases * 0.25 < 5:
        tolerance = 5 / cases
    else:
        tolerance = 0.25
    if model_cases > (cases * (1 + tolerance)) or model_cases < (cases * (1 - tolerance)):
        return True

    if deaths * 0.25 < 5:
        tolerance = 5 / deaths
    else:
        tolerance = 0.25    

    if model_deaths > (deaths * (1 + tolerance)) or model_deaths < (deaths * (1 - tolerance)):
        return True

    return False


@celery.task()
def run_sim(seed, beta, response_time=7, daily_vacc=20, env_beta=0.5, initial_prev=0.02, vacc_used='conjugate', vacc_ages='1-29',
            test_prob=0.03, forced_detect=np.nan, initial=10, disease='ebola', return_sim=False, **kwargs):
    """
    Run sim for scenarios

    This function saves additional summary outputs not used by the calibration
    Additional arguments are also passed to the `run_Ebola` function`

    beta: Global beta value
    seed: Random seed
    return_sim: If True, return the Sim instance
    **kwargs: Additional arguments passed to `run_Ebola` usually used to specify a scenario

    Returns:
        - Dataframe with simulation output

    """

    if disease.lower() == 'ebola':
        sim = run_Ebola(seed, beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'measles':
        sim = run_Measles(seed, beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'cholera':
        sim = run_Cholera(seed, beta, env_beta, test_prob, response_time, daily_vacc, initial, forced_detect, **kwargs)
    elif disease == 'meningitis':
        sim = run_Meningitis(seed, beta, test_prob, response_time, daily_vacc, initial_prev, vacc_used, vacc_ages, **kwargs)
    else:
        print(disease + ' is not a supported disease yet.')
        return
    sim.run()

    # Retrieve dataframe (per-timestep output to save in as the CSV for this run)
    df_res = ssm.export_results(sim)
    df = pd.DataFrame.from_dict(df_res)
    df.index = sim.tivec[0: len(df)]
    df.index.name = "day"
    df["seed"] = seed
    if disease == 'measles':
        df["r_eff"] = sim.results.r_eff

    # Retrieve summary states (scalar outputs to store in summary.csv)
    summary = sc.dcp(kwargs)
    summary["beta"] = beta
    if disease == 'cholera':
        summary["env_beta"] = env_beta
    elif disease == 'meningitis':
        summary["initial_prev"] = initial_prev
        summary["average_death_age"] = sim.results["average_death_age"][250]
        summary["average_symp_dur"] = sim.results["average_symp_dur"][250]
        summary["final_prev"] = sim.results[disease + "-carriage_prevalence"][250]
        age_bins = ["0-4", "4-9", "10-19", "20-29", "30+"]
        infection_cat = ['all', 'symp', 'asymp']
        for cat in infection_cat:
            for age_range in age_bins:
                summary[cat + '_inf_' + age_range] = sim.results[cat + '_inf_' + age_range][250]

    elif disease.lower() == 'ebola':
        summary["average_death_age"] = sim.results["average_death_age"]
        summary["average_symp_dur"] = sim.results["average_symp_dur"]

    summary["seed"] = seed
    summary["response_time"] = response_time
    summary["daily_vacc"] = daily_vacc
    summary["test_prob"] = test_prob
    summary["outbreak_dur"] = sim.results[disease+"-outbreak_dur"][0]
    summary["detect_time"] = sim.results[disease+"-outbreak_detection"][0]

    # summary["n_agents_vaccinated"] = sim.people.vaccinated.sum()
    if disease.lower() == 'ebola':
        final_day_quantities = [
            "cum_diagnoses",
            "cum_infections",
            "cum_deaths",
            "cum_dalys",
            "cum_costs",
        ]
    if disease == 'measles':
        final_day_quantities = [
            "cum_diagnoses",
            "cum_infections",
            "cum_deaths"
        ]
    if disease == 'cholera':
        final_day_quantities = [
            "cum_diagnoses",
            "cum_infections",
            "cum_deaths",
            "average_death_age",
            'average_symp_dur'
        ]
    if disease == 'meningitis':
        final_day_quantities = [
            "cum_diagnoses",
            "cum_infections",
            "cum_deaths",
        ]


    for quantity in final_day_quantities:
        if disease == 'meningitis':
            if "dalys" in quantity or "costs" in quantity or "average" in quantity:
                summary[quantity] = df.iloc[250][quantity]
            else:
                summary[quantity] = df.iloc[250][disease + "-" + quantity]
        else:
            if "dalys" in quantity or "costs" in quantity or "average" in quantity:
                summary[quantity] = df.iloc[-1][quantity]
            else:
                summary[quantity] = df.iloc[-1][disease + "-" + quantity]

    summary["worker_hostname"] = celery.current_task.request.hostname


    if return_sim:
        return df, summary, sim
    else:
        return df, summary
