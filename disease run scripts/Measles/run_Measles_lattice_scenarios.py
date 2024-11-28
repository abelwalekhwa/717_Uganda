import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
import numpy as np
from celery import group
from tqdm import tqdm
from gavi import MultiSim
from gavi.celery import run_calibration, celery
from starsim import Samples
import pandas as pd

debug_mode = False  # If True, run just one set of parameters and do not use threading
result_dir = Path("results/Scenarios")
# Mean and standard deviation for sampling transmission beta
mean_beta = 0.17
std_beta = 0.025

# parameter values for constructing array of outbreak simulations
baselines_to_consider = 10
response_times = [30, 60, 90, 120, 170, 220]
baseline_vax_coverages = np.arange(10, 110, 10)
daily_vaccs = [500, 2500, 5000, 7000, 8500]
to_run = []

scenarios = pd.read_csv(
    r"scenarios_seeds_to_run.csv")

def get_betas(mean_beta, std_beta, nruns=None):
    """
    Sample beta values and create seed array for baseline simulations for each outbreak
    """
    seeds = np.arange(nruns)
    betas = np.random.normal(mean_beta, std_beta, nruns)
    return seeds, betas

def run_scenario(kwargs, filter=False):
    """
        Pull out the parameters from filtered baseline simulations for each outbreak
    """
    nruns = kwargs['nruns']
    name = kwargs['name']
    outbreak_id = kwargs['outbreak_id']
    del kwargs['nruns']

    response_time = kwargs['response_time']
    daily_vacc = kwargs['daily_vacc']
    baseline_vax_coverage = kwargs['baseline_vax_coverage']
    del kwargs['response_time']
    del kwargs['daily_vacc']

    seeds = scenarios[scenarios['outbreak'] == outbreak_id]['seeds'].tolist()
    betas = scenarios[scenarios['outbreak'] == outbreak_id]['betas'].tolist()

    description_baseline = str(name) + "_outbreakID_" + str(outbreak_id)

    if not hasattr(thread_local, "pbar"):
        thread_local.pbar = tqdm(total=len(seeds))
    pbar = thread_local.pbar
    pbar.set_description(description_baseline)
    pbar.n = 0
    pbar.refresh()
    pbar.unpause()

    fname = description_baseline

    if (result_dir / fname).exists():
        return

    tasks = []
    for ind, seed in enumerate(seeds):
        if name == 'NoORI':
            tasks.append(run_calibration.s(seed, betas[ind], response_time=0,
                                           daily_vacc=0, disease='measles', **kwargs))
        if name == '7day':
            tasks.append(run_calibration.s(seed, betas[ind], response_time=7,
                                           daily_vacc=daily_vacc, disease='measles', **kwargs))

    job = group(*tasks)
    result = job.apply_async()
    ready = False

    while not ready:
        time.sleep(1)
        n_ready = sum(int(result.ready()) for result in result.results)
        ready = n_ready == len(seeds)
        pbar.n = n_ready
        if pbar.n == 0:
            pbar.reset(total=len(seeds))
        else:
            pbar.refresh()

    if result.successful():
        outputs = result.join()
        if filter:
            Samples.new(result_dir, outputs, ["seed"], fname="baseline_filtered.zip")
        else:
            Samples.new(result_dir, outputs, ["seed"], fname=fname + '.zip')
    else:
        pbar.set_description("baseline ERROR")
        for x in result.results:
            if x.failed():
                with open(result_dir / f"error_{x.id}.txt", "w") as log:
                    log.write(str(x.__dict__))

    result.forget()

    return True

if __name__ == "__main__":

    # Set up Scenarios:
    # Read scenarios
    outbreaks = np.unique(scenarios['outbreak'])
    for outbreak in outbreaks:
        this_outbreak_seeds = scenarios[scenarios['outbreak'] == outbreak]
        this_outbreak_daily_vaccine = this_outbreak_seeds['daily_vacc'].tolist()[0]
        this_outbreak_baseline_coverage = this_outbreak_seeds['baseline_vax_coverage'].tolist()[0]
        this_outbreak_response_time = this_outbreak_seeds['response_time'].tolist()[0]

        to_run.append({"name": "NoORI", "initial": 3,  # number of seed infections
                                "test_prob": 0.10,  # testing prob per day
                                "popsize": 50000,
                                "daily_vacc": this_outbreak_daily_vaccine * (1-this_outbreak_baseline_coverage/100),
                                "response_time": this_outbreak_response_time,
                                "baseline_vax_coverage": this_outbreak_baseline_coverage,
                                "filter": False})

        to_run.append({"name": "7day", "initial": 3,  # number of seed infections
                                "test_prob": 0.10,  # testing prob per day
                                "popsize": 50000,
                                "daily_vacc": this_outbreak_daily_vaccine * (1-this_outbreak_baseline_coverage/100),
                                "response_time": this_outbreak_response_time,
                                "baseline_vax_coverage": this_outbreak_baseline_coverage,
                                "filter": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--nruns", default=8, type=int, help="Number of seeds to run per scenario")
    parser.add_argument("--celery", default=False, type=bool, help="If True, use Celery for parallelization")

    args = parser.parse_args()
    thread_local = threading.local()

    if debug_mode:
        # Use debug mode to run the full sampling over seeds, but without Celery
        to_run[0]['nruns'] = args.nruns
        run_scenario(to_run[0])

    elif args.celery:
        futures = []
        result_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(to_run), desc=f"Total progress") as pbar:
            pbar.n = 0
            pbar.refresh()
            pbar.unpause()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

                for i, run_args in enumerate(to_run):
                    run_args['nruns'] = args.nruns
                    response_time = run_args['response_time']
                    daily_vacc = run_args['daily_vacc']
                    futures.append(executor.submit(run_scenario, run_args))
                    if i == 0:
                        time.sleep(5)

                while True:

                    done = [x for x in futures if x.done()]

                    for result in done:
                        if result.exception():
                            [x.cancel() for x in futures]
                            celery.control.purge()
                            celery.control.shutdown()
                            raise result.exception()

                    pbar.n = len(done)
                    pbar.refresh()
                    if len(done) == len(futures):
                        break
                    time.sleep(1)

        # Shut down the workers
        celery.control.shutdown()

    else:
        import matplotlib.pyplot as plt
        import sciris as sc

        #kwargs = to_run[0]

        kwargs = {
            "initial": 5, # number of seed infections
            "test_prob": 0.03, # testing prob per day
            "popsize": 50000,
            "return_sim": True,
        }

        seeds, betas = get_betas(std_beta, mean_beta,  4)
        response_time = np.array([4, 7, 10, 14])
        daily_vaccs = np.array([5, 10, 20, 35])
        baseline_vax_coverages = np.array([10, 20, 30, 40])

        n_runs = 1
        if n_runs > 1:
            outputs = sc.parallelize(run_calibration, iterarg=[(seed, beta, response_time, daily_vaccs, baseline_vax_coverage) for seed, beta, response_time, daily_vaccs, baseline_vax_coverage in zip(seeds, betas, response_time, daily_vaccs, baseline_vax_coverages)], disease='measles', kwargs=kwargs)  # Run them in parallel
        else:
            with sc.Timer(label="Run model") as _:
                kwargs = {
                    "initial": 5,  # number of seed infections
                    "test_prob": 0.03,  # testing prob per day
                    "popsize": 10000,
                    "return_sim": True,
                    "baseline_vax_coverage": 100 # aiming for percentage of eligible people, who received two doses
                }
                outputs = [run_calibration(0, mean_beta, 7, 30, disease='measles', **kwargs)]
                # Save output df to csv
                outputs[0][0].to_csv("measels_output.csv")
            # df, summary, sim = run_sim(beta, seed, return_sim=True, **kwargs)

        s = MultiSim([x[-1] for x in outputs])
        s.reduce(quantiles={"low": 0.25, "high": 0.75})
        sim = s.base_sim
