# LATTICE FOR EBOLA OUTBREAKS

import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
import numpy as np
from celery import group
from tqdm import tqdm
from starsim.gavi import MultiSim
from starsim.gavi.celery import run_calibration, stop_calibration_Ebola, celery
from starsim import Samples
import functools

debug_mode = False  # If True, run just one set of parameters and do not use threading
result_dir = Path("results")
# Define parameters constant across all simulations
constant_kwargs = {"initial": 3, # number of seed infections
        "test_prob": 0.02, # testing prob per day
        "popsize": 100000
        }
# Mean and standard deviation for sampling transmission beta
mean_beta = 0.26
std_beta = 0.025
# Create array of response parameters for outbreak simulations
to_run = [
    {'response_time': 4, 'daily_vacc': 5, 'nruns': 1000, 'start_seed': 0, 'ID': '12', 'loc_date': 'COD_2022'},
    {'response_time': 4, 'daily_vacc': 20, 'nruns': 120000, 'start_seed': 1000, 'ID': '08', 'loc_date': 'COD_2020'},
    {'response_time': 4, 'daily_vacc': 35, 'nruns': 5000, 'start_seed': 121000, 'ID': '11', 'loc_date': 'COD_2021'},
    {'response_time': 10, 'daily_vacc': 5, 'nruns': 50000, 'start_seed': 126000, 'ID': '13', 'loc_date': 'COD_2022'},
    {'response_time': 10, 'daily_vacc': 10, 'nruns': 4000, 'start_seed': 130000, 'ID': '09', 'loc_date': 'COD_2021'},
    {'response_time': 10, 'daily_vacc': 20, 'nruns': 6000, 'start_seed': 136000, 'ID': '10', 'loc_date': 'GIN_2021'},
    {'response_time': 14, 'daily_vacc': 5, 'nruns': 70000, 'start_seed': 206000, 'ID': '06', 'loc_date': 'COD_2018'}
]

def get_betas(mean_beta, std_beta, nruns=None, start_seed=0):
    """
    Sample beta values and create seed array for baseline simulations for each outbreak
    """
    seeds = np.arange(start=start_seed, stop=start_seed+nruns)
    betas = np.random.normal(mean_beta, std_beta, nruns)
    return seeds, betas


def run_scenario(kwargs):
    """
        Pull out the parameters for baseline simulations for each outbreak
    """
    nruns = kwargs['nruns']
    response_time = kwargs['response_time'] * np.ones(nruns)
    daily_vaccs = kwargs['daily_vacc'] * np.ones(nruns)
    start_seed = kwargs['start_seed']
    seeds, betas = get_betas(mean_beta, std_beta, len(response_time), start_seed)
    if not hasattr(thread_local, "pbar"):
        thread_local.pbar = tqdm(total=len(seeds))
    pbar = thread_local.pbar
    description = "baseline_lattice_" + str(kwargs['response_time']) + "_" + str(kwargs['daily_vaccs']) + "_" + str(kwargs['ID']) + "_" + str(kwargs['loc_date'])
    pbar.set_description(description)
    pbar.n = 0
    pbar.refresh()
    pbar.unpause()

    fname = description + ".zip"
    if (result_dir / fname).exists():
        return

    # Run simulations using celery
    job = group([run_calibration.s(seed, beta, response_time, daily_vaccs, stopping_func=functools.partial(stop_calibration_Ebola, response_time=response_time,
                                                                   daily_vacc=daily_vaccs), **constant_kwargs) for seed, beta, response_time, daily_vaccs in zip(seeds, betas, response_time, daily_vaccs)])
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
        Samples.new(result_dir, outputs, ["seed"], fname="baseline_filtered_"  + str(kwargs['response_time']) + "_" + str(kwargs['daily_vaccs']) + "_" + str(kwargs['ID']) + "_" + str(kwargs['loc_date']) +".zip")
    else:
        pbar.set_description("baseline ERROR")
        for x in result.results:
            if x.failed():
                with open(result_dir / f"error_{x.id}.txt", "w") as log:
                    log.write(str(x.__dict__))

    result.forget()

    return True

if __name__ == "__main__":
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
            "popsize": 10000,
            "return_sim": True,
        }

        seeds, betas = get_betas(std_beta, mean_beta,  4)
        response_time = np.array([4, 7, 10, 14])
        daily_vaccs = np.array([5, 10, 20, 35])

        n_runs = 1
        if n_runs > 1:
            outputs = sc.parallelize(run_calibration, iterarg=[(seed, beta, response_time, daily_vaccs) for seed, beta, response_time, daily_vaccs in zip(seeds, betas, response_time, daily_vaccs)], kwargs=kwargs)  # Run them in parallel
        else:
            with sc.Timer(label="Run model") as _:
                outputs = [run_calibration(0, mean_beta, 7, 30, **kwargs)]

            # df, summary, sim = run_sim(beta, seed, return_sim=True, **kwargs)

        s = MultiSim([x[-1] for x in outputs])
        s.reduce(quantiles={"low": 0.25, "high": 0.75})
        sim = s.base_sim
