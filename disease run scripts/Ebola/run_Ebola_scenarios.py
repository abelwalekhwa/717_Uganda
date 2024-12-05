# SCENARIOS FOR EBOLA OUTBREAKS

import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
import numpy as np
from celery import group
from tqdm import tqdm
from gavi.celery import run_sim, celery
from starsim.samples import Samples
import pandas as pd

debug_mode = False  # If True, run just one set of parameters and do not use threading
data_lp = [(4,5), (4,20), (4,35), (10,5), (10,10), (10,20), (14,5)]
result_dir = Path("results")
# Define parameters constant across all simulations
kwargs = {"initial": 3, # number of seed infections
        "popsize": 100000
        }

#to_run = [{"Name": "Med improved detect"}, {"Name": "High improved detect"},
#          {"Name": "Detect day 5"}, {"Name": "Detect day 6"}, {"Name": "Detect day 7"}, {"Name": "Detect day 8"}, {"Name": "Detect day 12"}, {"Name": "Detect day 19"}]
to_run = [{"Name": "No ORI"}]

def get_params(run_name="Baseline"):
    """
        Pull out the parameters from filtered baseline simulations for each outbreak
    """
    if run_name == "No ORI":
        param_file = "filtered_seeds.csv"
    else:
        param_file = "scenario_seeds.csv"
    df = pd.read_csv(param_file)
    seeds = df['seeds'].values
    betas = df['betas'].values
    daily_vaccs = df['daily_vaccs'].values
    response_times = df['response_times'].values
    test_probs = df['test_probs'].values

    if run_name == "No ORI":
        seeds = np.array([seed for s, seed in enumerate(seeds) if (response_times[s], daily_vaccs[s]) in data_lp])
        betas = np.array([beta for s, beta in enumerate(betas) if (response_times[s], daily_vaccs[s]) in data_lp])
        daily_vaccs = np.zeros(len(seeds))
        response_times = np.zeros(len(seeds))
    if run_name == "Med improved detect":
        test_probs = df['test_probs'].values * 5
    elif run_name == "High improved detect":
        test_probs = df['test_probs'].values * 10
    if "Detect day" in run_name:
        if len(run_name) > 12:
            detect_days = int(str(run_name[-2])+str(run_name[-1])) * np.ones(len(seeds))
        else:
            detect_days = int(run_name[-1]) * np.ones(len(seeds))
    else:
        detect_days = np.nan * np.ones(len(seeds))

    return seeds, betas, test_probs, response_times, daily_vaccs, detect_days


def run_scenario(run_args):
    """
        Pull out the parameters from filtered baseline simulations for each outbreak
    """
    # Get input parameters for each simulation to be run
    seeds, betas, test_probs, response_times, daily_vaccs, detect_days = get_params(run_args["Name"])
    if not hasattr(thread_local, "pbar"):
        thread_local.pbar = tqdm(total=len(seeds))
    pbar = thread_local.pbar
    description = run_args["Name"]
    pbar.set_description(description)
    pbar.n = 0
    pbar.refresh()
    pbar.unpause()

    fname = description + ".zip"
    
    if (result_dir / fname).exists():
        return

    # Run simulations using celery
    job = group([run_sim.s(seed, beta, response_time, daily_vaccs, test_prob, detect_day, **kwargs)
                 for seed, beta, test_prob, response_time, daily_vaccs, detect_day in zip(seeds, betas, test_probs, response_times, daily_vaccs, detect_days)])
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
        Samples.new(result_dir, outputs, ["seed"], fname=run_args["Name"] + ".zip")
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

