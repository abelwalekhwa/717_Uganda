from starsim.gavi.celery import run_sim, run_calibration, stop_calibration_Ebola
from starsim.gavi import multisim as ssm
from starsim.gavi.plotting import diagnostic_plots
import matplotlib.pyplot as plt
import sciris as sc
import functools

if __name__ == "__main__":
    # some test parameters for running Ebola model
    beta = 0.26
    kwargs = {
        'test_prob': 0.02,
        'response_time': 4,
        'daily_vacc': 20,
        'initial': 3,
        'return_sim': True
    }
    n_runs = 1
    if n_runs > 1:
        outputs = sc.parallelize(run_calibration, iterarg=[(seed, beta) for seed in range(n_runs)],
                                 kwargs=kwargs)  # Run them in parallel
    else:
        with sc.Timer(label="Run model") as _:
            outputs = [run_calibration(0, beta,  **kwargs)]

    #outputs[0][0].to_csv("ebola_output.csv")

    s = ssm.MultiSim([x[-1] for x in outputs], label="ORI calibration")
    s.reduce(quantiles={"low": 0.25, "high": 0.75})
    sim = s.base_sim

    diagnostic_plots(sim, 'ebola')
    plt.show()