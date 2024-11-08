from starsim.gavi.celery import run_sim
from starsim.gavi import multisim as ssm
import matplotlib.pyplot as plt
import sciris as sc

if __name__ == "__main__":
    # some parameters for testing measles outbreak simulation
    beta = 0.38
    kwargs = {
        'disease': 'measles',
        'test_prob': 0.10,
        'response_time': 0,
        'daily_vacc': 0,
        'initial': 5,
        'baseline_vax_coverage': 0, # percentage of eligible people fully vaccinated
        'return_sim': True,
        'popsize': 50000,
    }
    n_runs = 1
    if n_runs > 1:
        outputs = sc.parallelize(run_sim, iterarg=[(seed, beta) for seed in range(n_runs)],
                                 kwargs=kwargs)  # Run them in parallel
    else:
        with sc.Timer(label="Run model") as _:
            outputs = [run_sim(0, beta, **kwargs)]

    # Save output df to csv
    outputs[0][0].to_csv("measels_output.csv")

    s = ssm.MultiSim([x[-1] for x in outputs], label="ORI calibration")
    s.reduce(quantiles={"low": 0.25, "high": 0.75})
    sim = s.base_sim

    # diagnostic_plots(sim, 'measles')
    plt.show()