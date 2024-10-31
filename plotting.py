####### PLOT RESULTS
import matplotlib.pyplot as plt
import numpy as np
from starsim.gavi import interventions as ssi
import sciris as sc


def plot_cum_infections(sim, ax, disease):
    if sim.results[disease+"-cum_infections"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-cum_infections"].low, sim.results[disease+"-cum_infections"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-cum_infections"].values[:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " infections")

def plot_cum_diagnoses(sim, ax, disease):
    if sim.results[disease+"-cum_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-cum_diagnoses"].low, sim.results[disease+"-cum_diagnoses"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-cum_diagnoses"].values[:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " diagnoses")

def plot_cum_deaths(sim, ax, disease):
    if sim.results[disease+"-cum_deaths"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-cum_deaths"].low, sim.results[disease+"-cum_deaths"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-cum_deaths"].values[:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " deaths")

def plot_cum_sb(sim, ax):
    if sim.results["cum_safe_buried"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results["cum_safe_buried"].low, sim.results["cum_safe_buried"].high, **fill_args)
    ax.plot(sim.tivec, sim.results["cum_safe_buried"].values[:], color="b", alpha=1)

    ax.set_title("Cumulative safe burials")

def plot_cum_tests(sim, ax):
    test_vals = np.zeros(len(sim.tivec))
    for intervention in sim.pars["interventions"]:
        if isinstance(intervention, ssi.test_prob_quarantine):
            name = intervention.label
            test_vals += sim.results[name+"-new_tests"].values[:]
    ax.plot(sim.tivec, test_vals.cumsum(), color="b", alpha=1)

    ax.set_title("Cumulative tests")

def plot_new_contacts(sim, ax, disease):
    if sim.results[disease+"-new_contacts"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-new_contacts"].low, sim.results[disease+"-new_contacts"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-new_contacts"].values[:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " contacts traced")


def plot_new_infections(sim, ax, disease):
    if sim.results[disease+"-new_infections"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-new_infections"].low, sim.results[disease+"-new_infections"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-new_infections"].values[:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " infections")


def plot_new_diagnoses(sim, ax, disease):
    if sim.results[disease+"-new_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-new_diagnoses"].low, sim.results[disease+"-new_diagnoses"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-new_diagnoses"].values[:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " diagnoses")


def plot_new_deaths(sim, ax, disease):
    if sim.results[disease+"-new_deaths"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-new_deaths"].low, sim.results[disease+"-new_deaths"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-new_deaths"].values[:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " deaths")

def plot_new_tests(sim, ax):
    test_vals = np.zeros(len(sim.tivec))
    for intervention in sim.pars["interventions"]:
        if isinstance(intervention, ssi.test_prob_quarantine):
            name = intervention.label
            test_vals += intervention.results[name+"-new_tests"].values[:]
    ax.plot(sim.tivec, test_vals, color="b", alpha=1)

    ax.set_title("Daily tests")

def plot_cum_contacts(sim, ax, disease):
    if sim.results[disease+"-new_contacts"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, np.cumsum(sim.results[disease+"-new_contacts"].low), np.cumsum(sim.results[disease+"-new_contacts"].high), **fill_args)
    ax.plot(sim.tivec, np.cumsum(sim.results[disease+"-new_contacts"].values[:]), color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " contacts traced")

def plot_cum_severe(sim, ax, disease):
    if sim.results[disease+"-new_severe"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, np.cumsum(sim.results[disease+"-new_severe"].low), np.cumsum(sim.results[disease+"-new_severe"].high), **fill_args)

    ax.plot(sim.tivec, np.cumsum(sim.results[disease+"-new_severe"].values[:]), color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " severe cases")


def plot_cum_recovered(sim, ax, disease):
    if sim.results[disease+"-cum_recoveries"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease+"-cum_recoveries"].low, sim.results[disease+"-cum_recoveries"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease+"-cum_recoveries"].values[:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " recoveries")

def plot_cum_severe_YF(sim, ax, disease):
    if sim.results[disease + "-cum_severe"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease + "-cum_severe"].low,
                        sim.results[disease + "-cum_severe"].high, **fill_args)

    ax.plot(sim.tivec, sim.results[disease + "-cum_severe"].values[:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " severe cases")


def plot_vaccine_coverage(sim, ax, disease):
    ax.plot(sim.tivec, sim.results[disease+"-n_vaccinated"].values[:], label="Vaccine doses", color="b")

    ax.legend(loc='upper left')
    ax.set_title("Vaccine coverage (" + disease +")")

def plot_vaccinated_contacts(sim, ax, disease):
    ax.plot(sim.tivec, sim.results[disease+"-cum_vaccinated_contacts"].values[:], color="b", alpha=1)

    ax.set_title("Vaccine coverage in contacts of cases (" + disease +")")

def plot_env_prev(sim, ax):
    ax.plot(sim.tivec, sim.results["cholera-environmental_prev"].values[:], color="b", alpha=1)

    ax.set_title("Prevalence of cholera in environment")

def plot_env_conc(sim, ax):
    ax.plot(sim.tivec, sim.results["cholera-environmental_conc"].values[:], color="b", alpha=1)

    ax.set_title("Concentration of cholera in environment")

def plot_prevalence(sim, ax):
    if sim.results["cholera-prevalence"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results["cholera-prevalence"].low,
                        sim.results["cholera-prevalence"].high, **fill_args)

    ax.plot(sim.tivec, sim.results["cholera-prevalence"].values[:], color="b", alpha=1)

    ax.set_title("Prevalence of cholera in the population")

def plot_yf_prevalence(sim, ax):
    if sim.results["yellow_fever-prevalence"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results["yellow_fever-prevalence"].low,
                        sim.results["yellow_fever-prevalence"].high, **fill_args)

    ax.plot(sim.tivec, sim.results["yellow_fever-prevalence"].values[:], color="b", alpha=1)

    ax.set_title("Prevalence of Yellow Fever in the population")


def plot_mosquito_prev(sim, ax):
    if sim.results["yellow_fever-mosquito_prev"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results["yellow_fever-mosquito_prev"].low,
                        sim.results["yellow_fever-mosquito_prev"].high, **fill_args)

    ax.plot(sim.tivec, sim.results["yellow_fever-mosquito_prev"].values[:], color="b", alpha=1)

    ax.set_title("Prevalence in mosquitoes")

def plot_carrier_prev(sim, ax):
    if sim.results["meningitis-carriage_prevalence"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results["meningitis-carriage_prevalence"].low,
                        sim.results["meningitis-carriage_prevalence"].high, **fill_args)

    ax.plot(sim.tivec, sim.results["meningitis-carriage_prevalence"].values[:], color="b", alpha=1)

    ax.set_title("Asymptomatic meningitis carriage prevalence")


def plot_cum_diagnoses_calib(sim, data, ax):
    if sim.results["cholera-cum_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results["cholera-cum_diagnoses"].low, sim.results["cholera-cum_diagnoses"].high, **fill_args)
    fill_args = {"alpha": 0.3, "color": "r"}
    ax.fill_between(list(range(len(data['median']))), np.cumsum(data['low']), np.cumsum(data['high']), **fill_args)
    ax.plot(sim.tivec, sim.results["cholera-cum_diagnoses"].values[:], color="b", alpha=1)
    ax.plot(list(range(len(data['median']))), np.cumsum(data['median']), color="r", alpha=1)
    ax.set_title("Cumulative cholera diagnoses")

def plot_cum_diagnoses_string(s, data, ax):
    for sim in s.sims:
        ax.plot(sim.tivec, sim.results["cholera-cum_diagnoses"].values[:], color="b", alpha=1)

    for ob in data:
        ax.plot(list(range(len(data[ob]))), np.cumsum(data[ob]), color="r", alpha=0.5)
    ax.set_title("Cumulative cholera diagnoses")

def plot_infection_age_distribution(sim, ax):
    colours = sc.gridcolors(13)
    if sim.results["all_inf_0-1"].low is not None:
        fill_args = {"alpha": 0.3}
        for a, age in enumerate(["0-4", "4-9", "10-19", "20-29", "30+"]):
            ax.fill_between(sim.tivec, sim.results["all_inf_" + age].low, sim.results["all_inf_" + age].high, color=colours[a], **fill_args)
    for a, age in enumerate(["0-4", "4-9", "10-19", "20-29", "30+"]):
        ax.plot(sim.tivec, sim.results["all_inf_" + age].values[:], color=colours[a], alpha=1, label=age)
    ax.set_title("Infection prevalence by age bracket")
    ax.set_ylabel("Proportion infectious")
    ax.legend()

def plot_infection_age_hist(sim, ax, day):
    ages = ["0-4", "4-9", "10-19", "20-29", "30+"]
    ax.bar(ages, [sim.results["all_inf_" + age].values[day] for age in ages])
    ax.set_title("Infection prevalence by age bracket")
    ax.set_ylabel("Proportion infectious")
    ax.legend()

def plot_asymp_age_hist(sim, ax, day):
    ages = ["0-4", "4-9", "10-19", "20-29", "30+"]
    ax.bar(ages, [sim.results["asymp_inf_" + age].values[day] for age in ages])
    ax.set_title("Carriage prevalence by age bracket")
    ax.set_ylabel("Proportion asymp carrier")
    ax.legend()

def plot_symp_age_hist(sim, ax, day):
    ages = ["0-4", "4-9", "10-19", "20-29", "30+"]
    ax.bar(ages, [sim.results["symp_inf_" + age].values[day] for age in ages])
    ax.set_title("IMD prevalence by age bracket")
    ax.set_ylabel("Proportion IMD")
    ax.legend()

def plot_asymp_age_distribution(sim, ax):
    colours = sc.gridcolors(13)
    if sim.results["asymp_inf_0-1"].low is not None:
        fill_args = {"alpha": 0.3}
        for a, age in enumerate(["0-4", "4-9", "10-19", "20-29", "30+"]):
            ax.fill_between(sim.tivec, sim.results["asymp_inf_" + age].low, sim.results["asymp_inf_" + age].high, color=colours[a], **fill_args)
    for a, age in enumerate(["0-4", "4-9", "10-19", "20-29", "30+"]):
        ax.plot(sim.tivec, sim.results["asymp_inf_" + age].values[:], color=colours[a], alpha=1, label=age)
    ax.set_title("Carrier prevalence by age bracket")
    ax.set_ylabel("Proportion asymptomatic carrier")
    ax.legend()

def plot_symp_age_distribution(sim, ax):
    colours = sc.gridcolors(13)
    if sim.results["symp_inf_0-1"].low is not None:
        fill_args = {"alpha": 0.3}
        for a, age in enumerate(["0-4", "4-9", "10-19", "20-29", "30+"]):
            ax.fill_between(sim.tivec, sim.results["symp_inf_" + age].low, sim.results["symp_inf_" + age].high, color=colours[a], **fill_args)
    for a, age in enumerate(["0-4", "4-9", "10-19", "20-29", "30+"]):
        ax.plot(sim.tivec, sim.results["symp_inf_" + age].values[:], color=colours[a], alpha=1, label=age)
    ax.set_title("IMD prevalence by age bracket")
    ax.set_ylabel("Proportion IMD")
    ax.legend()

def plot_cum_active_imd_loop(sims, ax):
    fill_args = {"alpha": 0.3}
    colours = sc.gridcolors(6)
    for i, (prev, sim) in enumerate(sims.items()):
        ax.fill_between(sim.tivec[:250], sim.results["meningitis-n_symptomatic"].low[:250],
                        sim.results["meningitis-n_symptomatic"].high[:250], color=colours[i], **fill_args)

        ax.plot(sim.tivec[:250], sim.results["meningitis-n_symptomatic"].values[:250], color=colours[i], alpha=1, label=prev)

    ax.set_title("Active IMD cases")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['0.1%', '0.5%', '1%', '2%', '5%', '10%'], title="Initial prevalence")
    ax.set_xlabel("Time (days)")
    ax.set_xlim([0, 250])
    ax.set_ylabel("Number of people")

def plot_cum_active_carriers_loop(sims, ax):
    fill_args = {"alpha": 0.3}
    colours = sc.gridcolors(6)
    for i, (prev, sim) in enumerate(sims.items()):
        ax.fill_between(sim.tivec[:250], sim.results["meningitis-n_infectious"].low[:250]-sim.results["meningitis-n_symptomatic"].low[:250],
                        sim.results["meningitis-n_infectious"].high[:250]-sim.results["meningitis-n_symptomatic"].high[:250], color=colours[i], **fill_args)

        ax.plot(sim.tivec[:250], sim.results["meningitis-n_infectious"].values[:250]-sim.results["meningitis-n_symptomatic"].values[:250], color=colours[i], alpha=1, label=prev)

    ax.set_title("Active asymptomatic meningitis carriers")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel("Time (days)")
    ax.set_xlim([0, 250])
    #ax.legend(handles, ['0.1%', '0.5%', '1%', '2%', '5%', '10%'])

def diagnostic_plots(sim, disease):

    # MAIN FIGURE
    if disease == 'cholera' or disease == 'meningitis' :
        fig, ax = plt.subplots(3, 2)
    elif disease == 'yellow_fever':
        fig, ax = plt.subplots(3, 2)
    else:
        fig, ax = plt.subplots(2, 2)

    fig.set_size_inches(10, 10)
    fig.tight_layout(pad=5.0)
    plot_cum_infections(sim, ax[0, 0], disease)
    plot_cum_diagnoses(sim, ax[0, 1], disease)
    #plot_cum_tests(sim, ax[1, 0])
    plot_cum_deaths(sim, ax[1, 0], disease)
    #plot_cum_sb(sim, ax[1, 0])
    plot_vaccine_coverage(sim, ax[1, 1], disease)
    if disease == 'cholera':
        plot_env_prev(sim, ax[2, 0])
        plot_env_conc(sim, ax[2, 1])
        plot_cum_deaths(sim, ax[0, 1], disease)

    elif disease == 'yellow_fever':
        plot_mosquito_prev(sim, ax[2, 0])
        plot_cum_severe_YF(sim, ax[0, 1], disease)
        plot_yf_prevalence(sim, ax[2, 1])

    elif disease == 'meningitis':
        plot_asymp_age_hist(sim, ax[2, 0], -1)
        plot_symp_age_hist(sim, ax[2, 1], -1)

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Key outputs")
    fig.savefig(disease + '_output_check', transparent=True)

def cholera_calibration_plots(sim, s, data):

    # MAIN FIGURE
    fig, ax = plt.subplots(2,2)

    fig.set_size_inches(10, 10)
    fig.tight_layout(pad=5.0)
    plot_cum_diagnoses_string(s, data, ax[0,0])
    plot_cum_deaths(sim, ax[0,1], 'cholera')
    plot_prevalence(sim, ax[1, 0])
    plot_env_conc(sim, ax[1, 1])

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Calibration")
    fig.savefig('cholera_calibration_check', transparent=True)


def plot_r_eff(outputs):
    fig, ax = plt.subplots(1, 1)
    alpha = 0.1
    color = 'b'
    levels = np.arange(0.05, 0.50, 0.05)
    seeds = np.arange(0, len(outputs))
    vals = []
    for seed in seeds:
        vals.append(outputs[seed][0]["r_eff"].values)
    vals = np.array(vals)
    x = np.arange(0, vals.shape[1])
    for level in levels[::-1]:
        ax.fill_between(x, y1=np.quantile(vals, 0.5 - level, axis=0), y2=np.quantile(vals, 0.5 + level, axis=0),
                        linewidth=0, alpha=alpha, color=color)

    ax.plot(x, np.median(vals, axis=0), color=color, label="R_eff")[0]
    ax.set_xlabel("Days")
    ax.set_ylabel("R_eff")
    #ax.set_xlim([0, 200])

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Key outputs")
    fig.savefig("YF_r_eff", transparent=True)

def diagnostic_plots_loop(sims, disease):

    # MAIN FIGURE
    fig, ax = plt.subplots(1, 2)

    fig.set_size_inches(10, 6)
    fig.tight_layout(pad=5.0)
    plot_cum_active_imd_loop(sims, ax[0])
    plot_cum_active_carriers_loop(sims, ax[1])

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Key outputs")
    fig.savefig(disease + '_output_loop_test', transparent=True)

