import numpy as np
from numpy import ndarray

import starsim as ss
from starsim.gavi import multisim as ssm
from collections import defaultdict
import sciris as sc


class Yellow_Fever(ss.Disease):
    default_pars = {
        'dur_exp2inf': ss.uniform_discrete(low=3, high=6),  # WHO incubation period
        'dur_sym2sev': ss.uniform_discrete(low=4, high=5), # 3-4 days + patients enter a more toxic phase within 24hrs of recovering from initial symptoms (https://www.who.int/news-room/fact-sheets/detail/yellow-fever)
        'dur_sev2dead': ss.uniform_discrete(low=7, high=10), # Patients in toxic phase die within 7-10 days (https://www.who.int/news-room/fact-sheets/detail/yellow-fever)
        'dur_mild2rec': ss.uniform_discrete(low=3, high=4),  # Symptoms disappear after 3 to 4 days (ss.uniform(low=7, high=10))
        'dur_sev2rec': ss.uniform_discrete(low=7, high=10),  # No data, use sev2dead duration?
        'p_sev': 0.12, # https://academic.oup.com/trstmh/article/108/8/482/2765182, 12% severe, 55% asymptomatic, 33% mild
        'p_death': 0.20, # Calibrated Value
        'baseline_vax_coverage': 0,
        'second_dose_peak_coverage_rel2_first': 1,  # Default: 100%
        'sequence': None,
        'iso_factor': 0.8,  # Multiply beta by this factor for diagnosed cases to represent isolation
        'quar_factor': 0.9,  # Quarantine multiplier on tr
        'quar_period': 21,
        'beta_direct': None,
        'beta_human2mosquito': 0.15,
        'beta_mosquito2human': 0.15,
        # 'half_sat_rate': 1000000, # Infectious dose in water sufficient to produce infection in 50% of  exposed, from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/)
        #'biting_rate': 3,
        # Rate at which infectious people shed bacteria to the environment (per day), from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/)
        'mortality_rate': 0.033,
        'dur_exp2inf_mosquito': 5,
        'dur_inf2rec_mosquito': 5
    }

    def __init__(self, pars=None):
        """
            Load in initial disease states and parameters
        """
        super().__init__(pars=ss.omerge(self.default_pars, pars))

        if self.pars.beta_direct is None:
            self.pars.beta_direct = {}

        self.results = ss.ndict(type=ssm.MultiSimResult)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.infectious = ss.State('infectious', bool, False)
        self.severe = ss.State('severe', bool, False)
        self.recovered = ss.State('recovered', bool, False)
        self.dead = ss.State('dead', bool, False)

        self.mosquito_prev = None
        self.mosquito_exposed = None
        self.mosquito_susceptible = None
        self.mosquito_recovered = None

        self.tested = ss.State('tested', bool, False)
        self.diagnosed = ss.State('diagnosed', bool, False)
        self.vaccinated = ss.State('vaccinated', bool, False)
        self.fully_vaccinated = ss.State('fully_vaccinated', bool, False)
        self.isolated = ss.State('isolated', bool, False)
        self.quarantined = ss.State('quarantined', bool, False)
        self.known_contact = ss.State('known_contact', bool, False)

        self.ti_infected = ss.State('ti_infected', float, np.nan)
        self.ti_infectious = ss.State('ti_infectious', float, np.nan)
        self.ti_severe = ss.State('ti_severe', float, np.nan)
        self.ti_recovered = ss.State('ti_recovered', float, np.nan)
        self.ti_dead = ss.State('ti_dead', float, np.nan)

        self.ti_tested = ss.State('ti_tested', float, np.nan)
        self.ti_pos_test = ss.State('ti_pos_test', float, np.nan)
        self.ti_diagnosed = ss.State('ti_diagnosed', float, np.nan)
        self.ti_vaccinated = ss.State('ti_vaccinated', float, np.nan)
        self.ti_isolated = ss.State('ti_isolated', float, np.nan)
        self.ti_quarantined = ss.State('ti_quarantined', float, np.nan)
        self.ti_end_quarantine = ss.State('ti_end_quarantine', float, np.nan)
        self.ti_end_isolation = ss.State('ti_end_isolation', float, np.nan)
        self.ti_known_contact = ss.State('ti_known_contact', float, np.nan)

        self.immunity_inf = ss.State('immunity_inf', float, 1.0)
        self.immunity_trans = ss.State('immunity_trans', float, 1.0)
        self.base_immunity_inf = ss.State('immunity_inf', float, 1.0)
        self.base_immunity_trans = ss.State('immunity_trans', float, 1.0)

        self.prob_sev = ss.State('prob_sev', float, self.pars['p_sev'])
        self.prob_death = ss.State('prob_death', float, self.pars['p_death'])
        self._pending_quarantine = defaultdict(list)

        return

    @property
    def exposed(self):
        return self.infected & ~self.infectious

    @property
    def symptomatic(self):
        return self.infectious

    @property
    def _boolean_states(self):
        for state in self.states:
            if state.dtype == bool:
                yield state

    def initialize(self, sim):
        """
            Yellow fever-specific initialization for mosquito population arrays
        """
        super().initialize(sim)
        self.mosquito_prev = np.zeros(sim.npts)
        self.mosquito_susceptible = np.ones(sim.npts)
        self.mosquito_exposed = np.zeros(sim.npts)
        self.mosquito_recovered = np.zeros(sim.npts)

        return

    def init_results(self, sim):
        """
        Initialize results
        """
        for state in self._boolean_states:
            self.results += ssm.MultiSimResult(self.name, f'n_{state.name}', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ssm.MultiSimResult(self.name, 'new_infections', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'new_diagnoses', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_infections', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_diagnoses', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'new_deaths', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_deaths', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'new_severe', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_severe', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'mosquito_prev', sim.npts, dtype=float)

    def update_pre(self, sim):
        """
            Progress disease states each time step
        """
        # Progress exposed -> infectious
        infectious = ss.true(self.infected & (self.ti_infectious <= sim.ti))
        self.infectious[infectious] = True

        # Progress infectious -> recovered
        recovered = ss.true(self.infectious & ~self.severe & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.infectious[recovered] = False
        self.recovered[recovered] = True

        # Progress infectious -> severe
        severe = ss.true(self.infectious & (self.ti_severe <= sim.ti))
        self.severe[severe] = True

        # Progress severe -> recovered
        recovered = ss.true(self.severe & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.infectious[recovered] = False
        self.severe[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        time_increment = sim.tivec[1] - sim.tivec[0]
        previous_time_step_deaths = ss.true(self.ti_dead <= (sim.ti - time_increment))
        deaths = ss.true(self.ti_dead <= sim.ti)
        self.infected[deaths] = False
        self.infectious[deaths] = False
        self.severe[deaths] = False
        self.dead[deaths] = True
        sim.people.request_death(deaths)
        self.log.add_data(deaths, died=True)
        self.results.new_deaths[sim.ti] = len(deaths) - len(previous_time_step_deaths)

        # Update today's diagnoses
        diagnosed = ss.true(self.ti_diagnosed <= sim.ti)
        self.diagnosed[diagnosed] = True

        # Update today's mosquito prevalence
        self.calc_mosquito_prev(sim)

    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
        super().validate_pars(sim)
        pars_to_expand = ['iso_factor', 'quar_factor']
        for k in pars_to_expand:
            if not isinstance(self.pars[k], dict):
                self.pars[k] = {n: self.pars[k] for n in sim.people.networks}
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        initial_cases = np.random.choice(sim.people.uid, self.pars['initial'], replace=False)
        self.infect(sim, initial_cases)

        num_people_eligible = len(sim.pars.interventions[2].sequence)  # TODO: Find a prettier way of doing this..
        initial_vacc_abs = int(self.pars['baseline_vax_coverage'] * num_people_eligible)
        # Exclude the ones infected above
        seq_infected_removed = np.array([x for x in self.pars['sequence'] if x not in initial_cases])
        initial_vaccinated = seq_infected_removed[:initial_vacc_abs]
        self.vaccinated[initial_vaccinated] = True
        self.fully_vaccinated[initial_vaccinated] = True

        return

    def calc_mosquito_prev(self, sim):
        '''
        Keep track of number of newly infected mosquitoes per time step
        '''
        if sim.ti > 0:

            dt = 1
            new_exposed = self.mosquito_susceptible[sim.ti - 1] * self.pars['beta_human2mosquito'] * np.sum(self.infected)/(np.sum(self.susceptible) + np.sum(self.recovered))
            new_infectious = self.mosquito_exposed[sim.ti - 1] * dt/self.pars['dur_exp2inf_mosquito']
            new_recovered = self.mosquito_prev[sim.ti-1] * dt/self.pars['dur_inf2rec_mosquito']

            self.mosquito_susceptible[sim.ti] = self.mosquito_susceptible[sim.ti - 1] - new_exposed + new_recovered
            self.mosquito_exposed[sim.ti] = self.mosquito_exposed[sim.ti - 1] + new_exposed - new_infectious
            self.mosquito_prev[sim.ti] = self.mosquito_prev[sim.ti-1] + new_exposed - new_recovered

        else:
            self.mosquito_prev[sim.ti] = 0


    def infect(self, sim, uids, from_uids=None):
        """
            Infect new agents and determine outcomes
        """
        super().set_prognoses(sim, uids, from_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.infectious[uids] = False
        self.severe[uids] = False
        self.diagnosed[uids] = False
        self.recovered[uids] = False
        self.base_immunity_inf[uids] = 0.0  # Assumption of no reinfection
        self.immunity_inf[uids] = 0.0
        self.ti_infected[uids] = sim.ti
        self.ti_pos_test[uids] = np.nan
        self.ti_diagnosed[uids] = np.nan
        self.ti_severe[uids] = np.nan
        self.ti_recovered[uids] = np.nan

        # determine when all infected become infectious
        self.ti_infectious[uids] = sim.ti + self.pars['dur_exp2inf'].sample(len(uids))

        # determine who progresses to severe disease and when
        severe = np.random.random(len(uids)) < self.pars.p_sev
        #severe = np.array([True if uid in uids[self.infected][severe] else False for uid in uids])
        self.ti_severe[uids[severe]] = self.ti_infectious[uids[severe]] + self.pars['dur_sym2sev'].sample(sum(severe))

        # determine who dies and when
        dead = np.random.random(sum(severe)) < self.pars.p_death
        dead = np.array([True if uid in uids[severe][dead] else False for uid in uids])
        self.ti_dead[uids[dead]] = self.ti_severe[uids[dead]] + self.pars['dur_sev2dead'].sample(sum(dead))

        # determine when agents recover
        self.ti_recovered[uids[~dead]] = self.ti_severe[uids[~dead]] + self.pars['dur_sev2rec'].sample(sum(~dead))
        self.ti_recovered[uids[~severe]] = self.ti_infectious[uids[~severe]] + self.pars['dur_mild2rec'].sample(sum(~severe))
        return

    def compute_trans_sus(self, rel_trans, rel_sus, pars, layer):  # pragma: no cover
        '''
        Calculate relative transmissibility and susceptibility
        '''
        f_iso = ~self.isolated + self.isolated * pars["iso_factor"][
            layer]  # Isolation factor changes e.g. [0,1] with a factor of 0.2 to [1,0.2]
        f_quar = ~self.quarantined + self.quarantined * pars["quar_factor"][
            layer]  # Quarantine factor changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
        f_asymp = self.symptomatic + ~self.symptomatic * pars[
            "asymp_trans"]  # Quarantine factor changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
        rel_trans = rel_trans * self.infectious * f_quar * f_iso * f_asymp * self.immunity_trans  # Recalculate transmissibility
        rel_sus = rel_sus * self.susceptible * f_quar * self.immunity_inf  # Recalculate susceptibility
        return rel_trans, rel_sus

    def test(self, uids, t, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''

        uids = np.unique(uids)
        self.tested[uids] = True
        self.ti_tested[uids] = t  # Only keep the last time they tested

        is_symptomatic = uids[self.symptomatic[uids]]
        pos_test = ss.n_binomial(test_sensitivity, len(is_symptomatic))
        is_inf_pos = is_symptomatic[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.ti_diagnosed[is_inf_pos])]
        not_lost = ss.n_binomial(1.0 - loss_prob, len(not_diagnosed))
        final_uids = not_diagnosed[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.ti_diagnosed[final_uids] = t + test_delay
        self.ti_pos_test[final_uids] = t

        return final_uids

    def schedule_quarantine(self, uids, t, start_date=None, period=None):
        '''
        Schedule a quarantine. Typically not called by the user directly except
        via a custom intervention; see the contact_tracing() intervention instead.

        This function will create a request to quarantine a person on the start_date for
        a period of time. Whether they are on an existing quarantine that gets extended, or
        whether they are no longer eligible for quarantine, will be checked when the start_date
        is reached.

        Args:
            inds (int): indices of who to quarantine, specified by check_quar()
            start_date (int): day to begin quarantine (defaults to the current day, `sim.t`)
            period (int): quarantine duration (defaults to ``pars['quar_period']``)
        '''

        start_date = t if start_date is None else int(start_date)
        period = self.pars['quar_period'] if period is None else int(period)
        for uid in uids:
            self._pending_quarantine[start_date].append((uid, start_date + period))
        return

    def make_new_cases(self, sim):
        """
            Calculate infection probabilities for each interaction at this timestep.
        """
        prel_sus = (self.susceptible & sim.people.alive).astype(float)
        rel_sus = prel_sus * self.susceptible * self.immunity_inf  # Recalculate susceptibility
        p_transmit = self.mosquito_prev[sim.ti] * self.pars['beta_mosquito2human'] * rel_sus[self.susceptible.values]

        new_cases = np.random.random(sum(self.susceptible.values)) < p_transmit
        if new_cases.any():
            susceptible = self.susceptible[new_cases.uid[new_cases.values]]
            self.infect(sim, susceptible.uid)

    def check_uids(self, current, date, t, filter_uids=None):
        '''
        Return indices for which the current state is false and which meet the date criterion
        '''
        if filter_uids is None:
            not_current = ss.false(current)
        else:
            not_current = filter_uids[np.logical_not(current[filter_uids])]
        has_date = not_current[~np.isnan(date[not_current])]
        uids = has_date[t >= date[has_date]]
        return uids

    def check_quar(self, t):
        '''
        Update quarantine state
        '''

        for uid, end_day in self._pending_quarantine[t]:
            if self.quarantined[uid]:
                self.ti_end_quarantine[uid] = max(self.ti_end_quarantine[uid], end_day)  # Extend quarantine if required
            elif not (self.recovered[uid] or self.diagnosed[uid] or self.isolated[
                uid]):  # Unclear whether recovered should be included here # elif not (self.buried[ind] or self.diagnosed[ind]):
                self.quarantined[uid] = True
                self.ti_quarantined[uid] = t
                self.ti_end_quarantine[uid] = end_day

        # If someone has been diagnosed today, end their quarantine
        # By definition, 'quarantine' only applies to people that are not yet diagnosed
        # After diagnosis, they are 'isolating'
        diag_uids = ss.true(self.quarantined & (self.ti_diagnosed == t))
        self.ti_end_quarantine[diag_uids] = t

        # If someone on quarantine has reached the end of their quarantine, release them
        end_uids = self.check_uids(~self.quarantined, self.ti_end_quarantine, t,
                                   filter_uids=None)  # Note the double-negative here (~)
        self.quarantined[end_uids] = False  # Release from quarantine

    def check_enter_iso(self, t):
        '''
        Anyone diagnosed today enters isolation for the duration of their infection
        '''
        iso_uids = ss.true(self.ti_diagnosed == t)
        self.isolated[iso_uids] = True
        self.ti_end_isolation[iso_uids] = self.ti_recovered[iso_uids]

    def check_exit_iso(self, t):
        '''
        End isolation for anyone due to exit isolation
        '''
        end_uids = self.check_uids(~self.isolated, self.ti_end_isolation, t,
                                   filter_uids=None)  # Note the double-negative here (~)
        self.isolated[end_uids] = False  # Release from isolation

    def update_post(self, sim):
        """
        Clean up quarantine and isolation states at the end of the time step
        """
        self.check_quar(sim.ti)
        self.check_enter_iso(sim.ti)
        self.check_exit_iso(sim.ti)

    def update_results(self, sim):
        """
        Update results at the end of the time step
        """

        for state in self._boolean_states:
            self.results[f'n_{state.name}'].values[sim.ti] = np.count_nonzero(state)
        self.results['prevalence'].values[sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'].values[sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
        self.results['new_diagnoses'].values[sim.ti] = np.count_nonzero(self.ti_pos_test == sim.ti)
        self.results['cum_infections'].values[sim.ti] = np.sum(self.results['new_infections'].values[:sim.ti + 1])
        self.results['cum_diagnoses'].values[sim.ti] = np.sum(self.results['new_diagnoses'].values[:sim.ti + 1])
        self.results['cum_deaths'].values[sim.ti] = np.sum(self.results['new_deaths'].values[:sim.ti + 1])
        self.results['new_severe'].values[sim.ti] = np.count_nonzero(self.ti_severe == sim.ti)
        self.results['cum_severe'].values[sim.ti] = np.sum(self.results['new_severe'].values[:sim.ti + 1])
        self.results['mosquito_prev'].values[sim.ti] = self.mosquito_prev[sim.ti]

    def finalize_results(self, sim):
        pass


class Yellow_FeverVaccine:
    def __init__(self, name, immunity_timecourse, protection_timecourse, prevent_infection, prevent_transmission,
                 prevent_symp, prevent_death, prevent_severe):
        """
        Vaccine method for yellow fever
        Args:
            characteristics:  Vaccine characteristic dictionary - containing poi, pos, poh, poc, pod
            rising_immunity:
        """

        self.name = name
        self.disease = 'yellow_fever'
        self.prevent_infection = prevent_infection  # Prevention of infection
        self.prevent_transmission = prevent_transmission
        self.prevent_symp = prevent_symp  # Prevention of symptoms
        self.prevent_death = prevent_death  # Prevention of death
        self.prevent_severe = prevent_severe
        self.dose_interval = None

        self._immunity_timecourse = immunity_timecourse  # Timecourse for protection against infection (poi); [(t,v)] list of interpolation control points
        self._protection_timecourse = protection_timecourse  # Timecourse for protection against all other states (pos, poh, poc, pod); [(t,v)] list of interpolation control points

    def _interpolate(self, vals: list, t):
        vals = sorted(vals, key=lambda x: x[0])  # Make sure values are sorted
        assert len({x[0] for x in vals}) == len(vals)  # Make sure time points are unique
        return np.interp(t, [x[0] for x in vals], [x[1] for x in vals], left=vals[0][1], right=vals[-1][1])

    def immunity_timecourse(self, t: np.array) -> np.array:
        return self._interpolate(self._immunity_timecourse, t)

    def protection_timecourse(self, t: np.array) -> np.array:
        return self._interpolate(self._protection_timecourse, t)

    @property
    def full_protection_time(self) -> int:
        # Return time taken to reach full immunity
        return max(x[0] for x in self._immunity_timecourse + self._protection_timecourse)

    # Constructors for common vaccine types
    # ToDo: review and decide on waning
    @classmethod
    def yellow_fever_vacc(cls):
        # Yellow Fever Vaccine
        # WHO: 80-100% after 10 days, 99% after 30 days (https://www.who.int/news-room/fact-sheets/detail/yellow-fever)
        # Using Median 97.5% as per systematic meta review but with WHO time courses
        time_to_peak_1 = 10
        time_to_peak_2 = 30

        immunity_timecourse = [(0, 0), (time_to_peak_1, 0.80 / 0.975), (time_to_peak_2, 1)]
        protection_timecourse = [(0, 0), (time_to_peak_1, 0.80 / 0.975), (time_to_peak_2, 1)]

        vaccine_characteristics = {
            "prevent_infection": 0.975,
            # Keith Fraser Paper, median value Section 3.1 https://www.medrxiv.org/content/10.1101/2023.12.19.23300139v2.full.pdf#page=4&zoom=100,109,726
            "prevent_transmission": 0,  # no data
            "prevent_severe": 0, # no data
            "prevent_symp": 0,  # no data
            "prevent_death": 0}  # no data?
        return cls(name="yellow_fever_vacc", immunity_timecourse=immunity_timecourse,
                   protection_timecourse=protection_timecourse, **vaccine_characteristics)
