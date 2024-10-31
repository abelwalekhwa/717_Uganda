import numpy as np
import starsim as ss
from starsim.gavi import multisim as ssm
from collections import defaultdict
import starsim.gavi.utils as ssg
import sciris as sc


class Meningitis(ss.Disease):
    default_pars = {
        'dur_exp2inf': ss.lognormal_int(mean=2, std=1.5),  # Matched to duration stated by Bosis, Mayer, and Esposito (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4755120/) of 1-14 days, with majority of cases < 2 days
        'dur_inf2rec': ss.normal_pos(mean=7, std=1),  # Assume mean duration of infection of one week, aligns with Caroline Trotter's model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4639487/#CIV508C15)
        'dur_inf2dead': ss.normal_pos_int(mean=8, std=2), # Death can occur within a day, and up to 30 days but paper by Sharew et al argues that deaths beyond 14 days are from other complications, and median time to death is 8 days (https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-020-4899-x)
        'dur_carr2rec': ss.neg_binomial(mean=90, dispersion=7), # Assume mean duration of carriage of three months but long tail, mean aligns with source used in Caroline Trotter's model (https://pubmed.ncbi.nlm.nih.gov/7130749/) and tail with reports of carriage of 5-6 months+
        'prop_symp': 0.25, # Calibrated, as most infections are passive carriers of the bacteria (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC263032/)
        'p_death': 0.07,  # CFR is usually 5-15%, but can be higher according to CDC (https://www.cdc.gov/vaccines/pubs/pinkbook/mening.html) and several studies (https://www.ncbi.nlm.nih.gov/books/NBK549849/, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4755120/)
        'initial': 0,
        'init_carr_prev': 0.05, # proportion of population initialised as asymp carrier, to be calibrated
        'baseline_vax_coverage': 0,
        'second_dose_peak_coverage_rel2_first': 1, # Default: 100%
        'sequence': None,
        'iso_factor': 0.5,  # Multiply beta by this factor for diagnosed cases to represent isolation and impact of antibiotics
        'quar_factor': 1,  # Quarantine multiplier on transmission
        'quar_period': 0,
        'seasonality_modifier': 0.6, # Aligns with Caroline Trotter's model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4639487/#CIV508C15)
        'age_sus_carriage': {"0+": 1.0},
        'age_sus_imd': {"0+": 1.0},
        'beta': None,
    }

    def __init__(self, pars=None):
        """
            Load in initial disease states and parameters
        """
        super().__init__(pars=ss.omerge(self.default_pars, pars))

        if self.pars.beta is None:
            self.pars.beta = {}

        self.results = ss.ndict(type=ssm.MultiSimResult)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.infectious = ss.State('infectious', bool, False)
        self.symptomatic = ss.State('symptomatic', bool, False)
        self.recovered = ss.State('recovered', bool, False)
        self.dead = ss.State('dead', bool, False)

        self.tested = ss.State('tested', bool, False)
        self.diagnosed = ss.State('diagnosed', bool, False)
        self.treated = ss.State('treated', bool, False)
        self.vaccinated = ss.State('vaccinated', bool, False)
        self.fully_vaccinated = ss.State('fully_vaccinated', bool, False)
        self.isolated = ss.State('isolated', bool, False)
        self.quarantined = ss.State('quarantined', bool, False)

        self.ti_infected = ss.State('ti_infected', float, np.nan)
        self.ti_infectious = ss.State('ti_infectious', float, np.nan)
        self.ti_symptomatic = ss.State('ti_symptomatic', float, np.nan)
        self.ti_recovered = ss.State('ti_recovered', float, np.nan)
        self.ti_dead = ss.State('ti_dead', float, np.nan)

        self.ti_tested = ss.State('ti_tested', float, np.nan)
        self.ti_pos_test = ss.State('ti_pos_test', float, np.nan)
        self.ti_diagnosed = ss.State('ti_diagnosed', float, np.nan)
        self.ti_treated = ss.State('ti_treated', float, np.nan)
        self.ti_vaccinated = ss.State('ti_vaccinated', float, np.nan)
        self.ti_isolated = ss.State('ti_isolated', float, np.nan)
        self.ti_quarantined = ss.State('ti_quarantined', float, np.nan)
        self.ti_end_quarantine = ss.State('ti_end_quarantine', float, np.nan)
        self.ti_end_isolation = ss.State('ti_end_isolation', float, np.nan)

        self.immunity_inf = ss.State('immunity_inf', float, 1.0)
        self.immunity_trans = ss.State('immunity_trans', float, 1.0)
        self.immunity_symp = ss.State('immunity_symp', float, 1.0)
        self.base_immunity_inf = ss.State('immunity_inf', float, 1.0)
        self.base_immunity_trans = ss.State('immunity_trans', float, 1.0)
        self.base_immunity_symp = ss.State('base_immunity_symp', float, 1.0)

        self.prob_death = ss.State('prob_death', float, self.pars['p_death'])
        self._pending_quarantine = defaultdict(list)

        return

    def initialize(self, sim):
        """
            Meningitis-specific initialization to adjust age-based disease susceptibilities
        """
        super().initialize(sim)
        self.calc_age_sus(sim)

    @property
    def exposed(self):
        return self.infected & ~self.infectious

    @property
    def _boolean_states(self):
        for state in self.states:
            if state.dtype == bool:
                yield state

    def init_results(self, sim):
        """
        Initialize results
        """
        for state in self._boolean_states:
            self.results += ssm.MultiSimResult(self.name, f'n_{state.name}', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ssm.MultiSimResult(self.name, 'carriage_prevalence', sim.npts, dtype=float)
        self.results += ssm.MultiSimResult(self.name, 'new_infections', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'new_diagnoses', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'new_recoveries', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_infections', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_diagnoses', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_recoveries', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'new_deaths', sim.npts, dtype=int)
        self.results += ssm.MultiSimResult(self.name, 'cum_deaths', sim.npts, dtype=int)

    def update_pre(self, sim):
        """
            Progress disease states each time step
        """
        # Progress exposed -> infectious
        infectious = ss.true(self.infected & (self.ti_infectious <= sim.ti))
        self.infectious[infectious] = True

        # Progress exposed -> symptomatic
        symptomatic = ss.true(self.infected & (self.ti_symptomatic <= sim.ti))
        self.symptomatic[symptomatic] = True

        # Progress infectious -> recovered
        recovered = ss.true(self.infectious & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.infectious[recovered] = False
        self.symptomatic[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        time_increment = sim.tivec[1] - sim.tivec[0]
        previous_time_step_deaths = ss.true(self.ti_dead <= (sim.ti - time_increment))
        deaths = ss.true(self.ti_dead <= sim.ti)
        # deaths = ss.binomial_filter(sim.dt * self.pars['p_death'], ss.true(self.infected))
        self.infected[deaths] = False
        self.infectious[deaths] = False
        self.symptomatic[deaths] = False
        self.dead[deaths] = True
        sim.people.request_death(deaths)
        self.log.add_data(deaths, died=True)
        self.results.new_deaths[sim.ti] = len(deaths) - len(previous_time_step_deaths)

        # Update today's diagnoses
        diagnosed = ss.true(self.ti_diagnosed <= sim.ti)
        self.diagnosed[diagnosed] = True

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

    def calc_age_sus(self, sim):
        """
            Calculate age-based disease susceptibilities, called on initialization.
        """
        age_sus_carriage = self.pars['age_sus_carriage']
        age_sus_imd = self.pars['age_sus_imd']
        rel_age_sus_imd = {key: item / age_sus_carriage[key] for key, item in age_sus_imd.items()} # Conditional probability of IMD, given infection
        # Groups is of the form "16-39, 60+". Ages are inclusive
        # Update age based infection susceptibility
        groups = list(age_sus_carriage.keys())
        groups = [x.strip() for x in groups]
        include = np.full_like(sim.people.age, fill_value=False, dtype=bool)
        for group in groups:
            age_lower, age_upper = ssg.parse_age_range(group)
            use_include = include | ((sim.people.age >= age_lower) & (sim.people.age <= age_upper))
            self.base_immunity_inf[use_include] = age_sus_carriage[group]
            self.immunity_inf[use_include] = age_sus_carriage[group]
        # Update age based invasive disease susceptibility
        groups = list(rel_age_sus_imd.keys())
        groups = [x.strip() for x in groups]
        include = np.full_like(sim.people.age, fill_value=False, dtype=bool)
        for group in groups:
            age_lower, age_upper = ssg.parse_age_range(group)
            use_include = include | ((sim.people.age >= age_lower) & (sim.people.age <= age_upper))
            self.base_immunity_symp[use_include] = rel_age_sus_imd[group]
            self.immunity_symp[use_include] = rel_age_sus_imd[group]

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        if self.pars['initial'] > 0:
            initial_cases = np.random.choice(sim.people.uid, self.pars['initial'], replace=False)
            self.infect(sim, initial_cases)
        else:
            initial_cases = []

        include = np.full_like(sim.people.age, fill_value=False, dtype=bool)
        potential_carriers = include | (sim.people.age >= 1)
        num_people_carrier = int(self.pars['init_carr_prev'] * len(potential_carriers))  # initial prop of the population which are passive carriers
        # Exclude the ones infected above
        seq_infected_removed = np.array([x for x in potential_carriers if x not in initial_cases])
        initial_carriers = seq_infected_removed[:num_people_carrier]
        self.susceptible[initial_carriers] = False
        self.infected[initial_carriers] = True
        self.infectious[initial_carriers] = True
        self.ti_infected[initial_carriers] = 0
        self.ti_infectious[initial_carriers] = 0
        self.ti_recovered[initial_carriers] = self.pars['dur_carr2rec'].sample(sum(initial_carriers))
        return

    def infect(self, sim, uids, from_uids=None):
        """
            Infect new agents and determine outcomes
        """
        super().set_prognoses(sim, uids, from_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.symptomatic[uids] = False
        self.diagnosed[uids] = False
        self.treated[uids] = False
        self.recovered[uids] = False
        self.base_immunity_inf[uids] = 0  # Assumption of no reinfection
        self.immunity_inf[uids] = 0
        self.ti_infected[uids] = sim.ti
        self.ti_pos_test[uids] = np.nan
        self.ti_diagnosed[uids] = np.nan
        self.ti_recovered[uids] = np.nan

        #calc seasonal impact on probability of IMD/symptoms
        seasonal_term = self.seasonality_impact(sim)

        # determine when all infected become infectious
        self.ti_infectious[uids] = sim.ti + self.pars['dur_exp2inf'].sample(len(uids))

        # determine who becomes symptomatic and when
        symptomatic = np.random.random(len(uids)) < self.pars.prop_symp * seasonal_term * self.immunity_symp[uids].values
        self.ti_symptomatic[uids[symptomatic]] = sc.dcp(self.ti_infectious[uids[symptomatic]])
        self.ti_diagnosed[uids[symptomatic]] = self.ti_symptomatic[uids[symptomatic]]

        # determine who dies and when
        dead = np.random.random(sum(symptomatic)) < self.pars.p_death
        dead = np.array([True if uid in uids[symptomatic][dead] else False for uid in uids])
        self.ti_dead[uids[dead]] = self.ti_symptomatic[uids[dead]] + self.pars['dur_inf2dead'].sample(sum(dead))

        # determine when agents recover
        self.ti_recovered[uids[(~dead & symptomatic)]] = self.ti_infectious[uids[(~dead & symptomatic)]] + self.pars['dur_inf2rec'].sample(sum((~dead & symptomatic)))
        self.ti_recovered[uids[~symptomatic]] = self.ti_infectious[uids[~symptomatic]] + self.pars['dur_carr2rec'].sample(sum(~symptomatic))

        return

    def compute_trans_sus(self, rel_trans, rel_sus, pars, layer):  # pragma: no cover
        '''

        Calculate relative transmissibility and susceptibility '''
        f_iso = ~self.isolated + self.isolated * pars["iso_factor"][
            layer]  # Isolation factor changes e.g. [0,1] with a factor of 0.2 to [1,0.2]
        f_quar = ~self.quarantined + self.quarantined * pars["quar_factor"][
            layer]  # Quarantine factor changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
        rel_trans = rel_trans * self.infectious * f_quar * f_iso * self.immunity_trans # Recalculate transmissibility
        rel_sus = rel_sus * f_quar * self.immunity_inf  # Recalculate susceptibility
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

        is_infectious = uids[self.infectious[uids]]
        pos_test = ss.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.ti_diagnosed[is_inf_pos])]
        not_lost = ss.n_binomial(1.0 - loss_prob, len(not_diagnosed))
        final_uids = not_diagnosed[not_lost]

        return final_uids

    def seasonality_impact(self, sim):
        '''
        Seasonal forcing term transmission, high transmission in first half of year and low in second
        '''
        return 1 + self.pars['seasonality_modifier'] * np.cos(2 * np.pi * sim.ti / 365)

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
        pars = sim.pars[self.name]
        seasonal_term = self.seasonality_impact(sim)
        for k, layer in sim.people.networks.items():
            if k in pars['beta']:
                prel_trans = (self.infectious & sim.people.alive).astype(float)
                prel_sus = (self.susceptible & sim.people.alive).astype(float)
                rel_trans, rel_sus = self.compute_trans_sus(prel_trans, prel_sus, pars, k)
                for a, b, beta in [[layer.contacts.p1, layer.contacts.p2, pars['beta'][k]],
                                   [layer.contacts.p2, layer.contacts.p1, pars['beta'][k]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.contacts.beta * beta * seasonal_term
                    new_cases = np.random.random(len(a)) < p_transmit
                    if new_cases.any():
                        self.infect(sim, b[new_cases], a[new_cases])

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
        """
        Anyone diagnosed today enters isolation for the duration of their infection
        """
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

        if sim.ti > 0:
            for state in self._boolean_states:
                self.results[f'n_{state.name}'].values[sim.ti] = np.count_nonzero(state)
            self.results['prevalence'].values[sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
            self.results['new_infections'].values[sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
            self.results['new_diagnoses'].values[sim.ti] = np.count_nonzero(self.ti_diagnosed == sim.ti)
            self.results['new_recoveries'].values[sim.ti] = np.count_nonzero(self.ti_recovered == sim.ti)
            self.results['cum_infections'].values[sim.ti] = np.sum(self.results['new_infections'].values[:sim.ti])
            self.results['cum_diagnoses'].values[sim.ti] = np.sum(self.results['new_diagnoses'].values[:sim.ti])
            self.results['cum_recoveries'].values[sim.ti] = np.sum(self.results['new_recoveries'].values[:sim.ti])
            # self.results['new_deaths'].values[sim.ti] = np.count_nonzero(self.ti_dead == sim.ti)
            self.results['cum_deaths'].values[sim.ti] = np.sum(self.results['new_deaths'].values[:sim.ti])
            # self.results['new_contacts'].values[sim.ti] = np.count_nonzero(self.ti_known_contact == sim.ti)
        self.results['carriage_prevalence'].values[sim.ti] = np.sum((self.infectious & ~self.symptomatic)) / np.count_nonzero(sim.people.alive)
    def finalize_results(self, sim):
        pass



class MeningitisVaccine:
    def __init__(self, name, immunity_timecourse, protection_timecourse, prevent_infection, prevent_transmission,
                 prevent_symp, prevent_death, clear_carriage):
        """
        Vaccine method for meningitis
        Args:
            characteristics:  Vaccine characteristic dictionary - containing poi, pos, poh, poc, pod
            rising_immunity:
        """

        self.name = name
        self.disease = 'meningitis'
        self.prevent_infection = prevent_infection  # Prevention of infection
        self.prevent_transmission = prevent_transmission
        self.prevent_symp = prevent_symp  # Prevention of symptomatic disease/IMD
        self.prevent_death = prevent_death  # Prevention of death
        self.clear_carriage = clear_carriage  # Clearance of existing asymptomatic carriage
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
    @classmethod
    def meningitis_conjugate_vacc(cls):
        # Meningitis parameters, parametrized by dose interval

        # time_to_peak = 7*4  # Immunity after second dose reaches its peak after this time
        time_to_peak = 8  # Immunity after first dose reaches its peak after this time, estimated to take 7-10 days (https://www.sciencedirect.com/science/article/pii/S1877050913005693)
        waning_time = 3 * 365 # Efficacy wanes by about 11% p.a. over 3 years, maintains to at least 8 according to CDC (https://www.cdc.gov/vaccines/pubs/pinkbook/mening.html#:~:text=Immunogenicity%20and%20Vaccine%20Effectiveness&text=Effectiveness%20waned%20over%20time%3B%20VE,to%208%20years%20after%20vaccination.)
        immunity_timecourse = [(0, 0), (time_to_peak, 1)] #, (waning_time, 0.66)] # ignore waning
        protection_timecourse = [(0, 0), (time_to_peak, 1)] #, (waning_time, 0.66)]

        vaccine_characteristics = {
            "prevent_infection": 0.41,  # Align with impact from Caroline Trotter's model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10284610/#r6)
            "prevent_transmission": 0,  # no indication of protection against onward transmission
            "prevent_symp": 0.9,  # efficacy is only protective against IMD, effectiveness across A,C,Y,W-135 estimated to be 85-100% according to Martinez et al and others (https://www.sciencedirect.com/science/article/pii/S1877050913005693, https://www.sciencedirect.com/science/article/pii/S0140673607610162#bib53), and 90% according to Daugla et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3898950/)
            "prevent_death": 0,
            "clear_carriage": 0,
        }
        return cls(name="meningitis_conjugate_vacc", immunity_timecourse=immunity_timecourse,
                   protection_timecourse=protection_timecourse, **vaccine_characteristics)

    @classmethod
    def meningitis_polysacc_vacc(cls):
        # Meningitis parameters, parametrized by dose interval

        # time_to_peak = 7*4  # Immunity after second dose reaches its peak after this time
        time_to_peak = 8  # Immunity after first dose reaches its peak after this time, estimated to take 7-10 days (https://www.sciencedirect.com/science/article/pii/S1877050913005693)
        waning_time = 3 * 365  # Efficacy wanes by about 11% p.a. over 3 years, maintains to at least 8 according to CDC (https://www.cdc.gov/vaccines/pubs/pinkbook/mening.html#:~:text=Immunogenicity%20and%20Vaccine%20Effectiveness&text=Effectiveness%20waned%20over%20time%3B%20VE,to%208%20years%20after%20vaccination.)
        immunity_timecourse = [(0, 0), (time_to_peak, 1)]  # , (waning_time, 0.66)] # ignore waning
        protection_timecourse = [(0, 0), (time_to_peak, 1)]  # , (waning_time, 0.66)]

        vaccine_characteristics = {
            "prevent_infection": 0, # no indication of protection against carriage
            "prevent_transmission": 0,  # no indication of protection against onward transmission
            "prevent_symp": 0.9, # efficacy is only protective against IMD, effectiveness across A,C,Y,W-135 estimated to be 85-100% according to Martinez et al and others (https://www.sciencedirect.com/science/article/pii/S1877050913005693, https://www.sciencedirect.com/science/article/pii/S0140673607610162#bib53), and 90% according to Daugla et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3898950/)
            "prevent_death": 0,
            "clear_carriage": 0,
        }
        return cls(name="meningitis_polysacc_vacc", immunity_timecourse=immunity_timecourse,
                   protection_timecourse=protection_timecourse, **vaccine_characteristics)
