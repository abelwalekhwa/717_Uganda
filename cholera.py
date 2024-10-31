import numpy as np
import starsim as ss
from starsim.gavi import multisim as ssm
from collections import defaultdict

class Cholera(ss.Disease):
    default_pars = {
        'dur_exp2inf': ss.lognormal_int(mean=2.772, std=4.737),  # Calculated from Azman et al. estimates https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3677557/
        'dur_asymp2rec': ss.uniform(low=1, high=10),  # From WHO cholera fact sheet, asymptomatic individuals shed bacteria for 1-10 days (https://www.who.int/news-room/fact-sheets/detail/cholera)
        'dur_symp2rec': ss.lognormal_int(mean=5, std=1.8),  # According to Fung most modelling studies assume 5 days of symptoms (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3926264/), but found a range of 2.9-14 days. Distribution approximately fit to these values
        'dur_symp2dead': ss.lognormal_int(mean=1, std=0.5), #  There does not appear to be differences in timing/duration of mild vs severe disease, but death from severe disease happens rapidly https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5767916/
        'p_death': 0.005,  #  Probability of death is typically less than 1% when treated
        'prop_symp': 0.5, # Proportion of infected which are symptomatic, mid range of ~25% and 57% estimates from Jaclson et al (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3795095/) and Nelson et al (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3842031/), respectively
        'asymp_trans': 0.1, # Reduction in transmission probability for asymptomatic infection, asymptomatic carriers shed 100-1000 times less bacteria than symptomatic carriers (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3084143/ and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3842031/). Previous models assume a 10% relative transmissibility (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4238032/)
        'initial': 5,
        'baseline_vax_coverage': 0,
        'second_dose_peak_coverage_rel2_first': 1, # Default: 100%
        'sequence': None,
        'iso_factor': 0.8,  # Multiply beta by this factor for diagnosed cases to represent isolation
        'quar_factor': 0.9,  # Quarantine multiplier on tr
        'WASH_factor': 0.7, # Reduction in proportion of susceptibles at risk of environmental transmission due to access to clean water
        'hygiene_factor': 0.8,  # Reduction in rate of environmental shedding from infecteds due to improved hygiene behaviours
        'quar_period': 21,
        'beta_direct': None,
        'beta_environment_mult': 0.5/3, # Scaling factor for transmission from environment,
        'half_sat_rate': 1000000, # Infectious dose in water sufficient to produce infection in 50% of  exposed, from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/)
        'shedding_rate': 10, # Rate at which infectious people shed bacteria to the environment (per day), from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/)
        'decay_rate': 0.033, # Rate at which bacteria in the environment dies (per day), from Chao et al. and Mukandavire et al. citing https://pubmed.ncbi.nlm.nih.gov/8882180/
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
        self.symptomatic = ss.State('symptomatic', bool, False)
        self.recovered = ss.State('recovered', bool, False)
        self.dead = ss.State('dead', bool, False)

        self.environmental_prev = None
        self.environment_concentration = None

        self.tested = ss.State('tested', bool, False)
        self.diagnosed = ss.State('diagnosed', bool, False)
        self.vaccinated = ss.State('vaccinated', bool, False)
        self.fully_vaccinated = ss.State('fully_vaccinated', bool, False)
        self.isolated = ss.State('isolated', bool, False)
        self.quarantined = ss.State('quarantined', bool, False)
        self.known_contact = ss.State('known_contact', bool, False)

        self.ti_infected = ss.State('ti_infected', float, np.nan)
        self.ti_infectious = ss.State('ti_infectious', float, np.nan)
        self.ti_symptomatic = ss.State('ti_symptomatic', float, np.nan)
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

        self.prob_death = ss.State('prob_death', float, self.pars['p_death'])
        self._pending_quarantine = defaultdict(list)

        return

    @property
    def exposed(self):
        return self.infected & ~self.infectious

    @property
    def _boolean_states(self):
        for state in self.states:
            if state.dtype == bool:
                yield state

    def initialize(self, sim):
        super().initialize(sim)
        self.environmental_prev = np.zeros(sim.npts)
        self.environment_concentration = np.zeros(sim.npts)

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
        self.results += ssm.MultiSimResult(self.name, 'environmental_prev', sim.npts, dtype=float)
        self.results += ssm.MultiSimResult(self.name, 'environmental_conc', sim.npts, dtype=float)

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

        # Update today's environmental prevalence
        self.calc_environmental_prev(sim)

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

        return

    def calc_environmental_prev(self, sim):
        """
        Calculate prevalence and concentration of cholera bacteria at each time step
        """
        self.environmental_prev[sim.ti] = self.pars['shedding_rate'] * self.pars['hygiene_factor'] * (np.sum(self.symptomatic) + self.pars['asymp_trans']
                                                                        * np.sum((self.infected & ~self.symptomatic))) + \
                                          self.environmental_prev[sim.ti-1] * (1 - self.pars['decay_rate'])
        self.environment_concentration[sim.ti] = self.environmental_prev[sim.ti] / (self.environmental_prev[sim.ti] + self.pars['half_sat_rate'])


    def infect(self, sim, uids, from_uids=None):
        """
            Infect new agents and determine outcomes
        """
        super().set_prognoses(sim, uids, from_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.infectious[uids] = False
        self.symptomatic[uids] = False
        self.diagnosed[uids] = False
        self.recovered[uids] = False
        self.base_immunity_inf[uids] = 0.0  # Assumption of no reinfection
        self.immunity_inf[uids] = 0.0
        self.ti_infected[uids] = sim.ti
        self.ti_pos_test[uids] = np.nan
        self.ti_diagnosed[uids] = np.nan
        self.ti_recovered[uids] = np.nan

        # determine when all infected become infectious
        self.ti_infectious[uids] = sim.ti + self.pars['dur_exp2inf'].sample(len(uids))

        # determine who becomes symptomatic and when
        symptomatic = np.random.random(len(uids)) < self.pars.prop_symp
        self.ti_symptomatic[uids[symptomatic]] = sim.ti + self.pars['dur_exp2inf'].sample(sum(symptomatic))

        # determine who dies and when
        dead = np.random.random(sum(symptomatic)) < self.pars.p_death
        dead = np.array([True if uid in uids[symptomatic][dead] else False for uid in uids])
        self.ti_dead[uids[dead]] = self.ti_symptomatic[uids[dead]] + self.pars['dur_symp2dead'].sample(sum(dead))

        # determine when agents recover
        self.ti_recovered[uids[(~dead & symptomatic)]] = self.ti_infectious[uids[(~dead & symptomatic)]] + self.pars['dur_symp2rec'].sample(sum((~dead & symptomatic)))
        self.ti_recovered[uids[~symptomatic]] = self.ti_infectious[uids[~symptomatic]] + self.pars['dur_asymp2rec'].sample(sum(~symptomatic))

        return

    def compute_trans_sus(self, rel_trans, rel_sus, pars, layer):  # pragma: no cover
        """
        Calculate relative transmissibility and susceptibility
        """
        f_iso = ~self.isolated + self.isolated * pars["iso_factor"][layer]  # Isolation factor changes e.g. [0,1] with a factor of 0.2 to [1,0.2]
        f_quar = ~self.quarantined + self.quarantined * pars["quar_factor"][layer]  # Quarantine factor changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
        f_asymp = self.symptomatic + ~self.symptomatic * pars["asymp_trans"]  # Quarantine factor changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
        rel_trans = rel_trans * self.infectious * f_quar * f_iso * f_asymp * self.immunity_trans # Recalculate transmissibility
        rel_sus = rel_sus * self.susceptible * f_quar * self.immunity_inf  # Recalculate susceptibility
        return rel_trans, rel_sus

    def test(self, uids, t, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        """
        Method to test people. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.
        """

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
        """
        Schedule a quarantine. Typically not called by the user directly except
        via a custom intervention; see the contact_tracing() intervention instead.

        This function will create a request to quarantine a person on the start_date for
        a period of time. Whether they are on an existing quarantine that gets extended, or
        whether they are no longer eligible for quarantine, will be checked when the start_date
        is reached.

        """

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
        for k, layer in sim.people.networks.items():
            if k in pars['beta_direct']:
                prel_trans = (self.infectious & sim.people.alive).astype(float)
                prel_sus = (self.susceptible & sim.people.alive).astype(float)
                rel_trans, rel_sus = self.compute_trans_sus(prel_trans, prel_sus, pars, k)
                for a, b, beta in [[layer.contacts.p1, layer.contacts.p2, pars['beta_direct'][k]],
                                   [layer.contacts.p2, layer.contacts.p1, pars['beta_direct'][k]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.contacts.beta * beta
                    new_cases = np.random.random(len(a)) < p_transmit
                    if new_cases.any():
                        self.infect(sim, b[new_cases], a[new_cases])
            if layer.name == 'randomnetwork':
                prel_trans = (self.infectious & sim.people.alive).astype(float)
                prel_sus = (self.susceptible & sim.people.alive).astype(float)
                rel_trans, rel_sus = self.compute_trans_sus(prel_trans, prel_sus, pars, k)
                p_transmit = self.environment_concentration[sim.ti] * rel_sus[self.susceptible.values] * self.pars.beta_environment_mult * self.pars.WASH_factor

                new_cases = np.random.random(sum(self.susceptible.values)) < p_transmit
                if new_cases.any():
                    susceptible = self.susceptible[new_cases.uid[new_cases.values]]
                    self.infect(sim, susceptible.uid)



    def check_uids(self, current, date, t, filter_uids=None):
        """
        Return indices for which the current state is false and which meet the date criterion.
        """
        if filter_uids is None:
            not_current = ss.false(current)
        else:
            not_current = filter_uids[np.logical_not(current[filter_uids])]
        has_date = not_current[~np.isnan(date[not_current])]
        uids = has_date[t >= date[has_date]]
        return uids

    def check_quar(self, t):
        """
        Update quarantine state
        """

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
        self.check_quar(sim.ti)

        for state in self._boolean_states:
            self.results[f'n_{state.name}'].values[sim.ti] = np.count_nonzero(state)
        self.results['prevalence'].values[sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'].values[sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
        self.results['new_diagnoses'].values[sim.ti] = np.count_nonzero(self.ti_pos_test == sim.ti)
        self.results['cum_infections'].values[sim.ti] = np.sum(self.results['new_infections'].values[:sim.ti+1])
        self.results['cum_diagnoses'].values[sim.ti] = np.sum(self.results['new_diagnoses'].values[:sim.ti+1])
        self.results['cum_deaths'].values[sim.ti] = np.sum(self.results['new_deaths'].values[:sim.ti+1])
        self.results['environmental_prev'].values[sim.ti] = self.environmental_prev[sim.ti]
        self.results['environmental_conc'].values[sim.ti] = self.environment_concentration[sim.ti]

    def finalize_results(self, sim):
        pass


class CholeraVaccine:
    def __init__(self, name, immunity_timecourse, protection_timecourse, prevent_infection, prevent_transmission,
                 prevent_symp, prevent_death):
        """
        Vaccine method for cholera.
        Args:
            characteristics:  Vaccine characteristic dictionary - containing poi, pos, poh, poc, pod
            rising_immunity:
        """

        self.name = name
        self.disease = 'cholera'
        self.prevent_infection = prevent_infection  # Prevention of infection
        self.prevent_transmission = prevent_transmission
        self.prevent_symp = prevent_symp  # Prevention of symptoms
        self.prevent_death = prevent_death  # Prevention of death
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
    def cholera_vacc(cls):
        # Cholera parameters, parametrized by dose interval

        # estimates of effectiveness are highly variable, for both single and double dose, additionally protection seems to vary with age https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8708586/
        time_to_peak = 10  # Immunity after second dose reaches its peak after 7-10 days from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8708586/

        immunity_timecourse = [(0, 0), (time_to_peak, 1)] # dose schedule of two weeks
        protection_timecourse = [(0, 0), (time_to_peak, 1)]

        vaccine_characteristics = {
            "prevent_infection": 0.527,  # Malembaka et al. estimate of single dose effectiveness (https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(23)00742-9/fulltext)
            "prevent_transmission": 0,  # no data
            "prevent_symp": 0,  # no data
            "prevent_death": 0.5, # Song et al. proxy for single dose protection against severe disease (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8708586/)
        }
        return cls(name="cholera_vacc", immunity_timecourse=immunity_timecourse,
                   protection_timecourse=protection_timecourse, **vaccine_characteristics)
