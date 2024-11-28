import starsim as ss
import sciris as sc
import numpy as np
from collections import defaultdict
import gavi.utils as ssg
from gavi import multisim as ssm


__all__ = ['EventSchedule', 'DynamicTrigger', 'test_prob_quarantine', 'contact_tracing', 'TimedVaccinationProgram', 'get_vaccine_sequence_grouped']


class EventSchedule(ss.Intervention):
    """
    Run functions on different days

    iv = EventSchedule()
    iv[1] = lambda sim: print(sim.t)
    iv['2020-04-02'] = lambda sim: print('foo')

    """

    def __init__(self):
        super().__init__()
        self.schedule = defaultdict(list)

    def __getitem__(self, day):
        return self.schedule[day]

    def __setitem__(self, day, fcn):
        if day in self.schedule:
            raise Exception("Use a list instead to assign multiple functions - or to really overwrite, delete the function for this day first i.e. `del schedule[day]` before performing `schedule[day]=...`")
        self.schedule[day] = fcn

    def __delitem__(self, key):
        del self.schedule[key]

    def initialize(self, sim):
        super().initialize(sim)

        # Check that there are no other intervention types before this one
        # Otherwise, the schedule's actions might be delayed by a timestep
        for iv in sim.pars["interventions"]:
            if iv is self:
                break
            elif not isinstance(iv, EventSchedule):
                raise Exception(f"{self} appears after other intervention types - all schedules should appear at the start")

    def apply(self, sim):
        if sim.ti in self.schedule:
            if isinstance(self.schedule[sim.ti], list):
                for fcn in self.schedule[sim.ti]:
                    fcn(sim)
            else:
                self.schedule[sim.ti](sim)

class DynamicTrigger(ss.Intervention):
    """
    Execute callback during simulation execution
    """

    def __init__(self, condition, action, once_only=False, **kwargs):
        """
        Args:
            condition: A function `condition(sim)` function that returns True or False
            action: A function `action(sim)` that runs if the condition was true
            once_only: If True, the action will only execute once
        """
        super().__init__(**kwargs)
        self.condition = condition  #: Function that
        self.action = action
        self.once_only = once_only
        self._ran = False

    def apply(self, sim):
        """
        Check condition and execute callback
        """
        if not (self.once_only and self._ran) and self.condition(sim):
            self.action(sim)
            self._ran = True


class test_prob_quarantine(ss.Intervention):
    """
    Probability-based testing with quarantine

    This adds quarantine while testing, with a configurable compliance
    (which affects whether people enter quarantine or not)
    """

    def __init__(self, symp_prob, symp_quar_prob, sensitivity, quarantine_compliance, disease, *args, test_delay_mean=None, vac_symp_prob=np.nan, asymp_prob=np.nan, exclude=None, test_delay=None, **kwargs):
        """
        Args:
            symp_prob:
            symp_quar_prob:
            sensitivity:
            test_delay_mean: Mean test delay - note that the minimum test delay is 1, so test_delay_mean=0 will result in all tests taking exactly 1 day to be returned
            test_delay: If specified, don't sample from the test delay
            quarantine_compliance:
            *args:
            vac_symp_prob: If provided, specify an ABSOLUTE testing rate for symptomatic vaccinated people. If nan, they will test at the same rate as the unvaccinated people
            exclude: If provided, specify a list/array of indices of people that should not be tested via this intervention
            **kwargs:
        """

        super().__init__(*args, **kwargs)

        assert (test_delay_mean is None) != (test_delay is None), "Either the mean test delay or the absolute test delay must be specified"
        self.results = ss.ndict(type=ssm.MultiSimResult)

        if asymp_prob == np.nan:
            self.asymp_prob = 0
        else:
            self.asymp_prob = asymp_prob
        self.symp_prob = symp_prob
        self.symp_quar_prob = symp_quar_prob
        if not isinstance(sensitivity, dict):
            self.sensitivity = {"symptomatic": sensitivity}
        else:
            self.sensitivity = sensitivity
        self.disease = disease
        self.test_delay_mean = test_delay_mean
        self.test_delay = test_delay
        self.quarantine_compliance = quarantine_compliance  #: Compliance level for individuals in general population isolating after testing. People already in quarantine are assumed to be compliant
        self.vac_symp_prob = vac_symp_prob
        self.test_probs = ss.State('test_prob', float, 0.0)
        self.delays = ss.State('delay', float, np.nan)

        self.n_tests = None
        self.n_positive = None  # Record how many tests were performed that will come back positive
        self.exclude = exclude  # Exclude certain people - mainly to cater for simulations where the index case/incursion should not be diagnosed

        self._scheduled_tests = defaultdict(list)

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ssm.MultiSimResult(self.name, 'new_tests', sim.npts, dtype=float)
        self.n_tests = np.zeros(sim.npts)
        self.n_positive = np.zeros(sim.npts)
        self.test_probs.initialize(sim.people)
        self.delays.initialize(sim.people)

    def schedule_test(self, sim, uids, t: int):
        """
        Schedule a test in the future

        If the test is requested today, then test immediately. This is because testing should be run prior to quarantine so that
        quarantine can take place on the day of diagnosis even if the test_delay is 0.

        :param uids: Iterable with person indices to test
        :param t: Simulation day on which to test them
        :return:
        """

        if t == sim.ti:
            # If a person is scheduled to test on the same day (e.g., if they are a household contact and get tested on
            # the same day they are notified)

            not_dead_diag = sim.diseases[self.disease].diagnosed | sim.people.dead
            uids = uids[np.logical_not(not_dead_diag[uids])]  # Only test people that haven't been diagnosed and are alive
            self._test(sim, uids)
        else:
            self._scheduled_tests[t] += uids.tolist()

    def _test(self, sim, test_uids):
        # After testing (via self.apply or self.schedule_test) perform some post-testing tasks
        # test_uids are the indices of the people that were requested to be tested (i.e. that were
        # passed into sim.people.test, so a test was performed on them
        #
        # CAUTION - this method gets called via both apply() and schedule_test(), therefore it can be
        # called multiple times per timestep, quantities must be incremented rather than overwritten
        if len(test_uids) == 0:
            return

        symp_test_uids = test_uids[sim.diseases[self.disease].symptomatic[test_uids]]
        other_test_uids = test_uids[~sim.diseases[self.disease].symptomatic[test_uids]]

        if len(symp_test_uids):
            sim.diseases[self.disease].test(symp_test_uids, sim.ti, test_sensitivity=self.sensitivity['symptomatic'], loss_prob=0, test_delay=np.inf)  # Actually test people with mild symptoms
        if len(other_test_uids):
            sim.diseases[self.disease].test(other_test_uids, sim.ti, test_sensitivity=self.sensitivity['symptomatic'], loss_prob=0, test_delay=np.inf)  # Actually test people without symptoms

        if self.test_delay is not None:
            self.delays[test_uids] = self.test_delay
        else:
            self.delays[test_uids] = np.maximum(1, ss.n_poisson(self.test_delay_mean, len(test_uids)))

        # Update the date diagnosed
        positive_today = ss.true(sim.diseases[self.disease].ti_pos_test[test_uids] == sim.ti)

        sim.diseases[self.disease].ti_diagnosed[positive_today] = sim.ti + self.delays[positive_today]

        # Quarantine while waiting
        if self.quarantine_compliance and len(test_uids):
            # If people are meant to quarantine while waiting for their test, then quarantine some/all of the people waiting for tests
            if self.quarantine_compliance == 1:
                # If fully compliant, keep all indices straight away
                quar_uids = test_uids
                quar_delay = self.delays[test_uids]
            else:
                # Otherwise, filter by quarantine compliance
                to_quarantine = ss.n_binomial(self.quarantine_compliance, len(test_uids))  # Boolean array of test_uids to quarantine
                quar_uids = test_uids[to_quarantine]
                quar_delay = self.delays[quar_uids]  # Array of associated delays

            # Then iterate over delays, and schedule quarantine for each delay
            for delay in set(quar_delay):
                match_delay = quar_delay == delay  # Indices of quar with people with the specified delay
                sim.diseases[self.disease].schedule_quarantine(quar_uids[match_delay], sim.ti, period=delay)

        # Logging
        self.n_positive[sim.ti] = len(positive_today)  # Record how many people were tested by this program today, that ended up testing positive

        # For the purpose of counting tests, people in quarantine only count as one person (notwithstanding rescaling)
        # Otherwise, scale up by the pop_scale. Since we model general tests being distributed throughout the population, we assume
        # symptomatic and asymptomatic tests both get applied to people outside of the model, and need to account for the population scale
        # accordingly. However, we also assume that the entire epidemic is contained within the pool of agents, therefore
        tests_in_quarantine = sim.diseases[self.disease].quarantined[test_uids].sum()
        tests_not_in_quarantine = len(test_uids) - tests_in_quarantine

        # Store tests performed by this intervention
        n_tests = tests_in_quarantine + tests_not_in_quarantine * sim.pars["pop_scale"]
        self.n_tests[sim.ti] += n_tests  # Store tests performed by this intervention
        self.results["new_tests"].values[sim.ti] = n_tests  # Update total test count

    def select_people(self, sim):
        # First, insert any fixed test probabilities
        self.test_probs.values = np.ones(len(sim.people)) * self.asymp_prob
        self.test_probs[sim.diseases[self.disease].symptomatic] = self.symp_prob  # Symptomatic people test at a higher rate
        self.test_probs[sim.diseases[self.disease].symptomatic & sim.diseases[self.disease].vaccinated] = self.vac_symp_prob  # Symptomatic and vaccinated people test at a different (usually lower) rate
        self.test_probs[sim.diseases[self.disease].symptomatic & sim.diseases[self.disease].quarantined] = self.symp_quar_prob  # Symptomatic and quarantined people test at a different rate - note that this takes priority over vaccinated
        self.test_probs[~sim.diseases[self.disease].symptomatic & sim.diseases[self.disease].quarantined] = 0
        if hasattr(sim.diseases[self.disease], 'severe'):  # No severe state for measle
            self.test_probs[sim.diseases[self.disease].severe] = np.max([0.25, self.symp_prob]) # assumption
        if self.exclude is not None:
            self.test_probs[self.exclude] = 0  # If someone is excluded, then they shouldn't test via `apply()` (but can still test via a scheduled test)
        if sim.pars.remove_dead and len(self._scheduled_tests[sim.ti]) > 0:
            self.clean_uid(sim)
        self.test_probs[self._scheduled_tests[sim.ti]] = 1  # People scheduled to test (e.g. via contact tracing) are guaranteed to test
        self.test_probs[sim.diseases[self.disease].diagnosed] = 0  # People already diagnosed don't test again
        self.test_probs[sim.diseases[self.disease].dead] = 0  # Dead people don't get tested
        test_uids = ss.true(ss.binomial_arr(self.test_probs))  # Finally, calculate who actually tests
        return test_uids

    def apply(self, sim):

        test_uids = self.select_people(sim)
        self._test(sim, test_uids)

    def clean_uid(self, sim):
        "Removes uids of dead agents if simulation is removing them"
        self._scheduled_tests[sim.ti] = [uid for uid in self._scheduled_tests[sim.ti] if uid in sim.people.uid]


class contact_tracing(ss.Intervention):
    '''
    Contact tracing of people who are diagnosed. When a person is diagnosed positive
    (by either test_num() or test_prob(); this intervention has no effect if there
    is not also a testing intervention active), a certain proportion of the index
    case's contacts (defined by trace_prob) are contacted after a certain number
    of days (defined by trace_time). After they are contacted, they are placed
    into quarantine (with effectiveness quar_factor, a simulation parameter) for
    a certain period (defined by quar_period, another simulation parameter). They
    may also change their testing probability, if test_prob() is defined.

    Args:
        trace_probs (float/dict): probability of tracing, per layer (default: 100%, i.e. everyone is traced)
        trace_time  (float/dict): days required to trace, per layer (default: 0, i.e. no delay)
        start_day   (int):        intervention start day (default: 0, i.e. the start of the simulation)
        end_day     (int):        intervention end day (default: no end)
        presumptive (bool):       whether or not to begin isolation and contact tracing on the presumption of a positive diagnosis (default: no)
        capacity    (int):        optionally specify a maximum number of newly diagnosed people to trace each day
        quar_period (int):        number of days to quarantine when notified as a known contact. Default value is ``pars['quar_period']``
        kwargs      (dict):       passed to Intervention()

    **Example**::

        tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.01)
        ct = cv.contact_tracing(trace_probs=0.5, trace_time=2)
        sim = cv.Sim(interventions=[tp, ct]) # Note that without testing, contact tracing has no effect
    '''
    def __init__(self, disease, trace_probs, trace_time, quar_period, test_schedule, testing_intervention, start_day=0, end_day=None, presumptive=False, capacity=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.disease     = disease
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        self.capacity = capacity
        self.quar_period = quar_period # If quar_period is None, it will be drawn from sim.pars at initialization
        self.test_schedule = test_schedule
        self.testing_intervention = testing_intervention
        self.results = ss.ndict(type=ssm.MultiSimResult)

        self.test_on_notification = ss.State("test_on_notification", bool, False)

        return


    def initialize(self, sim):
        ''' Process the dates and dictionaries '''
        super().initialize(sim)
        self.days      = [self.start_day, self.end_day]
        if self.trace_probs is None:
            self.trace_probs = 1.0
        if self.trace_time is None:
            self.trace_time = 0.0
        if self.quar_period is None:
            self.quar_period = sim.pars['quar_period']
        if sc.isnumber(self.trace_probs):
            val = self.trace_probs
            self.trace_probs = {k:val for k in sim.people.networks.items()}
        if sc.isnumber(self.trace_time):
            val = self.trace_time
            self.trace_time = {k:val for k in sim.people.networks.items()}
        self.new_notifications = np.zeros(sim.npts)
        return

    def apply(self, sim):
        '''
        Trace and notify contacts

        Tracing involves three steps that can independently be overloaded or extended
        by derived classes

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        '''
        t = sim.ti
        start_day = self.start_day
        end_day   = self.end_day
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        trace_uids = self.select_cases(sim)
        contacts = self.identify_contacts(sim, trace_uids)
        self.notify_contacts(sim, contacts)
        return contacts


    def select_cases(self, sim):
        '''
        Return people to be traced at this time step
        '''
        if not self.presumptive:
            uids = ss.true(sim.diseases[self.disease].ti_diagnosed == sim.ti) # Diagnosed this time step, time to trace
        else:
            just_tested = ss.true(sim.diseases[self.disease].ti_tested == sim.ti) # Tested this time step, time to trace
            uids = just_tested[sim.diseases[self.disease].exposed[just_tested]] # This is necessary to avoid infinite chains of asymptomatic testing

        # If there is a tracing capacity constraint, limit the number of agents that can be traced
        if self.capacity is not None:
            capacity = int(self.capacity)
            if len(uids) > capacity:
                uids = np.random.choice(uids, capacity, replace=False)

        return uids


    def identify_contacts(self, sim, trace_uids):
        '''
        Return contacts to notify by trace time

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_uids: Indices of people to trace

        Returns: {trace_time: np.array(uids)} dictionary storing which people to notify
        '''

        if not len(trace_uids):
            return {}

        contacts = sc.ddict(list)

        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            traceable_uids = sim.people.networks[lkey].find_contacts(trace_uids)
            if len(traceable_uids):
                contacts[self.trace_time[lkey]].extend(ss.binomial_filter(this_trace_prob, traceable_uids)) # Filter the indices according to the probability of being able to trace this layer

        array_contacts = {}
        for trace_time, uids in contacts.items():
            array_contacts[trace_time] = np.fromiter(uids, dtype=ss.int_)

        return array_contacts


    def notify_contacts(self, sim, contacts):
        '''
        Notify contacts

        This method represents notifying people that they have had contact with a confirmed case.
        In this base class, that involves

        - Setting the 'known_contact' flag and recording the 'ti_known_contact'
        - Scheduling quarantine

        Args:
            sim: Simulation object
            contacts: {trace_time: np.array(uids)} dictionary storing which people to notify
        '''
        is_dead = ss.true(sim.people.dead) # Find people who are not alive
        for trace_time, contact_uids in contacts.items():
            contact_uids = np.setdiff1d(contact_uids, is_dead) # Do not notify contacts who are dead
            sim.diseases[self.disease].known_contact[contact_uids] = True
            sim.diseases[self.disease].ti_known_contact[contact_uids] = np.fmin(sim.diseases[self.disease].ti_known_contact[contact_uids], sim.ti + trace_time)
            sim.diseases[self.disease].schedule_quarantine(contact_uids, sim.ti, start_date=sim.ti + trace_time, period=self.quar_period - trace_time)  # Schedule quarantine for the notified people to start on the date they will be notified

        test_on_notification = sc.dcp(self.test_on_notification)  # Record whether an overdue test has already been performed (handle edge case where more than one test is overdue)

        for trace_time, contact_uids in contacts.items():

            for test_day in self.test_schedule:
                if (sim.ti + trace_time) > (sim.ti + test_day):
                    # If the scheduled test is overdue, perform the test immediately and record that this was done
                    contact_uids = contact_uids[~test_on_notification[contact_uids]]  # Only schedule one overdue test
                    self.testing_intervention.schedule_test(sim, contact_uids, sim.ti + trace_time)
                    test_on_notification[contact_uids] = True
                else:
                    self.testing_intervention.schedule_test(sim, contact_uids, sim.ti + test_day)

        self.new_notifications[sim.ti] = 0
        for contact_inds in contacts.values():
            self.new_notifications[sim.ti] += len(contact_inds)
        return


class TimedVaccinationProgram_Measles(ss.Intervention):
    # This intervention models people receiving a vaccine with immunity that builds over time

    #leaky = True  # Flag for leaky vs non-leaky vaccines (applies to all vaccination programs)

    def __init__(self, vaccine, sequence=None, num_doses=0, dynamic_sequence=False, *args, **kwargs):
        """

        Args:
            vaccine: A ``Vaccine`` instance (defined above)
            sequence:
            num_doses: - A scalar, a callable `fcn(sim)` or an array the same size as sim.tvec

        """
        super().__init__(*args, **kwargs)
        self.sequence = sequence  # Specify vaccine sequence, None means random order for everyone. Otherwise, an array or a callable
        self.num_doses = num_doses  # Specify number of doses as scalar, dict (by date or day), or callable function
        self.vaccine = vaccine  # e.g. `rising_immunity_pfizer_3w` - should be sorted
        self.n_people_vaccinated = None
        self.n_agents_vaccinated = None
        self.dynamic_sequence = dynamic_sequence
        self._vaccinated = ss.State('_vaccinated', bool, False)  # True if someone was vaccinated using THIS vaccine
        self._ti_immune = ss.State('_ti_vaccinated', float, np.nan)  # Track date people became immune due to this intervention
        self._pending_immunity = ss.State('pending_immunity', bool, False)  # Boolean flag for whether people are immune or not
        if 'second_dose_peak_coverage' in kwargs:
            self.second_dose_peak_coverage = kwargs['second_dose_peak_coverage']
            self.second_dose_peak_coverage_rel2_first = kwargs['second_dose_peak_coverage_rel2_first']
        else:
            self.second_dose_peak_coverage_rel2_first = None
            self.second_dose_peak_coverage = None

        self.infants_immune_age = kwargs["infants_immune_age"]
    def initialize(self, sim=None):
        super().initialize(sim)

        self._vaccinated.initialize(sim.people)
        self._ti_immune.initialize(sim.people)
        self._pending_immunity.initialize(sim.people)
        self.n_people_vaccinated = np.zeros(sim.npts)
        self.n_agents_vaccinated = np.zeros(sim.npts)
        self.results = ss.ndict(type=ssm.MultiSimResult)
        self.results += ssm.MultiSimResult(self.name, 'vac_doses', sim.npts, dtype=float)

        # Convert any dates to simulation days
        if isinstance(self.num_doses, dict):
            self.num_doses = {sim.day(k): v for k, v in self.num_doses.items()}

        # Convert the vaccine sequence into an array
        if callable(self.sequence):
            self.sequence = self.sequence(sim.people)
        elif self.sequence is None:
            self.sequence = np.random.permutation(sim.pars['n_agents'])
        else:
            self.sequence = sc.promotetoarray(self.sequence)

        # Update people vaccinated at the start:
        self._vaccinated[sim.diseases[0].fully_vaccinated] = True
        self.n_people_vaccinated[0] = sum(self._vaccinated)
        self.n_agents_vaccinated[0] = sum(self._vaccinated)
        # Set to maximum protection time for maximum immunity
        sim.diseases[self.vaccine.disease].ti_vaccinated[self._vaccinated] = - self.vaccine.full_protection_time

        self._immunity_timecourse = self.vaccine.immunity_timecourse(np.arange(0, self.vaccine.full_protection_time + 1))  # Cache the immunity function
        self._protection_timecourse = self.vaccine.protection_timecourse(np.arange(0, self.vaccine.full_protection_time + 1))  # Cache the immunity function

    # At the start, we want to vaccinate a bunch of people, and start them out with a level of prior immunity
    def update_immunity(self, sim, t):

        # Measles: Update immunity for <X months olds
        if sim.diseases[0].name == 'measles':
            infants_immune = sim.people.uid[sim.people.age < self.infants_immune_age/12]
            immunity = 1 - (1/(self.infants_immune_age*30)) * (sim.people.age[infants_immune]*365 + t)
            protection = 1 - (1/(self.infants_immune_age*30)) * (sim.people.age[infants_immune]*365 + t)
            sim.diseases[self.vaccine.disease].immunity_inf[infants_immune] = sim.diseases[self.vaccine.disease].base_immunity_inf[infants_immune] * (1 - 1 * immunity)
            sim.diseases[self.vaccine.disease].immunity_trans[infants_immune] = sim.diseases[self.vaccine.disease].base_immunity_trans[infants_immune] * (1 - 0 * protection)

        # For the remaining vaccine characteristics, scale the outcome by proportion of protection
        vaccinated = ss.true(self._vaccinated)  # Indices of people that were vaccinated using this intervention
        ti_vaccinated = sim.diseases[self.vaccine.disease].ti_vaccinated[vaccinated]  # Vaccination date for people vaccinated using this intervention
        duration_since_vaccinated = sim.ti - ti_vaccinated

        duration_since_vaccinated = np.minimum(duration_since_vaccinated, len(self._protection_timecourse) - 1).astype(ss.int_)  # Max out protection
        assert not np.any(duration_since_vaccinated < 0)  # Cannot have negative durations, can disable this check for performance if required

        # Update fully vaccinated status for anyone that has received their second dose (if applicable)
        if self.vaccine.dose_interval is not None:
            ready2_gain_fully_vaccinated = ss.true((~sim.diseases[self.vaccine.disease].fully_vaccinated[vaccinated]) & (duration_since_vaccinated == self.vaccine.dose_interval))
            # Ensure that we don't go over the second vax coverage (percentage of eligible people)
            if sum(sim.diseases[self.vaccine.disease].fully_vaccinated) <= ((self.second_dose_peak_coverage_rel2_first/ 100) * len(self.sequence)):
                gain_fully_vaccinated = ss.binomial_filter(self.second_dose_peak_coverage_rel2_first / 100, ready2_gain_fully_vaccinated)
                sim.diseases[self.vaccine.disease].fully_vaccinated[gain_fully_vaccinated] = True
                # Add to total number of vacc doses given for that day
                self.results["vac_doses"].values[sim.ti] = np.cumsum(self.results["vac_doses"].values[sim.ti]) + len(gain_fully_vaccinated)

        # Update protection for today
        immunity = self._immunity_timecourse[duration_since_vaccinated]
        protection = self._protection_timecourse[duration_since_vaccinated]
        sim.diseases[self.vaccine.disease].immunity_inf[vaccinated] = sim.diseases[self.vaccine.disease].base_immunity_inf[vaccinated] * (1 - self.vaccine.prevent_infection * immunity)
        sim.diseases[self.vaccine.disease].immunity_trans[vaccinated] = sim.diseases[self.vaccine.disease].base_immunity_trans[vaccinated] * (1 - self.vaccine.prevent_transmission * protection)
        if hasattr(sim.diseases[self.vaccine.disease], 'base_immunity_symp'):
            sim.diseases[self.vaccine.disease].immunity_symp[vaccinated] = sim.diseases[self.vaccine.disease].base_immunity_symp[vaccinated] * (1 - self.vaccine.prevent_symp * protection)
        if hasattr(sim.diseases[self.vaccine.disease], 'prob_sev'): # No severe for measles
            sim.diseases[self.vaccine.disease].prob_sev[vaccinated] = sim.diseases[self.vaccine.disease].prob_sev[vaccinated] * (1 - self.vaccine.prevent_severe * protection)
        if hasattr(self.vaccine, 'clear_carriage'): # Clear asymptomatic carriage for meningitis
            asymp_vaccinated = np.array([uid for uid in vaccinated if (~sim.diseases[self.vaccine.disease].symptomatic[uid] & sim.diseases[self.vaccine.disease].infectious[uid] & (sim.diseases[self.vaccine.disease].ti_vaccinated[uid]==sim.ti))], dtype=ss.int_)
            carriage_cleared = np.random.random(len(asymp_vaccinated)) < self.vaccine.clear_carriage
            sim.diseases[self.vaccine.disease].ti_recovered[asymp_vaccinated[carriage_cleared]] = sim.ti + 1
        sim.diseases[self.vaccine.disease].prob_death[vaccinated] = sim.diseases[self.vaccine.disease].prob_death[vaccinated] * (1 - self.vaccine.prevent_death * protection)

    @staticmethod
    def _get_ti_immune(uids, n_immune):
        """
        Args:
            uids: array of person indices
            n_immune: array of how many people should be immune for each day after vaccination (the first day is 0)
                      The length of this array is arbitrary, the length of this list defines the maximum value present
                      in the output array
        Returns: A list the same length as `uids`
        """

        n_gain_immunity = np.diff(n_immune)  # Number of people that gain immunity each day
        day_immune = np.zeros(uids.shape, dtype=ss.int_)

        count = 0
        for i in range(0, len(n_gain_immunity)):
            n_contacts = n_gain_immunity[i]
            day_immune[count : count + n_contacts] = i + 1  # The first entry (0th) in n_gain_immunity corresponds to gaining immunity 1 day afterwards
            count += n_contacts

            if count == len(day_immune):
                break

        return np.random.permutation(day_immune)  # Shuffle the order in which people gain immunity

    def vaccinate(self, sim, uids, t=None, update_immunity=True):
        """
        Use this function to vaccinate a group of people

        Args:
            sim:
            uids:
            t: Override vaccination date relative to simulation date (e.g. for historical vaccination). Can be a day index or a date

        Returns:

        """

        if not sc.isnumber(t):
            t = sim.ti

        # Validate indices of people to vaccinate - *essential* that we don't vaccinate anyone using this
        # intervention that has already been vaccinated, otherwise this intervention will interact unintentionally
        # with any other interventions that vaccinated that same person
        assert not np.any(sim.diseases[self.vaccine.disease].vaccinated[uids] | sim.diseases[self.vaccine.disease].dead[uids]), "Cannot re-vaccinate people with this intervention"  # nb. can disable this check for performance later on if needed

        # logger.debug(f'{self.label}: Vaccinating {len(uids)} agents at {t=}')
        sim.diseases[self.vaccine.disease].vaccinated[uids] = True
        if self.vaccine.dose_interval is None:
            # If it's a single dose vaccine, they are fully vaccinated after the first dose
            sim.diseases[self.vaccine.disease].fully_vaccinated[uids] = True
        self._vaccinated[uids] = True  # Record that they have been vaccinated by this intervention
        sim.diseases[self.vaccine.disease].ti_vaccinated[uids] = t

        if t >= 0:
            # Record the new vaccinations etc. if they are taking place during the simulation timeframe
            # We assume that the vaccination rollout continues outside the pool of agents being modelled here.
            # Following from the example in the `apply` method, if we vaccinate 10 agents, this corresponds to
            # vaccinating 5 people within the area being modelled i.e. 10*5. However, the area being modelled
            # accounts for a (5/10) fraction of the total population. Therefore the number of people vaccinated
            # is in fact 10*5/(5/10). As before, the factors cancel out, and we actually just multiply
            # by the pop_scale here
            new_vaccinated = int(len(uids) * sim.pars["pop_scale"])
            self.n_people_vaccinated[t] += new_vaccinated
            self.n_agents_vaccinated[t] += int(len(uids))
            self.results["vac_doses"].values[sim.ti] = np.cumsum(self.results["vac_doses"].values[sim.ti]) + new_vaccinated  # Update total test count

        if update_immunity:
            self.update_immunity(sim, t=t)

    def apply(self, sim, t=None, num_people=None):
        """
        Use this function to vaccinate a number of people, selected based on the stored sequence
        """

        if not sc.isnumber(t):
            t = sim.ti

        # Work out how many *people* to vaccinate today - matches reported numbers of doses for a jurisdiction
        if num_people is None:
            if sc.isnumber(self.num_doses):
                num_people = self.num_doses
            elif callable(self.num_doses):
                num_people = self.num_doses(sim)
            elif t in self.num_doses:
                num_people = self.num_doses[t]
            else:
                num_people = 0

        # Suppose we have a pop_scale of 10, and a current scale factor of 5, with 100 agents.
        # That means that we have a total population of 1000 people, and currently 1 person
        # represents 5 people. If we want to vaccinate 100 people in the population, how many
        # agents does this correspond to? Since we are currently modelling 500 people with 100
        # agents, assuming the 100 vaccines are uniformly distributed, we need to allocate
        # n = 100*(5/10) = 50 vaccines to the pool of people being modelled. Further, we then
        # need to divide this number of people by the current scale factor to get the number of
        # agents corresponding to 50 people. That is, 50/5=10 agents. Since the overall calculation is
        # (100*5/10)/5 this is just equivalent to dividing by the overall pop scale.
        num_agents = int(np.round(num_people / sim.pars["pop_scale"]))

        if self.dynamic_sequence:
            self.sequence = get_vaccine_sequence_grouped(sim.people, sim.diseases[self.vaccine.disease], "18+", eligible=None, priority=True)

        if num_agents and len(self.sequence):
            # People are ineligible for vaccination if they are already vaccinated or if they are dead
            # However, people who have died and are then returned to the simulation by rescaling still need
            # to be eligible for vaccination. Therefore we don't actually remove them from the schedule since they
            # could come back to life at any time in the simulation
            eligible = self.sequence[~sim.diseases[self.vaccine.disease].vaccinated[self.sequence] & ~sim.diseases[self.vaccine.disease].dead[self.sequence]]
            uids = eligible[:num_agents]  # nb. this indexing
        else:
            uids = np.array([], dtype=ss.int_)

        # Vaccinate them
        if len(uids):
            # Check uids at this point, in case num_agents > 0 but nobody was eligible and thus no vaccinations were performed
            self.vaccinate(sim, uids, t=t)
        else:
            self.update_immunity(sim, t=t)

        return uids

class TimedVaccinationProgram(ss.Intervention):
    # This intervention models people receiving a vaccine with immunity that builds over time

    #leaky = True  # Flag for leaky vs non-leaky vaccines (applies to all vaccination programs)

    def __init__(self, vaccine, sequence=None, num_doses=0, dynamic_sequence=False, *args, **kwargs):
        """

        Args:
            vaccine: A ``Vaccine`` instance (defined above)
            sequence:
            num_doses: - A scalar, a callable `fcn(sim)` or an array the same size as sim.tvec

        """
        super().__init__(*args, **kwargs)
        self.sequence = sequence  # Specify vaccine sequence, None means random order for everyone. Otherwise, an array or a callable
        self.num_doses = num_doses  # Specify number of doses as scalar, dict (by date or day), or callable function
        self.vaccine = vaccine  # e.g. `rising_immunity_pfizer_3w` - should be sorted
        self.n_people_vaccinated = None
        self.n_agents_vaccinated = None
        self.dynamic_sequence = dynamic_sequence
        self._vaccinated = ss.State('_vaccinated', bool, False)  # True if someone was vaccinated using THIS vaccine
        self._ti_immune = ss.State('_ti_vaccinated', float, np.nan)  # Track date people became immune due to this intervention
        self._pending_immunity = ss.State('pending_immunity', bool, False)  # Boolean flag for whether people are immune or not
        if 'second_dose_peak_coverage' in kwargs:
            self.second_dose_peak_coverage = kwargs['second_dose_peak_coverage']
            self.second_dose_peak_coverage_rel2_first = kwargs['second_dose_peak_coverage_rel2_first']
        else:
            self.second_dose_peak_coverage_rel2_first = None
            self.second_dose_peak_coverage = None
    def initialize(self, sim=None):
        super().initialize(sim)

        self._vaccinated.initialize(sim.people)
        self._ti_immune.initialize(sim.people)
        self._pending_immunity.initialize(sim.people)
        self.n_people_vaccinated = np.zeros(sim.npts)
        self.n_agents_vaccinated = np.zeros(sim.npts)
        self.results = ss.ndict(type=ssm.MultiSimResult)
        self.results += ssm.MultiSimResult(self.name, 'vac_doses', sim.npts, dtype=float)

        # Convert any dates to simulation days
        if isinstance(self.num_doses, dict):
            self.num_doses = {sim.day(k): v for k, v in self.num_doses.items()}

        # Convert the vaccine sequence into an array
        if callable(self.sequence):
            self.sequence = self.sequence(sim.people)
        elif self.sequence is None:
            self.sequence = np.random.permutation(sim.pars['n_agents'])
        else:
            self.sequence = sc.promotetoarray(self.sequence)

        # Update people vaccinated at the start:
        self._vaccinated[sim.diseases[0].fully_vaccinated] = True
        self.n_people_vaccinated[0] = sum(self._vaccinated)
        self.n_agents_vaccinated[0] = sum(self._vaccinated)
        # Set to maximum protection time for maximum immunity
        sim.diseases[self.vaccine.disease].ti_vaccinated[self._vaccinated] = - self.vaccine.full_protection_time

        self._immunity_timecourse = self.vaccine.immunity_timecourse(np.arange(0, self.vaccine.full_protection_time + 1))  # Cache the immunity function
        self._protection_timecourse = self.vaccine.protection_timecourse(np.arange(0, self.vaccine.full_protection_time + 1))  # Cache the immunity function

    # At the start, we want to vaccinate a bunch of people, and start them out with a level of prior immunity
    def update_immunity(self, sim, t):

        # For the remaining vaccine characteristics, scale the outcome by proportion of protection
        vaccinated = ss.true(self._vaccinated)  # Indices of people that were vaccinated using this intervention
        ti_vaccinated = sim.diseases[self.vaccine.disease].ti_vaccinated[vaccinated]  # Vaccination date for people vaccinated using this intervention
        duration_since_vaccinated = sim.ti - ti_vaccinated

        duration_since_vaccinated = np.minimum(duration_since_vaccinated, len(self._protection_timecourse) - 1).astype(ss.int_)  # Max out protection
        assert not np.any(duration_since_vaccinated < 0)  # Cannot have negative durations, can disable this check for performance if required

        # Update fully vaccinated status for anyone that has received their second dose (if applicable)
        if self.vaccine.dose_interval is not None:
            ready2_gain_fully_vaccinated = ss.true((~sim.diseases[self.vaccine.disease].fully_vaccinated[vaccinated]) & (duration_since_vaccinated == self.vaccine.dose_interval))
            # Ensure that we don't go over the second vax coverage (percentage of eligible people)
            if sum(sim.diseases[self.vaccine.disease].fully_vaccinated) <= ((self.second_dose_peak_coverage_rel2_first/ 100) * len(self.sequence)):
                gain_fully_vaccinated = ss.binomial_filter(self.second_dose_peak_coverage_rel2_first / 100, ready2_gain_fully_vaccinated)
                sim.diseases[self.vaccine.disease].fully_vaccinated[gain_fully_vaccinated] = True
                # Add to total number of vacc doses given for that day
                self.results["vac_doses"].values[sim.ti] = np.cumsum(self.results["vac_doses"].values[sim.ti]) + len(gain_fully_vaccinated)

        # Update protection for today
        immunity = self._immunity_timecourse[duration_since_vaccinated]
        protection = self._protection_timecourse[duration_since_vaccinated]
        sim.diseases[self.vaccine.disease].immunity_inf[vaccinated] = sim.diseases[self.vaccine.disease].base_immunity_inf[vaccinated] * (1 - self.vaccine.prevent_infection * immunity)
        sim.diseases[self.vaccine.disease].immunity_trans[vaccinated] = sim.diseases[self.vaccine.disease].base_immunity_trans[vaccinated] * (1 - self.vaccine.prevent_transmission * protection)
        if hasattr(sim.diseases[self.vaccine.disease], 'base_immunity_symp'):
            sim.diseases[self.vaccine.disease].immunity_symp[vaccinated] = sim.diseases[self.vaccine.disease].base_immunity_symp[vaccinated] * (1 - self.vaccine.prevent_symp * protection)
        if hasattr(sim.diseases[self.vaccine.disease], 'prob_sev'): # No severe for measles
            sim.diseases[self.vaccine.disease].prob_sev[vaccinated] = sim.diseases[self.vaccine.disease].prob_sev[vaccinated] * (1 - self.vaccine.prevent_severe * protection)
        if hasattr(self.vaccine, 'clear_carriage'): # Clear asymptomatic carriage for meningitis
            asymp_vaccinated = np.array([uid for uid in vaccinated if (~sim.diseases[self.vaccine.disease].symptomatic[uid] & sim.diseases[self.vaccine.disease].infectious[uid] & (sim.diseases[self.vaccine.disease].ti_vaccinated[uid]==sim.ti))], dtype=ss.int_)
            carriage_cleared = np.random.random(len(asymp_vaccinated)) < self.vaccine.clear_carriage
            sim.diseases[self.vaccine.disease].ti_recovered[asymp_vaccinated[carriage_cleared]] = sim.ti + 1
        sim.diseases[self.vaccine.disease].prob_death[vaccinated] = sim.diseases[self.vaccine.disease].prob_death[vaccinated] * (1 - self.vaccine.prevent_death * protection)

    @staticmethod
    def _get_ti_immune(uids, n_immune):
        """
        Args:
            uids: array of person indices
            n_immune: array of how many people should be immune for each day after vaccination (the first day is 0)
                      The length of this array is arbitrary, the length of this list defines the maximum value present
                      in the output array
        Returns: A list the same length as `uids`
        """

        n_gain_immunity = np.diff(n_immune)  # Number of people that gain immunity each day
        day_immune = np.zeros(uids.shape, dtype=ss.int_)

        count = 0
        for i in range(0, len(n_gain_immunity)):
            n_contacts = n_gain_immunity[i]
            day_immune[count : count + n_contacts] = i + 1  # The first entry (0th) in n_gain_immunity corresponds to gaining immunity 1 day afterwards
            count += n_contacts

            if count == len(day_immune):
                break

        return np.random.permutation(day_immune)  # Shuffle the order in which people gain immunity

    def vaccinate(self, sim, uids, t=None, update_immunity=True):
        """
        Use this function to vaccinate a group of people

        Args:
            sim:
            uids:
            t: Override vaccination date relative to simulation date (e.g. for historical vaccination). Can be a day index or a date

        Returns:

        """

        if not sc.isnumber(t):
            t = sim.ti

        # Validate indices of people to vaccinate - *essential* that we don't vaccinate anyone using this
        # intervention that has already been vaccinated, otherwise this intervention will interact unintentionally
        # with any other interventions that vaccinated that same person
        assert not np.any(sim.diseases[self.vaccine.disease].vaccinated[uids] | sim.diseases[self.vaccine.disease].dead[uids]), "Cannot re-vaccinate people with this intervention"  # nb. can disable this check for performance later on if needed

        # logger.debug(f'{self.label}: Vaccinating {len(uids)} agents at {t=}')
        sim.diseases[self.vaccine.disease].vaccinated[uids] = True
        if self.vaccine.dose_interval is None:
            # If it's a single dose vaccine, they are fully vaccinated after the first dose
            sim.diseases[self.vaccine.disease].fully_vaccinated[uids] = True
        self._vaccinated[uids] = True  # Record that they have been vaccinated by this intervention
        sim.diseases[self.vaccine.disease].ti_vaccinated[uids] = t

        '''Removed leaky functionality
        if not self.leaky:
            immune_uids = ss.binomial_filter(self.vaccine.prevent_infection, uids)  # Indices of people that will eventually gain immunity if 100% relative protection is reached
            n_immune = (self._immunity_timecourse * len(immune_uids)).astype(int)
            self._ti_immune[immune_uids] = self._get_ti_immune(immune_uids, n_immune)
            self._pending_immunity[immune_uids] = True  # Flag that these people should have their immunity updated
        '''

        if t >= 0:
            # Record the new vaccinations etc. if they are taking place during the simulation timeframe
            # We assume that the vaccination rollout continues outside the pool of agents being modelled here.
            # Following from the example in the `apply` method, if we vaccinate 10 agents, this corresponds to
            # vaccinating 5 people within the area being modelled i.e. 10*5. However, the area being modelled
            # accounts for a (5/10) fraction of the total population. Therefore the number of people vaccinated
            # is in fact 10*5/(5/10). As before, the factors cancel out, and we actually just multiply
            # by the pop_scale here
            new_vaccinated = int(len(uids) * sim.pars["pop_scale"])
            self.n_people_vaccinated[t] += new_vaccinated
            self.n_agents_vaccinated[t] += int(len(uids))
            self.results["vac_doses"].values[sim.ti] = np.cumsum(self.results["vac_doses"].values[sim.ti]) + new_vaccinated  # Update total test count

        if update_immunity:
            self.update_immunity(sim, t=t)

    def apply(self, sim, t=None, num_people=None):
        """
        Use this function to vaccinate a number of people, selected based on the stored sequence
        """

        if not sc.isnumber(t):
            t = sim.ti

        # Work out how many *people* to vaccinate today - matches reported numbers of doses for a jurisdiction
        if num_people is None:
            if sc.isnumber(self.num_doses):
                num_people = self.num_doses
            elif callable(self.num_doses):
                num_people = self.num_doses(sim)
            elif t in self.num_doses:
                num_people = self.num_doses[t]
            else:
                num_people = 0

        # Suppose we have a pop_scale of 10, and a current scale factor of 5, with 100 agents.
        # That means that we have a total population of 1000 people, and currently 1 person
        # represents 5 people. If we want to vaccinate 100 people in the population, how many
        # agents does this correspond to? Since we are currently modelling 500 people with 100
        # agents, assuming the 100 vaccines are uniformly distributed, we need to allocate
        # n = 100*(5/10) = 50 vaccines to the pool of people being modelled. Further, we then
        # need to divide this number of people by the current scale factor to get the number of
        # agents corresponding to 50 people. That is, 50/5=10 agents. Since the overall calculation is
        # (100*5/10)/5 this is just equivalent to dividing by the overall pop scale.
        num_agents = int(np.round(num_people / sim.pars["pop_scale"]))

        if self.dynamic_sequence:
            self.sequence = get_vaccine_sequence_grouped(sim.people, sim.diseases[self.vaccine.disease], "18+", eligible=None, priority=True)

        if num_agents and len(self.sequence):
            # People are ineligible for vaccination if they are already vaccinated or if they are dead
            # However, people who have died and are then returned to the simulation by rescaling still need
            # to be eligible for vaccination. Therefore we don't actually remove them from the schedule since they
            # could come back to life at any time in the simulation
            eligible = self.sequence[~sim.diseases[self.vaccine.disease].vaccinated[self.sequence] & ~sim.diseases[self.vaccine.disease].dead[self.sequence]]
            uids = eligible[:num_agents]  # nb. this indexing
        else:
            uids = np.array([], dtype=ss.int_)

        # Vaccinate them
        if len(uids):
            # Check uids at this point, in case num_agents > 0 but nobody was eligible and thus no vaccinations were performed
            self.vaccinate(sim, uids, t=t)
        else:
            self.update_immunity(sim, t=t)

        return uids


def get_vaccine_sequence_grouped(people, disease, groups, eligible=None, priority=False):
    """

    Args:
        people:
        groups:
        eligible: Boolean flag same length as people, optionally specifying whether people should be excluded from the sequence (if True, person will be included)
        priority_layers:

    Returns:

    """
    # Groups is of the form "16-39, 60+". Ages are inclusive
    groups = [x.strip() for x in groups.split(",")]
    include = np.full_like(people.age, fill_value=False, dtype=bool)
    for group in groups:
        age_lower, age_upper = ssg.parse_age_range(group)
        include = include | ((people.age >= age_lower) & (people.age <= age_upper))

    if eligible is not None:
        include = include & eligible
    else:
        include = include & ~disease.vaccinated

    if priority:
        priority_eligible = sc.dcp(disease.known_contact)
        prioritised = include & priority_eligible
        nonprioritised = include & ~priority_eligible
        prioritised = ss.true(prioritised)  # Convert to indices, filter by eligible
        nonprioritised = ss.true(nonprioritised)
        prioritised_indices = np.random.permutation(prioritised)
        other_indices = np.random.permutation(nonprioritised)
        return np.concatenate((prioritised_indices, other_indices))

    else:
        include = ss.true(include)  # Convert to indices, filter by eligible
        return np.random.permutation(include)


def get_vaccine_sequence_grouped_priority_layer(people, disease, groups, eligible=None, priority_layers=None):
    """

    Args:
        people:
        groups:
        eligible: Boolean flag same length as people, optionally specifying whether people should be excluded from the sequence (if True, person will be included)
        priority_layers:

    Returns:

    """
    # Groups is of the form "16-39, 60+". Ages are inclusive
    groups = [x.strip() for x in groups.split(",")]
    include = np.full_like(people.age, fill_value=False, dtype=bool)
    for group in groups:
        age_lower, age_upper = ssg.parse_age_range(group)
        include = include | ((people.age >= age_lower) & (people.age <= age_upper))

    if eligible is not None:
        include = include & eligible
    else:
        include = include & ~disease.vaccinated

    if priority_layers:
        priority_eligible = np.full(len(people), fill_value=False)
        for layer in priority_layers:
            if layer == "children_under_5":
                priority_eligible[people.age <= 5] = True
            else:
                raise Exception(f"Invalid layer '{layer}' for vaccine prioritisation")

        prioritised = include & priority_eligible
        nonprioritised = include & ~priority_eligible
        prioritised = ss.true(prioritised)  # Convert to indices, filter by eligible
        nonprioritised = ss.true(nonprioritised)
        prioritised_indices = np.random.permutation(prioritised)
        other_indices = np.random.permutation(nonprioritised)
        return np.concatenate((prioritised_indices, other_indices))

    else:
        include = ss.true(include)  # Convert to indices, filter by eligible
        return np.random.permutation(include)