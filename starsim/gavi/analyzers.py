import starsim as ss
import gavi.multisim as ssm
import gavi as ssg
import numpy as np

__all__ = ['TrackOutbreakDur', 'UpdateResults']

def add_result(sim, name, vals, module='analyzer'):
    if name in sim.results.keys():
        raise Exception(f'Attempted to add result "{name}" which already exists')
    sim.results[name] = ssm.MultiSimResult(module=module, name=name, npts=sim.npts)
    sim.results[name].values = vals.copy()
    return

class TrackOutbreakDur(ss.Analyzer):
    """
    Record the average duration of symptoms during a model run
    """

    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.label = 'track_outbreak_dur_' + disease
        self.num_death_recover = np.zeros_like(sim.tivec, dtype=ss.float_)
        self.check_outbreak_end = True
        self.outbreak_end = 0

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, self.disease+"-num_death_recover", self.num_death_recover)
        sim.results[self.disease+'-outbreak_end'] = np.array([self.outbreak_end])
        sim.results[self.disease+'-outbreak_dur'] = np.array([sim.results[self.disease+'-outbreak_end'][0] - sim.results[self.disease+'-outbreak_detection'][0]])

    def apply(self, sim):
        recovered_inds = ss.true(sim.diseases[self.disease].ti_recovered <= sim.ti)
        dead_inds = ss.true(sim.diseases[self.disease].ti_dead <= sim.ti)
        self.num_death_recover[sim.ti] = len(recovered_inds) + len(dead_inds)
        if sim.ti >= 42 and self.num_death_recover[sim.ti-42] > 0 and self.num_death_recover[sim.ti] == self.num_death_recover[sim.ti-42] and self.check_outbreak_end:
            self.outbreak_end = sim.ti
            self.check_outbreak_end = False

class SafelyBuried(ss.Analyzer):
    """
    Record the number of burials which are handled safely
    """

    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.cum_safe_buried = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "cum_safe_buried", self.cum_safe_buried)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        dead_uids = ss.true(sim.diseases[self.disease].ti_dead == sim.ti)
        safely_buried = (sim.diseases[self.disease].ti_diagnosed[dead_uids] <= sim.diseases[self.disease].ti_dead[dead_uids]).values
        not_safely_buried = ~(sim.diseases[self.disease].ti_diagnosed[dead_uids] <= sim.diseases[self.disease].ti_dead[dead_uids]).values
        sim.diseases[self.disease].ti_buried[dead_uids[safely_buried]] = sim.diseases[self.disease].ti_dead[dead_uids[safely_buried]]
        sim.diseases[self.disease].ti_buried[dead_uids[not_safely_buried]] = sim.diseases[self.disease].ti_dead[dead_uids[not_safely_buried]] + sim.diseases[self.disease].pars['dur_dead2buried'].sample(sum(not_safely_buried))

        self.cum_safe_buried[sim.ti] = self.cum_safe_buried[sim.ti-1] + sum(safely_buried)

class AgeOfDeath(ss.Analyzer):
    """
    Record the average age of death during a model run
    """

    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.average_death_age = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "average_death_age", self.average_death_age)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        dead_uids = ss.true(sim.diseases[self.disease].ti_dead <= sim.ti)
        if len(dead_uids) > 0:
            self.average_death_age[sim.ti] = np.nanmean(sim.people.age[dead_uids])

class InfectionsByAge(ss.Analyzer):
    """
    Record the number of infection in specific age-groups during a model run
    """
    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.infection_cat = ['all', 'symp', 'asymp']
        self.age_bins = ["0-4", "4-9", "10-19", "20-29", "30+"]
        self.outputs = {}
        for cat in self.infection_cat:
            for age_range in self.age_bins:
                self.outputs[cat + '_inf_' + age_range] = np.zeros(sim.npts, dtype=ss.float_)

    def apply(self, sim):
        for cat in self.infection_cat:
            for age_range in self.age_bins:
                age_lower, age_upper = ssg.parse_age_range(age_range)
                age_inds = (sim.people.age >= age_lower) & (sim.people.age <= age_upper)
                if cat == 'all':
                    self.outputs[cat + '_inf_' + age_range][sim.ti] = np.count_nonzero(sim.diseases[self.disease].ti_infectious[age_inds] == sim.ti) / np.sum(age_inds)
                elif cat == 'symp':
                    self.outputs[cat + '_inf_' + age_range][sim.ti] = np.count_nonzero(((sim.diseases[self.disease].ti_infectious[age_inds] == sim.ti) & sim.diseases[self.disease].symptomatic[age_inds])) / np.sum(age_inds)
                elif cat == 'asymp':
                    self.outputs[cat + '_inf_' + age_range][sim.ti] = np.count_nonzero(((sim.diseases[self.disease].ti_infectious[age_inds] == sim.ti) & ~sim.diseases[self.disease].symptomatic[age_inds])) / np.sum(age_inds)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def finalize(self, sim):
        super().finalize(sim)
        for cat in self.infection_cat:
            for age_range in self.age_bins:
                add_result(sim, cat + '_inf_' + age_range, np.cumsum(self.outputs[cat + '_inf_' + age_range]))


class CurrentlyHospitalised(ss.Analyzer):
    """
    Record whether a person was severely ill at the time of their diagnosis
    """

    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.diagnosed_severe = np.zeros_like(sim.tivec, dtype=ss.float_)
        self.new_diagnosed_severe = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "diagnosed_severe", self.diagnosed_severe)
        add_result(sim, "new_diagnosed_severe", self.new_diagnosed_severe)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        uids = ss.true((sim.diseases[self.disease].ti_diagnosed <= sim.ti) & ~sim.diseases[self.disease].dead & ~sim.diseases[self.disease].recovered)
        self.diagnosed_severe[sim.ti] = sim.diseases[self.disease].severe[uids].sum()
        today_sev_uids = ss.true((sim.diseases[self.disease].ti_severe == sim.ti) & ~(sim.diseases[self.disease].ti_diagnosed == sim.ti))
        today_diag_uids = ss.true(sim.diseases[self.disease].ti_diagnosed == sim.ti)
        self.new_diagnosed_severe[sim.ti] = np.sum(sim.diseases[self.disease].diagnosed[today_sev_uids]) + np.sum(sim.diseases[self.disease].severe[today_diag_uids])

class DurationOfSymptoms(ss.Analyzer):
    """
    Record the average duration of symptoms during a model run
    """

    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.average_symp_dur = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "average_symp_dur", self.average_symp_dur)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        recovered_uids = ss.true(sim.diseases[self.disease].ti_recovered <= sim.ti)
        dead_uids = ss.true(sim.diseases[self.disease].ti_dead <= sim.ti)
        if (len(recovered_uids) > 0) & (len(dead_uids) > 0):
            self.average_symp_dur[sim.ti] = np.nanmean(np.concatenate((sim.diseases[self.disease].ti_recovered[recovered_uids],
                                                                       sim.diseases[self.disease].ti_dead[dead_uids])) -
                                                       np.concatenate((sim.diseases[self.disease].ti_infectious[recovered_uids],
                                                                       sim.diseases[self.disease].ti_infectious[dead_uids])))

class ProportionChildrenInfected(ss.Analyzer):
    """
    Record the proportion of children infected during a model run
    """
    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.prop_child_infected = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "prop_child_infected", self.prop_child_infected)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        age_vals = (sim.people.age < 18)
        exposed = sim.diseases[self.disease].severe
        if exposed.sum() > 0:
            self.prop_child_infected[sim.ti] = exposed[age_vals].sum() / exposed.sum()

'''Deprecated, DALYs and costs calculated post-simulation
class CalcDALYsCosts(ss.Analyzer): #ToDo: update values
    def calc_dalys(self, infections, deaths, av_death_age, inf_dur):
        life_exp = 60.0
        dw_acute = 0.133  # from GBD DWs
        dw_chronic = 0.219  # from GBD DWs
        prop_chronic = 0.7  # assumption, in line with https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5818139/, https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(15)00259-5/fulltext, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4806950/
        chronic_dur = 1  # assumption, there is limited data from longitudinal studies but WHO reports 2+ years
        year_discounts = [0.97 ** i for i in list(range(np.max((int(life_exp - av_death_age), 0))))]
        if year_discounts != []:
            yll = deaths * np.max((life_exp - av_death_age), 0) * np.mean(year_discounts)
        else:
            yll = 0
        yld = infections * dw_acute * inf_dur/365 + (infections - deaths) * prop_chronic * dw_chronic * chronic_dur
        daly = yll + yld
        return daly

    def calc_cost(self, hosp_time, num_hosp, num_sb, prop_child, daly):
        gdp = 524.7/1.24
        svly = 1.75
        cost_paracet = 0.00301
        cost_ors = 0.1584
        cost_metaclop = 0.00895
        cost_morph = 0.2679
        cost_ringlac = 1.0758
        cost_ceftriax = 0.836
        cost_diaz = 0.0942

        doc_daily = 36.25 * gdp / 365
        nurse_daily = 6.43 * gdp / 365
        # Drug, PPE, staffing costs based on methods from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4445295/
        adult_med_cost = (1 - prop_child) * (24 * hosp_time * (
                    8 * cost_paracet / 6 + 0.6 * cost_ors / 6 + cost_metaclop / 8 + 0.2 * (
                        3 * cost_morph / 24 + 1.2 * cost_ringlac + 2 * cost_ceftriax / 24)) + 4.5 * cost_ors * num_hosp + 0.2 * 4 * cost_diaz)
        child_med_cost = prop_child * (24 * hosp_time * (
                    3 * cost_paracet / 6 + 0.3 * cost_ors / 6 + cost_metaclop / 8 + 0.2 * (
                        2 * cost_morph / 24 + 1.2 * cost_ringlac + 2 * cost_ceftriax / 24 + 0.3 * cost_ors / 6)) + 4.5 * 0.1584 * num_hosp + 0.2 * 4 * cost_diaz)
        ppe_cost = 7 * hosp_time * (2.6 + 0.57 + 0.16 + 0.74 + 0.07 + 0.7 + 1.32 + 0.21)
        burial_cost = 25 * num_sb
        wage_cost = (doc_daily + nurse_daily) / 16.4
        direct_cost = adult_med_cost + child_med_cost + ppe_cost + wage_cost + burial_cost
        indirect_cost = daly * svly * gdp

        total_cost = direct_cost + indirect_cost
        return total_cost * 1.24  # 2014 USD to 2022 USD

    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        assert (any([analyzer == 'ageofdeath' for analyzer in sim.analyzers]) and any([analyzer == 'durationofsymptoms' for analyzer in sim.analyzers])
        and any([analyzer == 'currentlyhospitalised' for analyzer in sim.analyzers]) and any([analyzer == 'proportionchildreninfected' for analyzer in sim.analyzers])
        and any([analyzer == 'safelyburied' for analyzer in sim.analyzers])), 'CalcDALYsCosts analyzer requires other analyzers'
        self.AODInd = [analyzer == 'ageofdeath' for analyzer in sim.analyzers].index(True)
        self.DOSInd = [analyzer == 'durationofsymptoms' for analyzer in sim.analyzers].index(True)
        self.CHInd = [analyzer == 'currentlyhospitalised' for analyzer in sim.analyzers].index(True)
        self.PCIInd = [analyzer == 'proportionchildreninfected' for analyzer in sim.analyzers].index(True)
        self.SBInd = [analyzer == 'safelyburied' for analyzer in sim.analyzers].index(True)
        self.cum_dalys = np.zeros_like(sim.tivec, dtype=ss.float_)
        self.cum_costs = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "cum_dalys", self.cum_dalys)
        add_result(sim, "cum_costs", self.cum_costs)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        self.cum_dalys[sim.ti] = self.calc_dalys(np.sum(sim.diseases[self.disease].ti_infected <= sim.ti), np.sum(sim.diseases[self.disease].ti_dead <= sim.ti),
                                                sim.analyzers[self.AODInd].average_death_age[sim.ti], sim.analyzers[self.DOSInd].average_symp_dur[sim.ti])
        self.cum_costs[sim.ti] = self.calc_cost(sim.analyzers[self.CHInd].diagnosed_severe[sim.ti],
                                               sim.analyzers[self.CHInd].new_diagnosed_severe[sim.ti],
                                               sim.analyzers[self.SBInd].cum_safe_buried[sim.ti],
                                               sim.analyzers[self.PCIInd].prop_child_infected[sim.ti], self.cum_dalys[sim.ti])

'''

class ProportionSevere(ss.Analyzer):
    """
    Cumulative severe/diagnosed
    """

    def apply(self, sim):
        return

    def finalize(self, sim):
        super().finalize(sim)
        a = sim.results["cum_severe"].values
        b = sim.results["cum_diagnoses"].values
        add_result(sim, "proportion_severe", np.divide(a, b, out=np.zeros_like(a), where=b > 0))

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

class VaccinatedKnownContacts(ss.Analyzer):
    """
    Track how well ring-based vaccination is working by recording contacts who get vaccinated
    """
    def initialize(self, sim, disease=None):
        super().initialize(sim)
        if disease is None:
            disease = sim.diseases[0].name
        self.disease = disease
        self.vaccinated_contacts = np.zeros_like(sim.tivec, dtype=ss.float_)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, self.disease+"-cum_vaccinated_contacts", self.vaccinated_contacts * sim.pars["pop_scale"])

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def apply(self, sim):
        self.vaccinated_contacts[sim.ti] = np.sum((sim.diseases[self.disease].ti_known_contact <= sim.ti) & sim.diseases[self.disease].vaccinated)

class UpdateResults(ss.Analyzer):
    """
    Work around for making default starsim results usable for Multisim
    """

    def initialize(self, sim):
        super().initialize(sim)
        self.diseases = []
        for disease in sim.diseases:
            self.diseases.append(disease)
        self.results = ss.ndict(type=ssm.MultiSimResult)

    def update_results(self, sim):
        # Makes Analyzer.update_results(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(sim)

    def finalize(self, sim):
        super().finalize(sim)
        for key, result in sim.results.items():
            if isinstance(result, ssm.MultiSimResult):
                self.results += result
            elif isinstance(result, ss.Result):
                self.results += ssm.MultiSimResult(module=result.module, name=result.name, npts=len(result), dtype=result.dtype)
                self.results[-1].values[:] = result[:]
            elif key in self.diseases:
                for k, res in result.items():
                    if isinstance(res, ssm.MultiSimResult):
                        self.results[key+'-'+k] = res
                    elif isinstance(res, ss.Result):
                        self.results += ssm.MultiSimResult(module=res.module, name=key+'-'+res.name, npts=len(res), dtype=res.dtype)
                        self.results[-1].values[:] = res[:]
                    else:
                        self.results[key+'-'+k] = res
            else:
                self.results[key] = result
        for key, interv in sim.interventions.items():
            if len(interv.results) > 0:
                for k, result in interv.results.items():
                    if isinstance(result, ssm.MultiSimResult):
                        self.results[key+'-'+k] = result
                    elif isinstance(result, ss.Result):
                        self.results += ssm.MultiSimResult(module=result.module, name=key+'-'+result.name, npts=result.shape, dtype=result.dtype)
                        self.results[-1].values[:] = result[:]
        sim.results = self.results


    def apply(self, sim):
        pass

