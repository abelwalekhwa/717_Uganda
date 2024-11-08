"""
Define interventions
"""

import starsim as ss
import sciris as sc
import pylab as pl


__all__ = ['Intervention']


class Intervention(ss.Module):

    def __call__(self, *args, **kwargs):
        # Makes Intervention(sim) equivalent to Intervention.apply(sim)
        if not self.initialized:  # pragma: no cover
            errormsg = f'Intervention (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)

    '''
        Base class for interventions. By default, interventions are printed using a
        dict format, which they can be recreated from. To display all the attributes
        of the intervention, use disp() instead.

        To retrieve a particular intervention from a sim, use sim.get_intervention().

        Args:
            label       (str): a label for the intervention (used for plotting, and for ease of identification)
            show_label (bool): whether or not to include the label in the legend
            do_plot    (bool): whether or not to plot the intervention
            line_args  (dict): arguments passed to pl.axvline() when plotting
        '''

    def __init__(self, show_label=False, do_plot=None, line_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_label = show_label  # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else True  # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0),
                                       line_args)  # Do not set alpha by default due to the issue of overlapping interventions
        self.days = []  # The start and end days of the intervention
        self.initialized = False  # Whether or not it has been initialized
        self.finalized = False  # Whether or not it has been initialized
        return

    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k, v in pars.items()])
                output = f"cv.{which}({parstr})"
            except Exception as E:
                output = f'{type(self)} (error: {str(E)})'  # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'

    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)

    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        super().initialize(sim)
        self.initialized = True
        self.finalized = False
        return

    def finalize(self, sim=None):
        '''
        Finalize intervention

        This method is run once as part of `sim.finalize()` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:  # pragma: no cover
            raise RuntimeError(
                'Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        super().finalize(sim)
        self.finalized = True
        self.finalize_results(sim)
        return

    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError

    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place:  # pragma: no cover
            return self
        else:
            return sc.dcp(self)

    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines on days when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.

        Note 2: By default, the intervention is plotted at the days stored in self.days.
        However, if there is a self.plot_days attribute, this will be used instead.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_days'):
                days = self.plot_days
            else:
                days = self.days
            if sc.isiterable(days):
                label_shown = False  # Don't show the label more than once
                for day in days:
                    if sc.isnumber(day):
                        if self.show_label and not label_shown:  # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        date = sc.date(sim.date(day))
                        ax.axvline(date, label=label, **line_args)
        return

    def finalize_results(self, sim):
        pass
