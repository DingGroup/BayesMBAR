# Bayesian Multistate Bennett Acceptance Ratio Method

This repository contains the code for the Bayesian Multistate Bennett Acceptance Ratio Method as described in the [paper](https://doi.org/10.1021/acs.jctc.3c01212). BayesMBAR is a Bayesion generalization of the Multistate Bennett Acceptance Ratio (MBAR) method for computing free energy differences between multiple states. 

Besides its theoretical interest, BayesMBAR has two practical advantages over MBAR. First, it provides a more accurate uncertainty estimate, especially when the number of samples is small or the phase space overlap between states is poor. Second, it allows for the incorporation of prior information to improve the accuracy of the free energy estimates. For example, when the free energy surface over a collective variable is known to be smooth, BayesMBAR can use this information to improve the accuracy of the free energy estimates.  The [paper](https://doi.org/10.1021/acs.jctc.3c01212) has more details on the method and its applications.

We are committed to making the code as user-friendly as possible. We are actively working on improving the [documentation](http://xdinglab.com/BayesMBAR/) and adding more examples. If you have any questions or suggestions, please feel free to open an issue or [contact us](mailto:Xinqiang.Ding@tufts.edu) directly. 
