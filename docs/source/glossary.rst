Glossary
--------

.. glossary::
   :sorted:
      
   object
   instance
      A programming entity that may store internal data (:term:`state`), have
      functionality (:term:`behavior`) or both [RS16]_ .
      
   class
      Classes [PSF18]_ [RS16]_ are the building blocks for object-oriented programs. The class
      defines both attributes (:term:`state`) and functionality (:term:`behavior`) of an :term:`object`.
      
   client code
      In the case of pyEMU, client code is a python script that uses the class definitions to
      make a desired :term:`instance` (also know as an :term:`object`) and to use the :term:`attributes`
      and :term:`instance methods` to build the desired analysis.  The client code
      can be developed as a traditional python script or within a Jupyter notebook [KRP+16]_. 
      Jupyter notebooks are used extensively in this documentation to illustrate features of
      pyEMU.
      
   state
      The state of the object refers to the current value of all the internal data 
      stored in and object [RS16]_ .
      
   instance methods
      Functions defined in a class that become associated with a particular
      instance or object after :term:`instantiation`.
      
   instantiation
      Make a class instance, also known as an object, from a class.
      
   behavior
      Object behavior refers to the set of actions an object can perform often reporting
      or modifying its internal :term:`state` [RS16]_ .
      
   attributes
      Internal data stored by an object.
      
   static methods
   helper methods
      Functions that are defined for the python module and available to the script. They
      are not tied to specified objects.  pyEMU has a number of helper methods including 
      functions to plot results, read or write specific data types, and interact with the
      :term:`FloPY` module in analysis of MODFLOW models.
      
   FloPY
      Object-oriented python module to build, run, and process MODFLOW models [BPL+16]_ . Because
      of the similar modeling structure, the combination of flopy and pyEMU can be
      very effective in constructing, documenting, and analyzing groundwater-flow models. 
      
   inheritance
      A object-oriented programming technique where one class, referred to as the
      :term:`derived class` or subclass, extends a second class, known as the :term:`base class`
      or superclass.  The subclass has all the defined attributes and behavior
      of the superclass and typically adds additional attributes or methods. Many
      derived classes :term:`override` methods of the superclass to perform specific
      functions [RS16]_.
      
   override
      Implement a new version of a method within a derived class that would have
      otherwise been inherited from the base class customizing the behavior
      of the method for the derived class [RS16]_.  Overriding allows objects to
      call a method by the same name but have behavior specific to the 
      object's type.
      
   base class
      A base class, or superclass, is a class that is extended by another class 
      in a technique called :term:`inheritance`. Inheritance can make programming
      efficient by having one version of base attributes and methods that can be
      tested and then extended by a :term:`derived class`.
      
   derived class
      A derived class, or subclass, inherits attributes and methods from its
      :term:`base class`.  Derived classes then add attributes and methods as
      needed.  For example, in pyEMU, the linear analysis base class is inherited by the
      Monte Carlo derived class.
      
   GLUE
      Generalized Likelihood Uncertainty Estimation [KB09]_ .
      
   parameters
      Variable input values for models, typically representing system properties
      and forcings. Values to be estimated in the history matching process.

   observation
      Measured system state values. These values are used to compare with model
      outputs collocated in space and time. The term is often used to mean
      *both* field measurements and outputs from the model. These are denoted 
      by :math:`y` for a scalar value or :math:`\bf{y}`
      for a vector of values in this documentation.

   modeled equivalent 
   simulated equivalent
      A modeled value collocated to correspond in time and space with an observation. 
      To make things confusing, they are often *also* called
      "observations"!  These are denoted by :math:`f(x)` for a scalar value or :math:`f\left( \bf{x} \right)`
      for a vector of values in this documentation.  Note that in addition to
      :math:`f(x)`, authors use several different alternate
      notations to distinguish an observation from a 
      corresponding modeled equivalent including subscripts, :math:`y` and :math:`y_m`,
      superscripts, :math:`y` and :math:`y^\prime`, or diacritic marks, :math:`y` and :math:`\hat{y}`.

   forecasts
      Model outputs for which field observations are not available. Typically these
      values are simulated under an uncertain future condition.

   Phi
      For pyEMU and consistent with PEST and PEST++, Phi refers to the :term:`objective function`, 
      defined as the weighted sum of squares of residuals. Phi, :math:`\Phi`, is typically calculated as
 
      .. math::
         \begin{array}{ccc}
         \Phi=\sum_{i=1}^{n}\left(\frac{y_{i}-f\left(x_{i}\right)}{w_{i}}\right)^{2} & or & \Phi=\left(\mathbf{y}-\mathbf{Jx}\right)^{T}\mathbf{Q}^{-1}\left(\mathbf{y}-\mathbf{Jx}\right)
         \end{array}
         
      When regularization is included, an additional term is added, 
      quantifying a penalty assessed for parameter sets that violate the preferred 
      conditions regarding the parameter values. 
      In such a case, the value of :math:`\Phi` as stated above is 
      renamed :math:`\Phi_m` for "measurement Phi" and the additional regularization 
      term is named :math:`\Phi_r`. A scalar, :math:`\gamma`,  parameter controls the 
      tradeoff between these two dual components of the total objective function :math:`\Phi_t`.

      .. math::
         \Phi_t = \Phi_m + \gamma  \Phi_r

   weight 
   epistemic uncertainty
      A value by which a residual is divided by when constructing the sum of
      squared residuals. In principal, :math:`w_i\approx\frac{1}{\sigma_i}` where :math:`\sigma_i` is
      an approximation of the expected error between model output and collocated
      observation values. While the symbol :math:`\sigma` implies a standard deviation,
      it is important to note that measurement error only makes up a portion of
      this error. Other aspects such as structural error (e.g. inadequacy inherent
      in all models to perfectly simulate the natural world) also contribute to
      this expected level of error. The reciprocal of weights are also called
      Epistemic Uncertainty terms.

   residuals
   
      The difference between observation values and modeled equivalents
      :math:`r_i=y_i-f\left(x_i\right)`.

   sensitivity
      The incremental change of an observation (actually the modeled equivalent)
      due to an incremental change in a parameter. Typically expressed as a
      finite-difference approximation of a partial derivative, :math:`\frac{\partial
      f(x)}{\partial x}`.

   Jacobian matrix
      A matrix of the sensitivity of all observations in an inverse model to all
      parameters. This is often shown as a matrix by various names :math:`\mathbf{X}`,
      :math:`\mathbf{J}`, or :math:`\mathbf{H}`. Each element of the matrix is a single
      sensitivity value  :math:`\frac{\partial f(x_i)}{\partial x_j}` for :math:`i\in NOBS`, :math:`j
      \in NPAR`.
      
   *NOBS*
      For PEST and PEST++, number of observations.
      
   *NPAR*
      For PEST and PEST++, number of parameters.

   regularization
      A preferred condition pertaining to parameters, the deviation from which,
      elicits a penalty added to the objective function. This serves as a
      balance between the level of fit or "measurement Phi"
      :math:`(\mathbf{\Phi_M})` and the coherence with soft knowledge/previous
      conditions/prior knowledge/regularization :math:`(\mathbf{\Phi_R})`. These terms
      can also be interpreted as the likelihood function and prior distribution
      in Bayes theorem.

   PHIMLIM
      A PEST input parameter the governs the strength with which regularization
      is applied to the objective function. A high value of PHIMLIM indicates a
      strong penalty for deviation from preferred parameter conditions while a
      low value of PHIMLIM indicates a weak penalty. The reason this "dial"" is
      listed as a function of PHIM (e.g. :math:`\mathbf{\Phi_M}`) is because it can
      then be interpreted as a limit on how well we want to fit the observation
      data. PHIMLIM is actually controlling the value :math:`\gamma` appearing in the
      definition for :term:`Phi` and formally trading off :math:`\Phi_m` asnd :math:`\Phi_r`.

   objective function
      Equation quantifying how well as simulation matches observations.  Inverse
      modeling is the process of calibrating a numerical model by mathematically
      seeking the minimum of an objective function. (see :term:`Phi`)

   Gaussian (multivariate)
      The equation for Gaussian (Normal) distribution for a single variable (:math:`x`) is 
      
      .. math::
        \begin{equation}
        f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2}\frac{\left(x-\mu\right)^2}{\sigma^2}}
        \end{equation}
        
      where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation
      The equation for a multivariate Gaussian for a vector of :math:`k` variables (:math:`\mathbf{x}`) is
        
      .. math::
        \begin{equation}
        f(\mathbf{x} | \mathbf{\mu},\mathbf{\Sigma})=\frac{1}{\sqrt{(2\pi)^k\left|\mathbf{\Sigma}\right|}}e^{-\frac{1}{2}\left( \left(\mathbf{x}-\mathbf{\mu} \right)^T  \mathbf{\Sigma}^{-1}\left(\mathbf{x}-\mathbf{\mu} \right)\right)}
        \end{equation}
        
     where :math:`\mu` is a :math:`k`-length vector of mean values, :math:`\mathbf{\Sigma}` is 
     the covariance matrix, and :math:`\left|\mathbf{\Sigma}\right|` is the determinant of 
     the covariance matrix.  These quantities are often abbreviated 
     as :math:`\mathcal{N}\left( \mu, \sigma \right)` 
     and :math:`\mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma} \right)` 
     for univariate and multivariate Gaussian distributions, respectively.

   weight covariance matrix (correlation matrix)
      In practice, this is usually a :math:`NOBS\times NOBS` diagonal matrix with
      values of weights on the diagonal representing the inverse of the
      observation covariance. This implies a lack of correlation among the
      observations. A full covariance matrix would indicate correlation among
      the observations which, in reality, is present but, in practice, is rarely
      characterized. The weight matrix is often identified as :math:`\mathbf{Q}^{-1}`
      or :math:`\mathbf{\Sigma_\epsilon}^{-1}`.

   parameter covariance matrix
      The uncertainty of parameters can be expressed as a matrix as well. This
      is formed also as a diagonal matrix from the bounds around parameter
      values (assuming that the range between the bounds indicates :math:`4\sigma`
      (e.g. 95 percent of a normal distribution).) In pyEMU, some functions
      accept a *sigma\_range* argument which can override the :math:`4\sigma`
      assumption. In many cases of our applications, parameters are spatially
      distributed (e.g. hydraulic conductivity fields) so a covariance matrix
      with off-diagonal terms can be formed to characterize not only their
      variance but also their correlation/covariance. We often use
      geostatistical variograms to characterize the covariance of parameters.
      The parameter covariance matrix is often identified as :math:`C(\mathbf{p})`,
      :math:`\mathbf{\Sigma_\theta}`, or :math:`\mathbf{R}`.

   measurement noise/error
      Measurement noise is a contribution to Epistemic Uncertainty. This is the
      expected error of repeated measurements due to things like instrument
      error and also can be compounded by error of surveying a datum, location
      of an observation on a map, and other factors.
   
   structural (model) error
      Epistemic uncertainty is actually dominated by structural error relative
      to measurement noise. The structural error is the expected misfit between
      measured and modeled values at observation locations due to model
      inadequacy (including everything from model simplification due to the
      necessity of discretizing the domain, processes that are missing from the
      model, etc.).
   
   Monte Carlo parameter realization
      A set of parameter values, often but not required to be a multi-Gaussian
      distribution, sampled from the mean values of specified parameter values
      (either starting values or, in some cases, optimal values following
      parameter estimation) with covariance from a set of variance values, or a
      covariance matrix. Can be identified as :math:`\mathbf{\theta}`.
   
   Monte Carlo observation realization
      A set of observation values, often but not required to be a multi-Gaussian
      distribution, sampled using the mean values of measured observations and
      variance from the observation weight covariance matrix. Can be identified
      as :math:`\boldsymbol{d_{obs}}`.

   
   Monte Carlo ensemble
      A group of realizations of parameters, :math:`\mathbf{\Theta}`, observations, 
      :math:`\mathbf{D_{obs}}`, and the simulated equivalent values,
      :math:`\mathbf{D_{sim}}`. Note that these three matrices are made up of column
      vectors representing all of the :math:`\boldsymbol{\theta}`, :math:`\mathbf{d_{obs}}`,
      and :math:`\mathbf{d_{sim}}` vectors.

   Bayes' Theorem
      .. math::
         \begin{equation}
         P\left(\boldsymbol{\theta}|\boldsymbol{d}\right) = 
         \frac{\mathcal{L}\left(\textbf{d}|\boldsymbol{\theta}\right) P\left(\boldsymbol{\theta}\right)}
         {P\left(\textbf{d}\right)} \ldots 
         \underbrace{P\left(\boldsymbol{\theta}|\textbf{d}\right)}_{\text{posterior pdf}} \propto 
         \underbrace{\mathcal{L}\left(\boldsymbol{d}|\boldsymbol{\theta}\right)}_{\text{likelihood function}}
         \quad
         \underbrace{P\left(\boldsymbol{\theta}\right)}_{\text{prior pdf}}
         \end{equation}
      
      where :math:`\boldsymbol{\theta}` is a vector of parameters,
      and :math:`\mathbf{d}` is a vector of observations It is computationally
      expedient to assume that these quantities can be characterized by
      multivariate Gaussian distributions and, thus, characterized only by their
      first two moments --- mean and covariance.
   
   posterior (multivariate distribution)
      The posterior distribution is the updated distribution (mean and
      covariance) of parameter values :math:`\boldsymbol{\theta}`  updated from their
      prior by an experiment (encapsulated in the likelihood function). In other
      words, information gleaned from observations :math:`\mathbf{d}` is used to
      update the initial values of the parameters.
   
   prior (multivariate distribution)
      This distribution represents what we know about parameter values prior to
      any modeling. It is also called "soft knowledge" or "expert
      knowledge". This information is more than just starting values, but also
      encapsulates the understanding of uncertainty (characterized through the
      covariance) based on direct estimation of parameters (e.g. pumping tests,
      geologic maps, and grain size analysis, for example). In one
      interpretation of the objective function, this is also where the
      regularization information is contained.
   
   likelihood (multivariate distribution)
      This is a function describing how much is learned from the model. It is
      characterized by the misfit between modeled equivalents and observations.
   
   FOSM 
   linear uncertainty analysis
      First Order Second Moment (FOSM) is a technique to use an assumption of Gaussian
      distributions to, analytically, calculate the covariance of model outputs
      considering both the prior covariance and the likelihood function. In
      other words, it's an analytical calculation of the posterior covariance of
      parameters using Bayes' Theoerem. The equation for this calculation is the
      Schur Complement. The key advantage of this is that we really only need a
      few quantities --- a Jacobian Matrix :math:`\mathbf{J}`, the prior covariance of
      parameters :math:`\boldsymbol{\Sigma_\theta}`, and the observation covariance
      :math:`\boldsymbol{\Sigma_\epsilon}`.
   
   Schur complement
      The formula used to propagate uncertainty from a prior through a
      "notional" calibration (via the Jacobian) to the posterior update.

      .. math::
         \begin{equation}
         \underbrace{\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}}}_{\substack{\text{what we} \\ \text{know now}}} = \underbrace{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}}_{\substack{\text{what we} \\ \text{knew}}} - \underbrace{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\bf{J}^T\left[\bf{J}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\bf{J}^T + \boldsymbol{\Sigma}_{\boldsymbol{\epsilon}}\right]^{-1}\bf{J}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}}_{\text{what we learned}}
         \end{equation}
 