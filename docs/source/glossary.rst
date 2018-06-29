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
      "observations"!  These are denoted by :math:`f(x)` for a scalar value or :math:`f(\bf{x})`
      for a vector of values in this documentation.

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
      squared residuals. In principal, :math:`w\approx\frac{1}{\sigma}` where :math:`\sigma` is
      an approximation of the expected error between model output and collocated
      observation values. While the symbol :math:`\sigma` implies a standard deviation,
      it is important to note that measurement error only makes up a portion of
      this error. Other aspects such as structural error (e.g. inadequacy inherent
      in all models to perfectly simulate the natural world) also contribute to
      this expected level of error. The reciprocal of weights are also called
      Epistemic Uncertainty terms and in matrix form is denoted by :math:`\bf{Q}^{-1}`.

   residuals
   
      The difference between observation values and modeled equivalents
      :math:`r_i=y_i-f\left(x_i\right)`.

   sensitivity
      The incremental change of an observation (actually the modeled equivalent)
      due to an incremental change in a parameter. Typically expressed as a
      finite-difference approximation of a partial derivative, :math:`\frac{\partial
      y}{\partial x}`

   Jacobian matrix
      A matrix of the sensitivity of all observations in an inverse model to all
      parameters. This is often shown as a matrix by various names :math:`\mathbf{X}`,
      :math:`\mathbf{J}`, or :math:`\mathbf{H}`. Each element of the matrix is a single
      sensitivity value  :math:`\frac{\partial y_i}{\partial x_j}` for :math:`i\in NOBS`, :math:`j
      \in NPAR`.

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



