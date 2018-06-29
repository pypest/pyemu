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