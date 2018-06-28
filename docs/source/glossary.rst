Glossary
--------

.. glossary::
   :sorted:
      
   object
   instance
      generated from the class...
      
   class
      Classes [PSF18]_ are the building blocks for object-oriented programs. The class
      defines both attributes and functionality for an :term:`object`.
      
   client code
      In the case of pyEMU, client code is a python script that uses the class definitions to
      make a desired :term:`instance` (also know as an :term:`object`) and to use the attributes
      (data associated with the instance) and methods (functions that may be applied to
      the instance or done by the instance) to build the desired analysis.  The client code
      can be developed as a traditional python script or within a Jupyter notebook [KRP+16]_. 
      Jupyter notebooks are used extensively in this documentation to illustrate features of
      pyEMU.
      
   state
      The state of the object refers to the current value of all attributes assigned to an object.
      
   instance method
      Functionality that is defined in a class and becomes associated with a particular
      instance or object after :term:`instantiation`.
      
   instantiation
      Make a class instance, also known as an object, from a class.