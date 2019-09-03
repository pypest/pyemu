import os

nb_files = [f for f in os.listdir(".") if f.lower().endswith(".ipynb")]
for nb_file in nb_files:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {0}".format(nb_file))

