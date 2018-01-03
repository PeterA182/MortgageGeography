import os
import sys
import subprocess

"""
To predict whether or not a given loan will be in default within the next "x"
number of months given the performance data available on that mortgage at 
the present time
"""

#
# ---- ---- ----
# Run Variables

# Source of loans data
d_source = 'freddie'

# Name of current model being run
model_name = 'comb_logisticRegression'

#
# ---- ---- ----
# Set Up Paths

# Establish base path for scripts
base_path = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/' \
            'mortgageResearch/'

# Establish base path for data
data_path = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/output/' \
            'combinedDefaultPred_logit/'
if not os.path.exists(data_path):
    os.makedirs(data_path)


# ---- ---- ----
# Order scripts for execution

# Order scripts into execution dictionary
scripts = {
    0: base_path + 'dataPrep/' + 'a_originationLoadPrep.py',
    1: base_path + 'dataPrep/' + 'a_monthlyLoadPrep.py',
    2: base_path + 'featureEngineering/' + 'combineFeatures.py',
    3: base_path + 'featureEngineering/' + 'b_combinedFeatures.py',
    4: base_path + 'model/' + 'comb_logisticRegression.py'
}

# Iteratively execute scripts for loadPrep, FE, model, residualanalysis
for i, s in scripts.iteritems():
    try:
        print "Running Script {}".format(str(i))
        print "     Filename: {}".format(str(s))
        subprocess.call(["Python", s, data_path, d_source, model_name])
    except:
        raise Exception(
            "Run failure on script {}, filename {}".format(
                str(i),str(s))
            )