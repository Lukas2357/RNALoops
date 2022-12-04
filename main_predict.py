"""Main module to predict planar angles with neural network"""
import matplotlib as plt

from rnaloops.engineer.add_features import load_agg_df
from rnaloops.predict.apply_model import converge_model
from rnaloops.predict.prepare_model import prep_model

plt.use('TkAgg')

# Use 'parts_seq' or 'whole_sequence' as the input features:
cat = 'parts_seq'
agg = load_agg_df(cat=cat, way=3)

# Prepare the model and converge it, check source code for details:
model = prep_model(agg)
model = converge_model(model, log_epochs=100, max_epochs=1000)