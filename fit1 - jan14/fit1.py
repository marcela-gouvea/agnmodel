# import numpy, astropy and matplotlib for basic functionalities
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pkg_resources

# import agnpy classes
from agnpy.spectra import BrokenPowerLaw
from agnpy.fit import SynchrotronSelfComptonModel, load_gammapy_flux_points
from agnpy.utils.plot import load_mpl_rc, sed_y_label

load_mpl_rc()

# import gammapy classes
from gammapy.modeling.models import SkyModel
from gammapy.modeling import Fit

# electron energy distribution
n_e = BrokenPowerLaw(
    k=1e-8 * u.Unit("cm-3"),
    p1=1.4,
    p2=3.6,
    gamma_b=7e4,
    gamma_min=1,
    gamma_max=3e6,
)

# initialise the Gammapy SpectralModel
ssc_model = SynchrotronSelfComptonModel(n_e, backend="gammapy")

ssc_model.parameters["z"].value = 0.034
ssc_model.parameters["delta_D"].value = 20
ssc_model.parameters["t_var"].value = (1 * u.d).to_value("s")
ssc_model.parameters["t_var"].frozen = True
ssc_model.parameters["log10_B"].value = -0.468


sed_path = pkg_resources.resource_filename( __name__, "sedjan.ecsv")

systematics_dict = {
    "Fermi": 0.10,
}

# define minimum and maximum energy to be used in the fit
E_min = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
E_max = 100 * u.TeV

datasets = load_gammapy_flux_points(sed_path, E_min, E_max, systematics_dict)

sky_model = SkyModel(spectral_model=ssc_model, name="Mrk501")
datasets.models = [sky_model]

fig, ax = plt.subplots(figsize=(8, 6))

for dataset in datasets:
    dataset.data.plot(ax=ax, label=dataset.name)

ssc_model.plot(
    ax=ax,
    energy_bounds=[1e-6, 1e14] * u.eV,
    energy_power=2,
    label="SSC model",
    color="k",
    lw=1.6,
)

ax.set_ylabel(sed_y_label)
ax.set_xlabel(r"$E\,/\,{\rm eV}$")
ax.set_xlim([1e-6, 1e14])
ax.legend(ncol=4, fontsize=9)

plt.savefig("fig1.jpeg")