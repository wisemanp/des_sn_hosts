import numpy as np

def power_law(ages_gyr, beta=1.14, norm=2.08e-13, tmin_gyr=0.04):
    t = np.clip(np.asarray(ages_gyr, float), tmin_gyr, None)
    return norm * t**(-beta)

def two_component(ages_gyr, f_prompt=0.1, tau_gyr=3.0, norm=1.0, tmin_gyr=0.04):
    t = np.clip(np.asarray(ages_gyr, float), tmin_gyr, None)
    return norm * (f_prompt * (t <= 0.5).astype(float) + (1 - f_prompt) * np.exp(-t / tau_gyr))

def broken_power_law(ages_gyr, beta1=1.0, beta2=1.5, t_break_gyr=0.5, norm=1.0, tmin_gyr=0.04):
    t = np.clip(np.asarray(ages_gyr, float), tmin_gyr, None)
    return np.where(t < t_break_gyr, t**(-beta1), (t_break_gyr**(-beta1)) * (t / t_break_gyr)**(-beta2)) * norm

DTD_MODELS = {"power_law": power_law, "two_component": two_component, "broken_power_law": broken_power_law}

def compute_age_dist(ages_gyr, model="power_law", **params):
    if model not in DTD_MODELS:
        raise ValueError(f"Unknown DTD model: {model}")
    return DTD_MODELS[model](ages_gyr, **params)