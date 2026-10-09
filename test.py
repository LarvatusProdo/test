"""
Calcul du géopotentiel et de l'altitude sur les niveaux modèles ERA5 (hybrides sigma-pression).

Méthode : intégration de l'équation hydrostatique depuis la surface (niveau 137)
vers le sommet du modèle (niveau 1), suivant la formule officielle ECMWF
(compute_geopotential_on_ml.py / IFS Documentation Part III, Dynamics).

Entrées attendues (xarray.DataArray) :
    t     : température (K) sur les niveaux modèles, dim "level" (1..137)
    q     : humidité spécifique (kg/kg), mêmes dims que t
    z     : géopotentiel de surface (m2/s2), niveau 1 uniquement dans MARS/ERA5
    lnsp  : log de la pression de surface, niveau 1 uniquement

Sortie :
    xr.Dataset avec "z" (géopotentiel), "z_height" (hauteur géopotentielle, m)
    et "altitude" (altitude géométrique approx., m).
"""

import numpy as np
import xarray as xr

R_D = 287.06      # constante des gaz parfaits pour l'air sec (J/kg/K)
R_G = 9.80665     # gravité standard (m/s^2)
R_EARTH = 6371000.0  # rayon terrestre moyen (m)


def get_ab_coefficients(ds):
    """
    Extrait les coefficients hybrides A et B (demi-niveaux) depuis les
    métadonnées GRIB conservées par cfgrib dans ds.attrs["GRIB_pv"].

    Nécessite d'avoir ouvert le fichier avec engine="cfgrib".
    Retourne deux tableaux 1D de taille n_levels + 1.
    """
    if "GRIB_pv" not in ds.attrs:
        raise ValueError(
            "Coefficients A/B introuvables dans les attributs. "
            "Ouvre le fichier avec xr.open_dataset(..., engine='cfgrib') "
            "pour conserver la clé GRIB_pv, ou passe a_coef/b_coef manuellement."
        )
    pv = np.array(ds.attrs["GRIB_pv"])
    half = len(pv) // 2
    a_coef = pv[:half]
    b_coef = pv[half:]
    return a_coef, b_coef


def compute_geopotential_on_ml(t, q, z_surf, lnsp, level_dim="level",
                                a_coef=None, b_coef=None, target_levels=None):
    """
    Calcule le géopotentiel et l'altitude sur les niveaux modèles ERA5.

    Paramètres
    ----------
    t, q : xr.DataArray
        Température (K) et humidité spécifique (kg/kg) sur les niveaux modèles.
    z_surf : xr.DataArray
        Géopotentiel de surface (m2/s2), sans dim "level" ou niveau 1 seulement.
    lnsp : xr.DataArray
        Log de la pression de surface, sans dim "level" ou niveau 1 seulement.
    level_dim : str
        Nom de la dimension des niveaux modèles.
    a_coef, b_coef : array-like, optionnel
        Coefficients hybrides A et B. Extraits automatiquement depuis t si absents.
    target_levels : list[int], optionnel
        Niveaux à conserver en sortie (ex: [1] pour le seul niveau 1).
        Par défaut : tous les niveaux présents dans t.

    Retourne
    --------
    xr.Dataset : variables "z" (m2/s2), "z_height" (m), "altitude" (m).
    """
    t = t.sortby(level_dim)
    q = q.sortby(level_dim)
    levels = t[level_dim].values.astype(int)
    n_levels = int(levels.max())

    if a_coef is None or b_coef is None:
        a_coef, b_coef = get_ab_coefficients(t)

    a_coef = xr.DataArray(np.asarray(a_coef), dims="half_level",
                           coords={"half_level": np.arange(len(a_coef))})
    b_coef = xr.DataArray(np.asarray(b_coef), dims="half_level",
                           coords={"half_level": np.arange(len(b_coef))})

    sp = np.exp(lnsp)
    if level_dim in sp.dims:
        sp = sp.squeeze(level_dim, drop=True)
    if level_dim in z_surf.dims:
        z_surf = z_surf.squeeze(level_dim, drop=True)

    # Pression sur les demi-niveaux (0 = sommet du modèle, n_levels = surface)
    p_half = a_coef + b_coef * sp

    # Température virtuelle
    t_v = t * (1 + 0.609133 * q)

    phi_half_below = z_surf  # géopotentiel au demi-niveau de surface
    z_ml = {}

    # Intégration de la surface (k = n_levels) vers le sommet (k = 1)
    for k in range(n_levels, 0, -1):
        p_below = p_half.sel(half_level=k)
        p_above = p_half.sel(half_level=k - 1)
        tv_k = t_v.sel({level_dim: k})

        dlogp = np.log(p_below / p_above)

        if k == 1:
            alpha = np.log(2.0)  # cas particulier du sommet du modèle
        else:
            alpha = 1.0 - (p_above / (p_below - p_above)) * dlogp

        phi_k = phi_half_below + alpha * R_D * tv_k
        z_ml[k] = phi_k

        phi_half_above = phi_half_below + R_D * tv_k * dlogp
        phi_half_below = phi_half_above

    if target_levels is None:
        target_levels = levels.tolist()

    z_out = xr.concat(
        [z_ml[k].expand_dims({level_dim: [k]}) for k in sorted(target_levels)],
        dim=level_dim,
    )

    ds_out = xr.Dataset({"z": z_out})
    ds_out["z_height"] = ds_out["z"] / R_G
    ds_out["altitude"] = (ds_out["z_height"] * R_EARTH) / (R_EARTH - ds_out["z_height"])

    ds_out["z"].attrs["units"] = "m2 s-2"
    ds_out["z"].attrs["long_name"] = "Geopotential"
    ds_out["z_height"].attrs["units"] = "m"
    ds_out["z_height"].attrs["long_name"] = "Geopotential height"
    ds_out["altitude"].attrs["units"] = "m"
    ds_out["altitude"].attrs["long_name"] = "Approximate geometric altitude"

    return ds_out


if __name__ == "__main__":
    # Exemple d'utilisation avec des fichiers GRIB ouverts via cfgrib.
    # Adapter les noms de fichiers et le filtrage selon tes données.

    t = xr.open_dataset("t_ml.grib", engine="cfgrib")["t"]
    q = xr.open_dataset("q_ml.grib", engine="cfgrib")["q"]
    z = xr.open_dataset(
        "zlnsp.grib", engine="cfgrib",
        filter_by_keys={"shortName": "z"}
    )["z"]
    lnsp = xr.open_dataset(
        "zlnsp.grib", engine="cfgrib",
        filter_by_keys={"shortName": "lnsp"}
    )["lnsp"]

    # Ne calculer que le niveau 1 (sommet du modèle)
    result = compute_geopotential_on_ml(t, q, z, lnsp, target_levels=[1])

    print(result)
    print("Altitude approx. niveau 1 (m) :", result["altitude"].sel(level=1).values)
