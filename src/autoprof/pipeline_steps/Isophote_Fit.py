import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse as Photutils_Ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy, deepcopy
import logging

from ..autoprofutils.SharedFunctions import (
    _iso_extract_with_interp_cutoff,
    _iso_interpolate_radius,
    _isophote_band_limits,
    _has_enough_isophote_coverage,
    _interpolate_invalid_isophote_samples,
    _validate_interpolate_method,
    _photutils_masked_data,
    Read_Image,
    _DEFAULT_ELLIP_MAX,
    _resolve_ellip_max,
    _ellip_to_logq,
    _logq_to_ellip,
    _ellipse_shape,
    _perturb_ellipse,
    _x_to_pa,
    _x_to_eps,
    _inv_x_to_eps,
    _inv_x_to_pa,
    Angle_TwoAngles_sin,
    Angle_TwoAngles_cos,
    Angle_Scatter,
    LSBImage,
    AddLogo,
    PA_shift_convention,
    autocolours,
)
from ..autoprofutils.Diagnostic_Plots import Plot_Isophote_Fit

__all__ = (
    "Photutils_Fit",
    "Isophote_Fit_FixedPhase",
    "Isophote_Fit_FFT_Robust",
    "Isophote_Fit_Forced",
    "Isophote_Fit_FFT_mean",
)


def _finite_isophote_samples(isovals):
    isovals = np.asarray(isovals)
    return isovals[np.isfinite(isovals)]


def _best_finite_index(losses):
    losses = np.asarray(losses, dtype=float)
    if not np.any(np.isfinite(losses)):
        return 0
    return int(np.argmin(np.where(np.isfinite(losses), losses, np.inf)))


def _max_fit_mode(fit_coefs=None):
    return max(fit_coefs) if fit_coefs is not None and len(fit_coefs) > 0 else 2


def _has_enough_sample_count(choose, max_mode=2):
    return np.sum(choose) > 2 * max_mode


def _has_enough_isophote_samples(
    dat,
    radius,
    params,
    center,
    mask,
    max_mode=2,
    interp_method="bilinear",
    rad_interp=None,
    variance=None,
):
    if variance is not None:
        return _harmonic_ring_operator(dat, radius, params, center, variance, mask) is not None
    kwargs = {}
    if interp_method is not None:
        kwargs["interp_method"] = interp_method
    if rad_interp is not None:
        kwargs["rad_interp"] = rad_interp
    _, theta, choose, _ = _iso_extract_with_interp_cutoff(
        dat,
        radius,
        params,
        center,
        more=True,
        mask=mask,
        return_choose=True,
        **kwargs,
    )
    return (
        _has_enough_sample_count(choose, max_mode=max_mode)
        and _has_enough_isophote_coverage(theta, choose)
    )


def _extract_fft_isophote_samples(
    dat,
    radius,
    params,
    center,
    mask,
    max_mode=2,
    interp_method="bilinear",
    rad_interp=None,
):
    kwargs = {}
    if interp_method is not None:
        kwargs["interp_method"] = interp_method
    if rad_interp is not None:
        kwargs["rad_interp"] = rad_interp
    flux, theta, choose, _ = _iso_extract_with_interp_cutoff(
        dat,
        radius,
        params,
        center,
        more=True,
        mask=mask,
        return_choose=True,
        **kwargs,
    )
    has_enough_samples = (
        _has_enough_sample_count(choose, max_mode=max_mode)
        and _has_enough_isophote_coverage(theta, choose)
    )
    if mask is not None:
        if not has_enough_samples:
            return None
        flux, theta = _interpolate_invalid_isophote_samples(flux, theta, choose)
    elif not has_enough_samples:
        return None
    elif not np.all(choose):
        flux = flux[choose]
    return _finite_isophote_samples(flux)


def _active_neighbors(i, active_mask, n):
    if active_mask is None:
        return [j for j in (i + 1, i - 1) if 0 <= j < n]

    active_indices = np.flatnonzero(active_mask)
    neighbors = []
    inner = active_indices[active_indices < i]
    outer = active_indices[active_indices > i]
    if len(outer) > 0:
        neighbors.append(outer[0])
    if len(inner) > 0:
        neighbors.append(inner[-1])
    return neighbors


def _interpolate_inactive_parameters(
    sample_radii, parameters, active_mask, ellip_max=_DEFAULT_ELLIP_MAX
):
    active_indices = np.flatnonzero(active_mask)
    inactive_indices = np.flatnonzero(np.logical_not(active_mask))
    if len(active_indices) < 2 or len(inactive_indices) == 0:
        return

    active_radii = sample_radii[active_indices]
    active_pa = np.array(list(parameters[i]["pa"] for i in active_indices))
    active_ellip = np.array(list(parameters[i]["ellip"] for i in active_indices))
    pa_s = np.sin(2 * active_pa)
    pa_c = np.cos(2 * active_pa)
    inv_ellip = _ellip_to_logq(active_ellip, ellip_max)

    for i in inactive_indices:
        radius = sample_radii[i]
        parameters[i]["ellip"] = _logq_to_ellip(np.interp(radius, active_radii, inv_ellip), ellip_max)
        parameters[i]["pa"] = _x_to_pa(
            ((np.arctan2(np.interp(radius, active_radii, pa_s), np.interp(radius, active_radii, pa_c))) % (2 * np.pi))
            / 2
        )
        if parameters[i]["C"] is not None:
            active_C = np.array(list(parameters[j]["C"] for j in active_indices))
            parameters[i]["C"] = np.interp(radius, active_radii, active_C)
        if parameters[i]["m"] is not None:
            for m in range(len(parameters[i]["m"])):
                active_Am = np.array(list(parameters[j]["Am"][m] for j in active_indices))
                active_Phim = np.array(list(parameters[j]["Phim"][m] for j in active_indices))
                mode = parameters[i]["m"][m]
                parameters[i]["Am"][m] = np.interp(radius, active_radii, active_Am)
                phase_s = np.interp(radius, active_radii, np.sin(mode * active_Phim))
                phase_c = np.interp(radius, active_radii, np.cos(mode * active_Phim))
                parameters[i]["Phim"][m] = ((np.arctan2(phase_s, phase_c)) % (2 * np.pi)) / mode


def _activate_inactive_radii(
    dat,
    sample_radii,
    parameters,
    active_mask,
    center,
    mask,
    max_mode=2,
    interp_method="bilinear",
    rad_interp=None,
    ellip_max=_DEFAULT_ELLIP_MAX,
    variance=None,
):
    _interpolate_inactive_parameters(sample_radii, parameters, active_mask, ellip_max)
    activated = []
    for i in np.flatnonzero(np.logical_not(active_mask)):
        if _has_enough_isophote_samples(
            dat,
            sample_radii[i],
            parameters[i],
            center,
            mask,
            max_mode=max_mode,
            interp_method=interp_method,
            rad_interp=rad_interp,
            variance=variance,
        ):
            active_mask[i] = True
            activated.append(i)
    return activated


def _activate_inactive_ellipse_radii(
    dat,
    sample_radii,
    ellip,
    pa,
    active_mask,
    center,
    mask,
    interp_method="bilinear",
    rad_interp=None,
    ellip_max=_DEFAULT_ELLIP_MAX,
):
    active_indices = np.flatnonzero(active_mask)
    inactive_indices = np.flatnonzero(np.logical_not(active_mask))
    if len(active_indices) < 2 or len(inactive_indices) == 0:
        return []

    active_radii = sample_radii[active_indices]
    pa_s = np.sin(2 * pa[active_indices])
    pa_c = np.cos(2 * pa[active_indices])
    inv_ellip = _ellip_to_logq(ellip[active_indices], ellip_max)
    activated = []
    for i in inactive_indices:
        radius = sample_radii[i]
        ellip[i] = _logq_to_ellip(np.interp(radius, active_radii, inv_ellip), ellip_max)
        pa[i] = _x_to_pa(
            ((np.arctan2(np.interp(radius, active_radii, pa_s), np.interp(radius, active_radii, pa_c))) % (2 * np.pi))
            / 2
        )
        if _has_enough_isophote_samples(
            dat,
            radius,
            {"ellip": ellip[i], "pa": pa[i]},
            center,
            mask,
            interp_method=interp_method,
            rad_interp=rad_interp,
        ):
            active_mask[i] = True
            activated.append(i)
    return activated


def _initial_ellipse_profile(radii, results, ellip_max):
    """Interpolate sparse initial shapes in log radius, holding endpoints fixed."""
    radii = np.asarray(radii, dtype=float)
    if "init profile R" not in results:
        return (
            np.full(radii.shape, np.clip(results["init ellip"], 0.0, ellip_max)),
            np.full(radii.shape, results["init pa"]),
        )
    anchors = np.asarray(results["init profile R"], dtype=float)
    ellip = np.asarray(results["init profile ellip"], dtype=float)
    pa = np.asarray(results["init profile pa"], dtype=float)
    if (anchors.ndim != 1 or not len(anchors) or ellip.shape != anchors.shape
            or pa.shape != anchors.shape or not np.all(np.isfinite(anchors))
            or not np.all(np.isfinite(ellip)) or not np.all(np.isfinite(pa))
            or np.any(anchors <= 0) or np.any(np.diff(anchors) <= 0)):
        raise ValueError("Initial geometry profile requires finite, ordered positive radii and matching shapes.")
    shape = _ellipse_shape(np.clip(ellip, 0.0, ellip_max), pa)
    x, xp = np.log(radii), np.log(anchors)
    trial = np.interp(x, xp, shape.real) + 1j * np.interp(x, xp, shape.imag)
    return np.clip(-np.expm1(-np.abs(trial)), 0.0, ellip_max), (np.angle(trial) / 2) % np.pi


def _determine_sample_radii(
    dat,
    IMG,
    results,
    options,
    scale,
    mask,
    startR,
    fit_limit_default,
    average_func,
    interp_method="bilinear",
    rad_interp=None,
):
    """Select radii up to, but excluding, the first below-threshold sample."""
    ellip_max = _resolve_ellip_max(options)
    shrink = 0
    while shrink < 5:
        sample_radii = []
        radius = startR
        kwargs = {"interp_method": interp_method}
        if rad_interp is not None:
            kwargs["rad_interp"] = rad_interp
        while radius < (max(IMG.shape) / 2):
            ellip, pa = _initial_ellipse_profile(radius, results, ellip_max)
            isovals = _finite_isophote_samples(
                _iso_extract_with_interp_cutoff(
                    dat,
                    radius,
                    {"ellip": ellip, "pa": pa},
                    results["center"],
                    more=False,
                    mask=mask,
                    **kwargs,
                )
            )
            if (
                len(isovals) > 0
                and average_func(isovals)
                < (options["ap_fit_limit"] if "ap_fit_limit" in options else fit_limit_default)
                * results["background noise"]
            ):
                break
            sample_radii.append(radius)
            radius *= 1.0 + scale / (1.0 + shrink)
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception(
            "Unable to initialize ellipse fit, check diagnostic plots. Possible missed center."
        )
    return np.array(sample_radii)


def Photutils_Fit(IMG, results, options):
    """Photutils elliptical isophote wrapper.

    This simply gives users access to the photutils isophote
    fitting method. See: `photutils
    <https://photutils.readthedocs.io/en/stable/isophote.html>`_ for
    more information.

    Notes
    ----------
    :References:
    - 'background'
    - 'center'
    - 'init R'
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }
    """

    dat = IMG - results["background"]
    geo = EllipseGeometry(
        x0=results["center"]["x"],
        y0=results["center"]["y"],
        sma=results["init R"] / 2,
        eps=results["init ellip"],
        pa=results["init pa"],
    )
    photutils_dat = _photutils_masked_data(dat, results)
    ellipse = Photutils_Ellipse(photutils_dat, geometry=geo)

    isolist = ellipse.fit_image(fix_center=True, linear=False)
    res = {
        "fit R": isolist.sma[1:],
        "fit ellip": isolist.eps[1:],
        "fit ellip_err": isolist.ellip_err[1:],
        "fit pa": isolist.pa[1:],
        "fit pa_err": isolist.pa_err[1:],
        "fit photutils isolist": isolist,
        "auxfile fitlimit": (
            "fit limit semi-major axis: %.2f pix" % isolist.sma[-1]
            if len(isolist.sma) > 0
            else "fit limit semi-major axis: no valid isophotes"
        ),
    }

    if "ap_doplot" in options and options["ap_doplot"] and len(res["fit R"]) > 0:
        Plot_Isophote_Fit(
            dat,
            res["fit R"],
            res["fit ellip"],
            res["fit pa"],
            res["fit ellip_err"],
            res["fit pa_err"],
            results,
            options,
        )

    return IMG, res


def Isophote_Fit_FixedPhase(IMG, results, options):
    """Simply applies fixed position angle and ellipticity at the initialization values.

    Parameters
    -----------------
    ap_iso_ellip_max : float, default 0.95
      Maximum ellipticity (1 - b/a), strictly between 0 and 1.
      Shared by native initialization, fitting, uncertainties, and interpolation.

    ap_scale : float, default 0.2
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

    ap_fit_limit : float, default 1
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 1, smaller values will end
      fitting further out in the galaxy image.

    ap_isofit_interpolate_method : string, default 'bilinear'
      Select method for flux interpolation while determining the
      fitted isophote radii. Options are 'lanczos', 'bicubic', and
      'bilinear'.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'mask' (optional)
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }

    """

    ellip_max = _resolve_ellip_max(options)
    if "ap_scale" in options:
        scale = options["ap_scale"]
    else:
        scale = 0.2
    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None
    interp_method = (
        _validate_interpolate_method(
            options["ap_isofit_interpolate_method"],
            "ap_isofit_interpolate_method",
        )
        if "ap_isofit_interpolate_method" in options
        else "bilinear"
    )
    rad_interp = _iso_interpolate_radius(options, results)

    # Determine sampling radii
    ######################################################################
    sample_radii = _determine_sample_radii(
        dat,
        IMG,
        results,
        options,
        scale,
        mask,
        max(1.0, results["psf fwhm"] / 2),
        1,
        np.median,
        interp_method=interp_method,
        rad_interp=rad_interp,
    )
    ellip, pa = _initial_ellipse_profile(sample_radii, results, ellip_max)
    logging.debug("%s: sample radii: %s" % (options["ap_name"], str(sample_radii)))

    res = {
        "fit ellip": ellip,
        "fit pa": pa,
        "fit R": sample_radii,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % sample_radii[-1],
    }

    if "init ellip_err" in results:
        res["fit ellip_err"] = np.ones(len(sample_radii)) * results["init ellip_err"]
    if "init pa_err" in results:
        res["fit pa_err"] = np.ones(len(sample_radii)) * results["init pa_err"]

    return IMG, res


def _ellip_smooth(R, E, deg, ellip_max=_DEFAULT_ELLIP_MAX):
    model = make_pipeline(PolynomialFeatures(deg), HuberRegressor(epsilon=2.0))
    model.fit(np.log10(R).reshape(-1, 1), _inv_x_to_eps(E))
    return np.clip(_x_to_eps(model.predict(np.log10(R).reshape(-1, 1))), 0.0, ellip_max)


def _pa_smooth(R, PA, deg):

    model_s = make_pipeline(PolynomialFeatures(deg), HuberRegressor())
    model_c = make_pipeline(PolynomialFeatures(deg), HuberRegressor())
    model_c.fit(np.log10(R).reshape(-1, 1), np.cos(2 * PA))
    model_s.fit(np.log10(R).reshape(-1, 1), np.sin(2 * PA))
    pred_pa_s = np.clip(model_s.predict(np.log10(R).reshape(-1, 1)), a_min=-1, a_max=1)
    pred_pa_c = np.clip(model_c.predict(np.log10(R).reshape(-1, 1)), a_min=-1, a_max=1)

    #  np.arctan(pred_pa_s / pred_pa_c) + (np.pi * (pred_pa_c < 0))
    return ((np.arctan2(pred_pa_s, pred_pa_c)) % (2 * np.pi)) / 2


def _isofit_variance(dat, noise, options):
    """Resolve total variance from an RMS map or background plus optional source noise."""
    rms = options.get("ap_isofit_rms")
    if rms is None:
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("Harmonic fitting requires positive background noise or ap_isofit_rms.")
        variance = np.full(dat.shape, noise**2, dtype=float)
        gain = options.get("ap_isofit_gain")
        if gain is not None:
            gain = np.asarray(gain, dtype=float)
            if gain.ndim != 0 or not np.isfinite(gain) or gain <= 0:
                raise ValueError("ap_isofit_gain must be a finite positive scalar.")
            # dat is already background-subtracted; masked pixels remain excluded.
            variance += np.maximum(dat, 0.0) / gain
        return variance
    if isinstance(rms, str):
        rms = Read_Image(rms, {})
    rms = np.asarray(rms, dtype=float)
    if rms.shape != dat.shape:
        raise ValueError("ap_isofit_rms must have the same shape as the image.")
    # Nonpositive/nonfinite RMS pixels are excluded, not assigned infinite weight.
    return np.where(np.isfinite(rms) & (rms > 0), rms**2, np.nan)


def _set_isofit_bands(dat, radii, parameters, center, noise, variance, mask, threshold):
    """Fix band membership from initial brightness and harmonic-mean boundaries."""
    lower, upper = _isophote_band_limits(radii)
    invalid = ~np.isfinite(variance) | (variance <= 0)
    if mask is not None:
        invalid |= mask
    for i, radius in enumerate(radii):
        flux, _, choose, _ = _iso_extract_with_interp_cutoff(
            dat, radius, parameters[i], center, more=True, mask=invalid,
            return_choose=True, interp_method="bilinear",
        )
        if np.any(choose) and np.median(flux[choose]) < threshold * noise:
            # Relative boundaries also move with the shifted uncertainty rings.
            parameters[i]["_isoband"] = (lower[i] / radius, upper[i] / radius)


def _harmonic_band_operator(dat, radius, params, center, variance, mask):
    """Fit polar harmonics to sector means of distinct, unmasked band pixels."""
    q, pa = 1 - params["ellip"], params["pa"]
    lower, upper = radius * np.asarray(params["_isoband"])
    cp, sp = np.cos(pa), np.sin(pa)
    dx = upper * np.hypot(cp, q * sp)
    dy = upper * np.hypot(sp, q * cp)
    x0, x1 = max(0, int(np.floor(center["x"] - dx))), min(dat.shape[1], int(np.ceil(center["x"] + dx)) + 1)
    y0, y1 = max(0, int(np.floor(center["y"] - dy))), min(dat.shape[0], int(np.ceil(center["y"] + dy)) + 1)
    xx = np.arange(x0, x1)[None, :] - center["x"]
    yy = np.arange(y0, y1)[:, None] - center["y"]
    x, y = xx * cp + yy * sp, -xx * sp + yy * cp
    rho = np.hypot(x, y / q)
    image = dat[y0:y1, x0:x1]
    var = variance[y0:y1, x0:x1]
    good = (rho >= lower) & (rho < upper) & np.isfinite(image) & np.isfinite(var) & (var > 0)
    if mask is not None:
        good &= ~mask[y0:y1, x0:x1]
    iy, ix = np.nonzero(good)
    if len(ix) < 8:
        return None
    x, y = x[good], y[good]
    # Uniform eccentric-anomaly sectors have equal area before masking.
    n = max(15, int(0.9 * 2 * np.pi * radius))
    theta = np.arctan2(y / q, x) % (2 * np.pi)
    bins = (np.floor(theta * n / (2 * np.pi) + 0.5).astype(int)) % n
    counts = np.bincount(bins, minlength=n)
    occupied = np.flatnonzero(counts)
    if len(occupied) < 8 or len(occupied) / n < 0.35:
        return None
    angles = occupied * 2 * np.pi / n
    if np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])) > np.pi + 2 * np.pi / n:
        return None
    phi = np.arctan2(y, x)
    # Average the harmonic basis over the same pixels as each sector mean.
    basis = (np.ones(len(phi)), np.cos(phi), np.sin(phi), np.cos(2 * phi), np.sin(2 * phi))
    design = np.column_stack([
        np.bincount(bins, weights=b, minlength=n)[occupied] / counts[occupied] for b in basis
    ])
    gram = design.T @ design
    eigen = np.linalg.eigvalsh(gram)
    if eigen[0] <= 1e-6 * eigen[-1]:
        return None
    harmonic = np.linalg.solve(gram, design.T)[3:5]
    operator = harmonic[:, np.searchsorted(occupied, bins)] / counts[bins]
    pixels = (iy + y0) * dat.shape[1] + ix + x0
    covariance = (operator * variance.ravel()[pixels]) @ operator.T
    eigen = np.linalg.eigvalsh(covariance)
    if eigen[0] <= 1e-10 * eigen[-1]:
        return None
    return pixels, operator, covariance


def _harmonic_ring_operator(dat, radius, params, center, variance, mask):
    """Fit polar-angle harmonics on a ring or its configured mean-sampled band."""
    ellip, pa = params["ellip"], params["pa"]
    if not np.all(np.isfinite((radius, ellip, pa))) or radius <= 0:
        return None
    if "_isoband" in params:
        return _harmonic_band_operator(dat, radius, params, center, variance, mask)
    n = max(15, int(0.9 * 2 * np.pi * radius))
    theta = np.arange(n) * 2 * np.pi / n
    height, width = dat.shape
    pixel_variance = variance.ravel()
    x, y = radius * np.cos(theta), radius * (1 - ellip) * np.sin(theta)
    xp = center["x"] + x * np.cos(pa) - y * np.sin(pa)
    yp = center["y"] + x * np.sin(pa) + y * np.cos(pa)
    ix, iy = np.floor(xp).astype(int), np.floor(yp).astype(int)
    good = (ix >= 0) & (iy >= 0) & (ix + 1 < width) & (iy + 1 < height)
    dx, dy = xp - ix, yp - iy
    # Safe indexing only; samples outside the image remain excluded.
    ix, iy = np.clip(ix, 0, width - 2), np.clip(iy, 0, height - 2)
    ids = np.stack((iy * width + ix, iy * width + ix + 1,
                    (iy + 1) * width + ix, (iy + 1) * width + ix + 1), axis=1)
    weights = np.stack(((1 - dx) * (1 - dy), dx * (1 - dy),
                        (1 - dx) * dy, dx * dy), axis=1)
    bad = (~np.isfinite(pixel_variance[ids]) | (pixel_variance[ids] <= 0)
           | ~np.isfinite(dat.ravel()[ids]))
    if mask is not None:
        bad |= mask.ravel()[ids]
    good &= ~np.any(bad, axis=1)
    k = int(np.sum(good))
    if k < 8 or k / n < 0.35:
        return None
    angles = theta[good]
    if np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])) > np.pi + 2 * np.pi / n:
        return None
    # Keep the sampling and coverage in eccentric anomaly; fit in polar angle.
    angles = np.arctan2(y[good], x[good])
    design = np.column_stack((np.ones(k), np.cos(angles), np.sin(angles),
                              np.cos(2 * angles), np.sin(2 * angles)))
    gram = design.T @ design
    eigen = np.linalg.eigvalsh(gram)
    if eigen[0] <= 1e-6 * eigen[-1]:
        return None
    harmonic = np.linalg.solve(gram, design.T)[3:5]
    ids, weights = ids[good], weights[good]
    coeff = (harmonic[:, :, None] * weights[None, :, :]).reshape(2, -1)
    pixels, inverse = np.unique(ids.ravel(), return_inverse=True)
    operator = np.array([np.bincount(inverse, weights=row, minlength=len(pixels)) for row in coeff])
    covariance = (operator * pixel_variance[pixels]) @ operator.T
    eigen = np.linalg.eigvalsh(covariance)
    if eigen[0] <= 1e-10 * eigen[-1]:
        return None
    return pixels, operator, covariance


def _harmonic_shape_loss(dat, R, parameters, i, center, variance, mask, reg_scale, active_mask):
    fitted = _harmonic_ring_operator(dat, R[i], parameters[i], center, variance, mask)
    if fitted is None:
        return np.inf
    pixels, operator, covariance = fitted
    harmonics = operator @ dat.ravel()[pixels]
    q2 = harmonics @ np.linalg.solve(covariance, harmonics)
    shape = _ellipse_shape(parameters[i]["ellip"], parameters[i]["pa"])
    penalty = sum(
        abs(shape - _ellipse_shape(parameters[j]["ellip"], parameters[j]["pa"])) / abs(i - j)
        for j in _active_neighbors(i, active_mask, len(R))
    )
    raw = q2 + reg_scale * penalty
    return float(raw / (1 + raw)) if np.isfinite(raw) else np.inf


def _FFT_Robust_loss(
    dat,
    R,
    PARAMS,
    i,
    C,
    noise,
    mask=None,
    reg_scale=1.0,
    robust_clip=0.15,
    fit_coefs=None,
    name="",
    active_mask=None,
    interp_method="bilinear",
    rad_interp=None,
    variance=None,
):
    if variance is not None:
        return _harmonic_shape_loss(dat, R, PARAMS, i, C, variance, mask, reg_scale, active_mask)
    if fit_coefs is not None and len(fit_coefs) == 0:
        fit_coefs = None
    max_mode = _max_fit_mode(fit_coefs)

    isovals = _extract_fft_isophote_samples(
        dat,
        R[i],
        PARAMS[i],
        C,
        mask,
        max_mode=max_mode,
        interp_method=interp_method,
        rad_interp=rad_interp,
    )
    if isovals is None:
        return np.inf

    f2_loss = None
    if len(isovals) > 2 * max_mode:
        try:
            coefs = fft(np.clip(isovals, a_max=np.quantile(isovals, 1.0 - robust_clip), a_min=None))
            norm = len(isovals) * (max(0, np.median(isovals)) + noise / np.sqrt(len(isovals)))
            if fit_coefs is None:
                f2_loss = np.abs(coefs[2]) / norm
            else:
                f2_loss = np.sum(np.abs(coefs[np.array(fit_coefs)])) / (len(fit_coefs) * norm)
        except Exception:
            f2_loss = None

    reg_loss = 0
    shape = _ellipse_shape(PARAMS[i]["ellip"], PARAMS[i]["pa"])
    has_fmodes = PARAMS[i]["m"] is not None and len(PARAMS[i]["m"]) > 0
    if has_fmodes:
        fmode_scale = 1.0 / len(PARAMS[i]["m"])
    for j in _active_neighbors(i, active_mask, len(R)):
        if any(p["C"] is not None or p["m"] is not None for p in (PARAMS[i], PARAMS[j])):
            # A round base ellipse can still have an oriented distortion.
            # Keep the existing penalty until the full shape is represented.
            neighbor_reg_loss = abs(
                (PARAMS[i]["ellip"] - PARAMS[j]["ellip"]) / (1 - PARAMS[j]["ellip"])
            )
            neighbor_reg_loss += abs(
                Angle_TwoAngles_sin(PARAMS[i]["pa"], PARAMS[j]["pa"]) / 0.2
            )
        else:
            neighbor_reg_loss = abs(shape - _ellipse_shape(PARAMS[j]["ellip"], PARAMS[j]["pa"]))
        if has_fmodes and PARAMS[j]["m"] is not None:
            for m in range(len(PARAMS[i]["m"])):
                neighbor_reg_loss += fmode_scale * abs(
                    (PARAMS[i]["Am"][m] - PARAMS[j]["Am"][m]) / 0.2
                )
                neighbor_reg_loss += fmode_scale * abs(
                    Angle_TwoAngles_cos(
                        PARAMS[i]["m"][m] * PARAMS[i]["Phim"][m],
                        PARAMS[j]["m"][m] * PARAMS[j]["Phim"][m],
                    )
                    / (PARAMS[i]["m"][m] * 0.1)
                )
        if PARAMS[i]["C"] is not None and PARAMS[j]["C"] is not None:
            neighbor_reg_loss += abs(np.log10(PARAMS[i]["C"] / PARAMS[j]["C"])) / 0.1
        # The geometric radius grid makes index separation proportional to log-radius distance.
        reg_loss += neighbor_reg_loss / abs(i - j)

    if f2_loss is None or not np.isfinite(f2_loss):
        return np.inf
    return f2_loss * (1 + reg_loss * reg_scale)


def _check_ellipse_directions(parameters, active_mask, loss, ellip_max, step):
    """Try the inner neighbor, then refine directions until a radius improves.

    Cache directions in pi/32 ticks; refine at offsets 22.5, 11.25 and 5.625
    degrees, using at most fourteen trials per length. The floor is .15/256.
    """
    changed = 0
    accepted_steps = {}
    diagonal = 1 / np.sqrt(2)
    ticks = (0, 32, 16, 48, 8, 56, 24, 40)
    directions = {k: (np.cos(k * np.pi / 32), np.sin(k * np.pi / 32)) for k in range(64)}
    # Preserve the original axis/diagonal values and tie-breaking order.
    directions.update(
        zip(
            ticks,
            (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (diagonal, diagonal), (diagonal, -diagonal),
                (-diagonal, diagonal), (-diagonal, -diagonal),
            ),
        )
    )
    for i in np.flatnonzero(active_mask):
        current_loss = loss(parameters, i)
        if not np.isfinite(current_loss):
            active_mask[i] = False
            changed += 1
            continue
        trial = list(parameters)
        radius_changed = False
        inner = np.flatnonzero(active_mask[:i])
        if len(inner):
            j = inner[-1]
            trial[i] = dict(parameters[i])
            trial[i]["ellip"] = parameters[j]["ellip"]
            trial[i]["pa"] = parameters[j]["pa"]
            candidate_loss = loss(trial, i)
            if np.isfinite(candidate_loss) and candidate_loss < current_loss:
                parameters[i] = trial[i]
                parameters[i].pop("loss", None)
                current_loss = candidate_loss
                radius_changed = True
        for scale in step * 2.0 ** np.arange(0, -9, -1):
            if scale < 0.15 / 256:
                break
            cache = {}

            def poll(tick):
                tick %= 64
                if tick not in cache:
                    dx, dy = directions[tick]
                    trial[i] = dict(parameters[i])
                    trial[i]["ellip"], trial[i]["pa"] = _perturb_ellipse(
                        parameters[i]["ellip"], parameters[i]["pa"], (scale * dx, scale * dy), ellip_max
                    )
                    value = loss(trial, i)
                    cache[tick] = (value if np.isfinite(value) else np.inf, trial[i])

            for tick in ticks:
                poll(tick)
            # Refine even when all eight initial trials fail to improve.
            for offset in (4, 2, 1):
                best_tick = min(cache, key=lambda tick: cache[tick][0])
                if not np.isfinite(cache[best_tick][0]):
                    break
                poll(best_tick - offset)
                poll(best_tick + offset)
            best_loss, best_params = min(cache.values(), key=lambda value: value[0])
            if best_loss < current_loss:
                parameters[i] = best_params
                accepted_steps[int(i)] = float(scale)
                parameters[i].pop("loss", None)
                radius_changed = True
                break
        changed += int(radius_changed)
    return changed, accepted_steps


def _harmonic_geometry_errors(dat, R, parameters, center, variance, mask, ellip_max):
    """Propagate harmonic covariance at fixed radii, without regularization.

    Finite log-q shape steps resolve whole-pixel band membership. NaN denotes
    an unsupported response, a boundary-limited estimate, or undefined PA.
    These local linear errors do not include systematic fitting offsets.
    """
    ellip_err = np.full(len(R), np.nan)
    pa_err = np.full(len(R), np.nan)
    step = 0.03
    logq_max = -np.log1p(-ellip_max)
    for i, (radius, params) in enumerate(zip(R, parameters)):
        sampled = _harmonic_ring_operator(dat, radius, params, center, variance, mask)
        if sampled is None:
            continue
        shape = _ellipse_shape(params["ellip"], params["pa"])
        z = np.array([shape.real, shape.imag])
        u = np.linalg.norm(z)
        if u <= 1e-10:
            continue  # The transformation to ellipticity/PA is singular at a circle.
        columns = []
        for axis in np.eye(2):
            harmonics = []
            for sign in (-1, 1):
                trial = z + sign * step * axis
                magnitude = np.linalg.norm(trial)
                if magnitude >= logq_max:
                    break
                trial_params = dict(params, ellip=-np.expm1(-magnitude),
                                    pa=(np.arctan2(trial[1], trial[0]) / 2) % np.pi)
                perturbed = _harmonic_ring_operator(
                    dat, radius, trial_params, center, variance, mask
                )
                if perturbed is None:
                    break
                pixels, operator, _ = perturbed
                harmonics.append(operator @ dat.ravel()[pixels])
            if len(harmonics) != 2:
                break
            columns.append((harmonics[1] - harmonics[0]) / (2 * step))
        if len(columns) != 2:
            continue
        jacobian = np.column_stack(columns)
        information = jacobian.T @ np.linalg.solve(sampled[2], jacobian)
        eigen = np.linalg.eigvalsh(information)
        if not np.all(np.isfinite(eigen)) or eigen[0] <= 1e-10 * max(eigen[-1], 0.0):
            continue
        covariance = np.linalg.inv(information)
        transform = np.array([np.exp(-u) * z / u, 0.5 * np.array([-z[1], z[0]]) / u**2])
        errors = np.sqrt(np.maximum(np.diag(transform @ covariance @ transform.T), 0.0))
        ellip_err[i] = errors[0]
        # A 68% joint shape region containing a circle cannot constrain PA.
        if z @ information @ z > 2.30:
            pa_err[i] = errors[1]
    return ellip_err, pa_err


def _FFT_Robust_Errors(
    dat,
    R,
    PARAMS,
    C,
    noise,
    mask=None,
    reg_scale=1.0,
    robust_clip=0.15,
    fit_coefs=None,
    name="",
    interp_method="bilinear",
    rad_interp=None,
    ellip_max=_DEFAULT_ELLIP_MAX,
    variance=None,
):
    if variance is not None:
        return _harmonic_geometry_errors(dat, R, PARAMS, C, variance, mask, ellip_max)

    PARAMS = [dict(p, ellip=np.clip(p["ellip"], 0.0, ellip_max)) for p in PARAMS]
    PA_err = np.zeros(len(R))
    E_err = np.zeros(len(R))
    for ri in range(len(R)):
        temp_fits = []
        x0 = np.array([PARAMS[ri]["ellip"], PARAMS[ri]["pa"]], dtype=float)
        if not np.all(np.isfinite(x0)):
            continue
        for i in range(10):
            low_ri = max(0, ri - 1)
            high_ri = min(len(R) - 1, ri + 1)
            trial_R = [R[low_ri], R[ri] * (1 - 0.05 + i * 0.1 / 9), R[high_ri]]

            def raw_loss(x):
                if not np.all(np.isfinite(x)):
                    return np.inf
                trial_params = deepcopy(PARAMS[ri])
                trial_params["ellip"] = float(x[0])
                trial_params["pa"] = float(x[1])
                return _FFT_Robust_loss(
                    dat,
                    trial_R,
                    [
                        PARAMS[low_ri],
                        trial_params,
                        PARAMS[high_ri],
                    ],
                    1,
                    C,
                    noise,
                    mask=mask,
                    reg_scale=reg_scale,
                    robust_clip=robust_clip,
                    fit_coefs=fit_coefs,
                    name=name,
                    interp_method=interp_method,
                    rad_interp=rad_interp,
                    variance=variance,
                )

            if not np.isfinite(raw_loss(x0)):
                continue

            # SLSQP expects a finite scalar objective even when a shifted isophote is unusable.
            def finite_loss(x):
                loss = raw_loss(x)
                return float(loss) if np.isfinite(loss) else 1e30

            try:
                method = "SLSQP"
                fit_options = {"ftol": 0.001}
                if "_isoband" in PARAMS[ri]:
                    # Pixel-centre band membership makes numerical gradients flat.
                    method = "Nelder-Mead"
                    fit_options = {"xatol": 1e-4, "fatol": 0.001, "maxiter": 150,
                                   "initial_simplex": np.array([x0] + [
                                       _perturb_ellipse(*x0, step, ellip_max)
                                       for step in ((0.03, 0.0), (0.0, 0.03))])}
                fit = minimize(
                    finite_loss,
                    x0=x0,
                    method=method,
                    bounds=[(0, ellip_max), (0, np.pi)],
                    options=fit_options,
                )
            except Exception:
                continue
            if np.all(np.isfinite(fit.x)) and np.isfinite(raw_loss(fit.x)):
                temp_fits.append(fit.x)
        if len(temp_fits) >= 2:
            temp_fits = np.array(temp_fits)
            E_err[ri] = iqr(temp_fits[:, 0], rng=[16, 84]) / 2
            PA_err[ri] = (
                Angle_Scatter(2 * (temp_fits[:, 1] % np.pi)) / 4.0
            )  # multiply by 2 to get [0, 2pi] range
        elif len(temp_fits) == 1:
            E_err[ri] = abs(float(temp_fits[0][0]) - PARAMS[ri]["ellip"])
            PA_err[ri] = (
                abs(Angle_TwoAngles_sin(2 * temp_fits[0][1], 2 * PARAMS[ri]["pa"])) / 2
            )
    return E_err, PA_err


def Isophote_Fit_FFT_Robust(IMG, results, options):
    """Fit isophotes with a harmonic objective and joint-shape regularization.

    Ordinary bilinear ellipse fits minimize Q2 + lambda * sum(abs(z_i-z_j)
    / abs(i-j)), where Q2 is the covariance-normalized second harmonic and
    z = -ln(b/a) exp(2j PA). Polar-angle harmonics are fit to valid samples,
    without clipping or filling masked angular gaps. There is no gradient
    significance gate. Shape proposals use an initial RMS displacement of
    0.15. Before stopping, try the inner neighbor's geometry and guided
    directional refinement. Other interpolation methods, custom loss modes,
    superellipses and Fourier shapes retain the clipped FFT objective.

    Harmonic fits propagate the pixel-noise covariance into ellipticity and
    PA at fixed radius, using 0.03 steps in joint log-q shape coordinates.
    These approximate data-only errors exclude regularization and systematic
    offsets. Unsupported estimates and PA consistent with circularity are NaN.
    Other fitting modes retain the scatter from refits at radii shifted by 5%.

    Parameters
    -----------------
    ap_iso_ellip_max : float, default 0.95
      Maximum ellipticity (1 - b/a), strictly between 0 and 1.
      Shared by native initialization, fitting, uncertainties, and interpolation.

    ap_scale : float, default 0.2
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

    ap_fit_limit : float, default 1
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 1, smaller values will end
      fitting further out in the galaxy image.

    ap_regularize_scale : float, default 4 for harmonic fitting, 1 otherwise
      Additive shape penalty for Q2; multiplicative penalty for FFT fitting.
      Larger values favour smoother shapes.

    ap_isofit_rms : ndarray or string, default None
      Total pixel RMS, including source noise, in image units. Supply an
      image-shaped array or a primary-HDU FITS / NPY filename. Takes precedence
      over ap_isofit_gain. Nonpositive/nonfinite RMS pixels are excluded.
      Applies to ordinary bilinear ellipse fitting only.

    ap_isofit_gain : float, default None
      Positive effective gain in electrons per image unit. Without an RMS map,
      add max(image - background, 0) / gain to the background variance, without
      smoothing. Omit both noise options to use background variance only.
      Applies to ordinary bilinear ellipse fitting with loss mode 2 only.

    ap_isofit_robustclip : float, default 0.15
      Upper flux fraction clipped in the legacy FFT objective. The harmonic
      objective does not clip; contaminants should be masked.

    ap_isofit_losscoefs : tuple, default (2,)
      Tuple of FFT coefficients to use in optimization
      procedure. AutoProf will attemp to minimize the power in all
      listed FFT coefficients. Must be a tuple, not a list.

    ap_isofit_superellipse : bool, default False
      If True, AutoProf will fit superellipses instead of regular
      ellipses. A superellipse is typically used to represent
      boxy/disky isophotes. The variable controlling the transition
      from a rectangle to an ellipse to a four-armed-star like shape
      is C. A value of C = 2 represents an ellipse and is the starting
      point of the optimization.

    ap_isofit_fitcoefs : tuple, default None
      Tuple of FFT coefficients to use in fitting procedure. AutoProf
      will attemp to fit ellipses with these Fourier mode
      perturbations. Such perturbations allow for lopsided, boxy,
      disky, and other types of isophotes beyond straightforward
      ellipses. Must be a tuple, not a list. Note that AutoProf will
      first fit ellipses, then turn on the Fourier mode perturbations,
      thus the fitting time will always be longer.

    ap_isofit_fitcoefs_FFTinit : bool, default False
      If True, the coefficients for the Fourier modes fitted from
      ap_isofit_fitcoefs will be initialized using an FFT
      decomposition along fitted elliptical isophotes. This can
      improve the fit result, though it is less stable and so users
      should examine the results after fitting.

    ap_isofit_interpolate_method : string, default 'bilinear'
      Select method for flux interpolation while fitting isophotes.
      Options are 'lanczos', 'bicubic', and 'bilinear'.

    ap_isofit_band_start : float, default 5 for harmonic fitting, None otherwise
      Enable mean-sector bands below this initial median intensity / background
      RMS. Shared harmonic-mean edges give half-width R*step/(2+step) on
      geometric grids, as in default extraction. Only
      ordinary harmonic fits support this option; None retains line sampling.

    ap_isofit_iterlimitmax : int, default 1000
      Maximum number of iterations (each iteration adjusts every
      isophote once) before automatically stopping optimization. For
      galaxies with lots of structure (ie detailed spiral arms) more
      iterations may be needed to fully fit the light distribution,
      but runtime will be longer.

    ap_isofit_iterlimitmin : int, default 0
      Minimum number of iterations before optimization is allowed to
      stop.

    ap_isofit_iterstopnochange : float, default 3
      Trigger a stopping check after this value times max(1, N - 1)
      consecutive unsuccessful updates, for N active radii. Accepted updates
      reset the counter. Ordinary ellipse fits verify stopping with guided
      searches down to a fixed step of 0.15/256. Improvements resume random
      fitting, using the largest accepted step for both searches; otherwise
      fitting stops. The initial random RMS/directional step is 0.15.
      C/Fourier fits use the counter-only rule. Fractional values are allowed;
      the maximum iteration limit still applies.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'mask' (optional)
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'fit C': , # optional, superellipse scale parameter (ndarray)
         'fit Fmodes': , # optional, fitted Fourier mode indices (tuple)
         'fit Fmode A*': , # optional, fitted Fourier mode amplitudes, * for each index (ndarray)
         'fit Fmode Phi*': , # optional, fitted Fourier mode phases, * for each index (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }

    """

    ellip_max = _resolve_ellip_max(options)
    if "ap_scale" in options:
        scale = options["ap_scale"]
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None
    interp_method = (
        _validate_interpolate_method(
            options["ap_isofit_interpolate_method"],
            "ap_isofit_interpolate_method",
        )
        if "ap_isofit_interpolate_method" in options
        else "bilinear"
    )
    rad_interp = _iso_interpolate_radius(options, results)

    # Determine sampling radii
    ######################################################################
    sample_radii = _determine_sample_radii(
        dat,
        IMG,
        results,
        options,
        scale,
        mask,
        max(1.0, results["psf fwhm"] / 2),
        1,
        np.median,
        interp_method=interp_method,
        rad_interp=rad_interp,
    )
    ellip, pa = _initial_ellipse_profile(sample_radii, results, ellip_max)
    logging.debug("%s: sample radii: %s" % (options["ap_name"], str(sample_radii)))
    # Fit isophotes
    ######################################################################
    perturb_scale = 0.03
    robust_clip = options["ap_isofit_robustclip"] if "ap_isofit_robustclip" in options else 0.15
    N_perturb = 5
    fit_coefs = options["ap_isofit_losscoefs"] if "ap_isofit_losscoefs" in options else None
    fit_params = options["ap_isofit_fitcoefs"] if "ap_isofit_fitcoefs" in options else None
    fit_superellipse = (
        options["ap_isofit_superellipse"] if "ap_isofit_superellipse" in options else False
    )
    use_harmonic = (
        fit_params is None and not fit_superellipse and interp_method == "bilinear"
        and (fit_coefs is None or tuple(fit_coefs) in ((), (2,)))
    )
    band_start = options.get("ap_isofit_band_start", 5.0 if use_harmonic else None)
    if band_start is not None:
        if not np.isfinite(band_start) or band_start <= 0:
            raise ValueError("ap_isofit_band_start must be finite and positive, or None.")
        if not use_harmonic:
            raise ValueError("ap_isofit_band_start requires ordinary bilinear harmonic fitting.")
    if not use_harmonic and (
        options.get("ap_isofit_rms") is not None or options.get("ap_isofit_gain") is not None
    ):
        raise ValueError(
            "ap_isofit_rms and ap_isofit_gain require ordinary bilinear ellipse fitting with loss mode 2."
        )
    variance = _isofit_variance(dat, results["background noise"], options) if use_harmonic else None
    regularize_scale = options.get("ap_regularize_scale", 4.0 if use_harmonic else 1.0)
    logging.info("%s: Using %s objective", options["ap_name"],
                 "additive harmonic Q2" if use_harmonic else "multiplicative clipped FFT")
    shape_step = 0.15
    # RMS random displacement equals the first directional trial length.
    shape_scatter = (
        shape_step / np.sqrt(2)
        if fit_params is None and not fit_superellipse else perturb_scale
    )
    parameters = list(
        {
            "ellip": ellip[i],
            "pa": pa[i],
            "m": fit_params,
            "C": 2 if fit_superellipse else None,
            "Am": None if fit_params is None else np.zeros(len(fit_params)),
            "Phim": None if fit_params is None else np.zeros(len(fit_params)),
        }
        for i in range(len(ellip))
    )
    if band_start is not None:
        _set_isofit_bands(dat, sample_radii, parameters, results["center"],
                         results["background noise"], variance, mask, band_start)

    count = 0

    iterlimitmax = (
        options["ap_isofit_iterlimitmax"] if "ap_isofit_iterlimitmax" in options else 1000
    )
    iterlimitmin = options["ap_isofit_iterlimitmin"] if "ap_isofit_iterlimitmin" in options else 0
    iterstopnochange = (
        options["ap_isofit_iterstopnochange"] if "ap_isofit_iterstopnochange" in options else 3
    )
    count_nochange = 0
    use_center = copy(results["center"])
    active_mask = np.array(
        list(
            _has_enough_isophote_samples(
                dat,
                sample_radii[i],
                parameters[i],
                use_center,
                mask,
                max_mode=_max_fit_mode(fit_coefs),
                interp_method=interp_method,
                rad_interp=rad_interp,
                variance=variance,
            )
            for i in range(len(sample_radii))
        )
    )
    if np.sum(active_mask) < 2:
        raise Exception(
            "Unable to initialize ellipse fit, too few radii have enough unmasked samples."
        )
    # Keep two ellipse slots to preserve the C/Fourier proposal schedule.
    param_cycle = 2
    base_params = 2 + int(fit_superellipse)
    while count < iterlimitmax:
        I = np.flatnonzero(active_mask)
        if len(I) == 0:
            break
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug("%s: count: %i" % (options["ap_name"], count))
        count += 1

        np.random.shuffle(I)
        N_perturb = int(1 + (10 / np.sqrt(count)))
        activated_this_pass = False

        def loss(params, i):
            return _FFT_Robust_loss(
                dat, sample_radii, params, i, use_center,
                results["background noise"], mask=mask,
                reg_scale=regularize_scale if count > 4 else 0,
                robust_clip=robust_clip, fit_coefs=fit_coefs,
                name=options["ap_name"], active_mask=active_mask,
                interp_method=interp_method, rad_interp=rad_interp,
                variance=variance,
            )

        for i in I:
            perturbations = []
            perturbations.append(deepcopy(parameters))
            perturbations[-1][i]["loss"] = loss(perturbations[-1], i)
            for n in range(N_perturb):
                perturbations.append(deepcopy(parameters))
                if count % param_cycle < 2:
                    perturbations[-1][i]["ellip"], perturbations[-1][i]["pa"] = _perturb_ellipse(
                        perturbations[-1][i]["ellip"],
                        perturbations[-1][i]["pa"],
                        np.random.normal(loc=0, scale=shape_scatter, size=2),
                        ellip_max,
                    )
                elif (count % param_cycle) == 2 and not parameters[i]["C"] is None:
                    perturbations[-1][i]["C"] = 10 ** (
                        np.log10(perturbations[-1][i]["C"])
                        + np.random.normal(loc=0, scale=np.log10(1.0 + perturb_scale))
                    )
                elif count % param_cycle < (base_params + len(parameters[i]["m"])):
                    perturbations[-1][i]["Am"][
                        (count % param_cycle) - base_params
                    ] += np.random.normal(loc=0, scale=perturb_scale)
                elif count % param_cycle < (base_params + 2 * len(parameters[i]["m"])):
                    phim_index = (count % param_cycle) - base_params - len(parameters[i]["m"])
                    perturbations[-1][i]["Phim"][phim_index] = (
                        perturbations[-1][i]["Phim"][phim_index]
                        + np.random.normal(
                            loc=0,
                            scale=2 * np.pi * perturb_scale / parameters[i]["m"][phim_index],
                        )
                    ) % (2 * np.pi / parameters[i]["m"][phim_index])
                else:
                    raise Exception(
                        "Unrecognized optimization parameter id: %i" % (count % param_cycle)
                    )
                perturbations[-1][i]["loss"] = loss(perturbations[-1], i)

            if not np.isfinite(perturbations[0][i]["loss"]):
                active_mask[i] = False
                continue
            losses = list(p[i]["loss"] for p in perturbations)
            best = _best_finite_index(losses)
            if best > 0:
                parameters = deepcopy(perturbations[best])
                del parameters[i]["loss"]
                count_nochange = 0
            else:
                count_nochange += 1
            stop_threshold = iterstopnochange * max(1, len(I) - 1)
            if not (
                count_nochange < stop_threshold
                or count < iterlimitmin
            ):
                activated = _activate_inactive_radii(
                    dat,
                    sample_radii,
                    parameters,
                    active_mask,
                    use_center,
                    mask,
                    max_mode=_max_fit_mode(fit_coefs),
                    interp_method=interp_method,
                    rad_interp=rad_interp,
                    ellip_max=ellip_max,
                    variance=variance,
                )
                if len(activated) > 0:
                    count_nochange = 0
                    activated_this_pass = True
                    break
                if param_cycle > 2 or (parameters[i]["m"] is None and not fit_superellipse):
                    break
                elif parameters[i]["m"] is None and fit_superellipse:
                    logging.info(
                        "%s: Started C fitting at iteration %i" % (options["ap_name"], count)
                    )
                    param_cycle = 3
                    iterstopnochange = max(iterstopnochange, param_cycle)
                    count_nochange = 0
                    count = 0
                    if fit_coefs is None:
                        fit_coefs = (2, 4)
                    active_mask = np.array(
                        list(
                            _has_enough_isophote_samples(
                                dat,
                                sample_radii[ii],
                                parameters[ii],
                                use_center,
                                mask,
                                max_mode=_max_fit_mode(fit_coefs),
                                interp_method=interp_method,
                                rad_interp=rad_interp,
                            )
                            for ii in range(len(sample_radii))
                        )
                    )
                else:
                    logging.info(
                        "%s: Started Fmode fitting at iteration %i" % (options["ap_name"], count)
                    )
                    if fit_superellipse:
                        logging.info(
                            "%s: Started C fitting at iteration %i" % (options["ap_name"], count)
                        )
                    param_cycle = base_params + 2 * len(parameters[i]["m"])
                    iterstopnochange = max(iterstopnochange, param_cycle)
                    count_nochange = 0
                    count = 0
                    if fit_coefs is None and not fit_params is None:
                        fit_coefs = fit_params
                        if not 2 in fit_coefs:
                            fit_coefs = tuple(sorted(set([2] + list(fit_coefs))))
                    if not parameters[i]["C"] is None and (
                        not "ap_isofit_losscoefs" in options
                        or options["ap_isofit_losscoefs"] is None
                    ):
                        fit_coefs = tuple(sorted(set([4] + list(fit_coefs))))
                    active_mask = np.array(
                        list(
                            _has_enough_isophote_samples(
                                dat,
                                sample_radii[ii],
                                parameters[ii],
                                use_center,
                                mask,
                                max_mode=_max_fit_mode(fit_coefs),
                                interp_method=interp_method,
                                rad_interp=rad_interp,
                            )
                            for ii in range(len(sample_radii))
                        )
                    )
                    if (
                        "ap_isofit_fitcoefs_FFTinit" in options
                        and options["ap_isofit_fitcoefs_FFTinit"]
                    ):
                        for ii in np.flatnonzero(active_mask):
                            if parameters[ii]["m"] is None or len(parameters[ii]["m"]) == 0:
                                continue
                            isovals = _extract_fft_isophote_samples(
                                dat,
                                sample_radii[ii],
                                parameters[ii],
                                use_center,
                                mask=mask,
                                max_mode=max(parameters[ii]["m"]),
                                interp_method=interp_method,
                                rad_interp=rad_interp,
                            )
                            if isovals is None:
                                continue
                            if len(isovals) <= 2 * max(parameters[ii]["m"]):
                                continue

                            quantile = 0.85 if mask is None else 0.9
                            coefs = fft(
                                np.clip(
                                    isovals,
                                    a_max=np.quantile(isovals, quantile),
                                    a_min=None,
                                )
                            )
                            if coefs[0] == 0 or not np.isfinite(coefs[0]):
                                continue
                            for m in range(len(parameters[ii]["m"])):
                                parameters[ii]["Am"][m] = np.abs(
                                    coefs[parameters[ii]["m"][m]] / coefs[0]
                                ) * np.sign(np.angle(coefs[parameters[ii]["m"][m]]))
                                parameters[ii]["Phim"][m] = np.angle(
                                    coefs[parameters[ii]["m"][m]]
                                ) % (2 * np.pi)

        if activated_this_pass:
            continue
        stop_threshold = iterstopnochange * max(1, len(np.flatnonzero(active_mask)) - 1)
        if not (count_nochange < stop_threshold or count < iterlimitmin):
            if fit_params is None and not fit_superellipse:
                checked_updates, accepted_steps = _check_ellipse_directions(
                    parameters, active_mask, loss, ellip_max, shape_step,
                )
                logging.debug("%s: Directional check changed %i radii", options["ap_name"], checked_updates)
                if checked_updates:
                    if accepted_steps:
                        shape_step = max(accepted_steps.values())
                        shape_scatter = shape_step / np.sqrt(2)
                    count_nochange = 0
                    continue
            break

    active_indices = np.flatnonzero(active_mask)
    if len(active_indices) < 4:
        raise Exception(
            "Unable to fit isophotes, too few radii have enough unmasked samples."
        )
    skipped_fit_radii = len(sample_radii) - len(active_indices)
    if skipped_fit_radii > 0:
        logging.info(
            "%s: Excluded %i fit radii with too few unmasked samples"
            % (options["ap_name"], skipped_fit_radii)
        )
    sample_radii = np.asarray(sample_radii)[active_indices]
    parameters = list(parameters[i] for i in active_indices)

    logging.info("%s: Completed isohpote fit in %i itterations" % (options["ap_name"], count))
    # Compute errors
    ######################################################################
    ellip_err, pa_err = _FFT_Robust_Errors(
        dat,
        sample_radii,
        parameters,
        use_center,
        results["background noise"],
        mask=mask,
        reg_scale=regularize_scale,
        robust_clip=robust_clip,
        fit_coefs=fit_coefs,
        name=options["ap_name"],
        interp_method=interp_method,
        rad_interp=rad_interp,
        ellip_max=ellip_max,
        variance=variance,
    )
    for i in range(len(parameters)):
        parameters[i]["ellip err"] = ellip_err[i]
        parameters[i]["pa err"] = pa_err[i]
    # Plot fitting results
    ######################################################################
    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Isophote_Fit(dat, sample_radii, parameters, results, options)

    res = {
        "fit ellip": np.array(list(parameters[i]["ellip"] for i in range(len(parameters)))),
        "fit pa": np.array(list(parameters[i]["pa"] for i in range(len(parameters)))),
        "fit R": sample_radii,
        "fit ellip_err": ellip_err,
        "fit pa_err": pa_err,
        "fit band start": band_start,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % sample_radii[-1],
    }
    if use_harmonic:
        res["fit error method"] = "harmonic covariance"
        res["auxfile fit uncertainty"] = (
            "geometry uncertainties: fixed-radius harmonic covariance (data only); "
            "unavailable ellipticity/PA errors: %i/%i"
            % (np.sum(~np.isfinite(ellip_err)), np.sum(~np.isfinite(pa_err)))
        )
    if band_start is not None:
        limits = np.array([p.get("_isoband", (1.0, 1.0)) for p in parameters]) * sample_radii[:, None]
        res["fit band lower R"], res["fit band upper R"] = limits.T
    if skipped_fit_radii > 0:
        res["auxfile fit masked radii"] = "fit radii excluded for insufficient samples: %i" % skipped_fit_radii
    if not fit_params is None:
        res.update({"fit Fmodes": fit_params})
        for m in range(len(fit_params)):
            res.update(
                {
                    "fit Fmode A%i"
                    % fit_params[m]: np.array(
                        list(parameters[i]["Am"][m] for i in range(len(parameters)))
                    ),
                    "fit Fmode Phi%i"
                    % fit_params[m]: np.array(
                        list(parameters[i]["Phim"][m] for i in range(len(parameters)))
                    ),
                }
            )
    if fit_superellipse:
        res.update({"fit C": np.array(list(parameters[i]["C"] for i in range(len(parameters))))})
    return IMG, res


def Isophote_Fit_Forced(IMG, results, options):
    """Read previously fit PA/ellipticity profile.

    Reads a .prof file and extracts the corresponding PA/ellipticity profile. The profile is extracted generically, so any csv file with columns for 'R', 'pa', 'ellip', and optionally 'pa_e' and 'ellip_e' will be able to create a forced fit. This can be used for testing purposes, such as selecting a specific isophote to extract or comparing AutoProf SB extraction methods with other softwares.

    Parameters
    -----------------
    ap_forcing_profile : string, default None
        File path to .prof file providing forced photometry PA and
        ellip values to apply to *ap_image_file* (required for forced
        photometry)

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'center'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)

        }

    """
    with open(options["ap_forcing_profile"], "r") as f:
        raw = f.readlines()
        for i, l in enumerate(raw):
            if l.strip() and not l.lstrip().startswith("#"):
                readfrom = i
                break
        header = list(h.strip() for h in raw[readfrom].split(","))
        # Only geometry columns are numeric inputs; sampling labels are metadata.
        force = {h: [] for h in header if h in ("R", "pa", "ellip", "pa_e", "ellip_e")}
        for l in raw[readfrom + 1 :]:
            if not l.strip() or l.lstrip().startswith("#"):
                continue
            row = dict(zip(header, l.split(",")))
            try:
                float(row["R"])
            except ValueError:
                continue  # Skip an uncommented units row.
            for h in force:
                force[h].append(float(row[h]))

    force["pa"] = PA_shift_convention(np.array(force["pa"]), deg=True)

    if "ap_doplot" in options and options["ap_doplot"]:
        parameters = []
        for i in range(len(force["R"])):
            parameters.append({"ellip": force["ellip"][i], "pa": force["pa"][i], "C": None})
        Plot_Isophote_Fit(
            IMG - results["background"],
            np.array(force["R"]),
            parameters,
            results,
            options,
        )

    res = {
        "fit ellip": np.array(force["ellip"]),
        "fit pa": np.array(force["pa"]) * np.pi / 180,
        "fit R": list(np.array(force["R"]) / options["ap_pixscale"]),
    }
    if "ellip_e" in force and "pa_e" in force:
        res["fit ellip_err"] = np.array(force["ellip_e"])
        res["fit pa_err"] = np.array(force["pa_e"]) * np.pi / 180
    return IMG, res


######################################################################
def _FFT_mean_loss(
    dat,
    R,
    E,
    PA,
    i,
    C,
    noise,
    mask=None,
    reg_scale=1.0,
    name="",
    active_mask=None,
    interp_method="bilinear",
    rad_interp=None,
):

    params = {"ellip": E[i], "pa": PA[i]}
    isovals = _extract_fft_isophote_samples(
        dat,
        R[i],
        params,
        C,
        mask,
        max_mode=2,
        interp_method=interp_method,
        rad_interp=rad_interp,
    )
    if isovals is None:
        return np.inf

    f2_loss = None
    if len(isovals) > 4:
        coefs = fft(isovals)
        norm = len(isovals) * (max(0, np.mean(isovals)) + noise)
        if norm > 0 and np.isfinite(norm):
            f2_loss = np.abs(coefs[2]) / norm

    reg_loss = 0
    shape = _ellipse_shape(E[i], PA[i])
    for j in _active_neighbors(i, active_mask, len(R)):
        neighbor_reg_loss = abs(shape - _ellipse_shape(E[j], PA[j]))
        # The geometric radius grid makes index separation proportional to log-radius distance.
        reg_loss += neighbor_reg_loss / abs(i - j)

    if f2_loss is None or not np.isfinite(f2_loss):
        return np.inf
    return f2_loss * (1 + reg_loss * reg_scale)


def Isophote_Fit_FFT_mean(IMG, results, options):
    """Fit elliptical isophotes to a galaxy image using FFT coefficients and regularization.

    Same as the standard isophote fitting routine, except uses less
    robust mean/std measures. This is only intended for low S/N data
    where pixels have low integer counts.

    Parameters
    -----------------
    ap_iso_ellip_max : float, default 0.95
      Maximum ellipticity (1 - b/a), strictly between 0 and 1.
      Shared by native initialization, fitting, uncertainties, and interpolation.

    ap_scale : float, default 0.2
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

    ap_fit_limit : float, default 1
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 1, smaller values will end
      fitting further out in the galaxy image.

    ap_regularize_scale : float, default 1
      Scale of the distance between neighbouring log-q shape vectors.
      Larger values favour smoother shapes. PA differences lose weight
      near circularity; this scale differs from the former separate penalties.

    ap_isofit_interpolate_method : string, default 'bilinear'
      Select method for flux interpolation while fitting isophotes.
      Options are 'lanczos', 'bicubic', and 'bilinear'.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'center'
    - 'psf fwhm'
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }

    """

    ellip_max = _resolve_ellip_max(options)
    if "ap_scale" in options:
        scale = options["ap_scale"]
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None
    interp_method = (
        _validate_interpolate_method(
            options["ap_isofit_interpolate_method"],
            "ap_isofit_interpolate_method",
        )
        if "ap_isofit_interpolate_method" in options
        else "bilinear"
    )
    rad_interp = _iso_interpolate_radius(options, results)

    # Determine sampling radii
    ######################################################################
    sample_radii = _determine_sample_radii(
        dat,
        IMG,
        results,
        options,
        scale,
        mask,
        3 * results["psf fwhm"] / 2,
        1,
        np.mean,
        interp_method=interp_method,
        rad_interp=rad_interp,
    )
    ellip, pa = _initial_ellipse_profile(sample_radii, results, ellip_max)
    logging.debug("%s: sample radii: %s" % (options["ap_name"], str(sample_radii)))

    # Fit isophotes
    ######################################################################
    perturb_scale = 0.03
    regularize_scale = options["ap_regularize_scale"] if "ap_regularize_scale" in options else 1.0
    N_perturb = 5

    count = 0

    count_nochange = 0
    use_center = copy(results["center"])
    active_mask = np.array(
        list(
            _has_enough_isophote_samples(
                dat,
                sample_radii[i],
                {"ellip": ellip[i], "pa": pa[i]},
                use_center,
                mask,
                interp_method=interp_method,
                rad_interp=rad_interp,
            )
            for i in range(len(sample_radii))
        )
    )
    if np.sum(active_mask) < 2:
        raise Exception(
            "Unable to initialize ellipse fit, too few radii have enough unmasked samples."
        )
    while count < 300 and count_nochange < (3 * len(sample_radii)):
        I = np.flatnonzero(active_mask)
        if len(I) == 0:
            break
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug("%s: count: %i" % (options["ap_name"], count))
        count += 1

        np.random.shuffle(I)
        activated_this_pass = False
        stop_this_pass = False
        for i in I:
            perturbations = []
            perturbations.append({"ellip": copy(ellip), "pa": copy(pa)})
            perturbations[-1]["loss"] = _FFT_mean_loss(
                dat,
                sample_radii,
                perturbations[-1]["ellip"],
                perturbations[-1]["pa"],
                i,
                use_center,
                results["background noise"],
                mask=mask,
                reg_scale=regularize_scale if count > 4 else 0,
                name=options["ap_name"],
                active_mask=active_mask,
                interp_method=interp_method,
                rad_interp=rad_interp,
            )
            for n in range(N_perturb):
                perturbations.append({"ellip": copy(ellip), "pa": copy(pa)})
                perturbations[-1]["ellip"][i], perturbations[-1]["pa"][i] = _perturb_ellipse(
                    perturbations[-1]["ellip"][i],
                    perturbations[-1]["pa"][i],
                    np.random.normal(loc=0, scale=perturb_scale, size=2),
                    ellip_max,
                )
                perturbations[-1]["loss"] = _FFT_mean_loss(
                    dat,
                    sample_radii,
                    perturbations[-1]["ellip"],
                    perturbations[-1]["pa"],
                    i,
                    use_center,
                    results["background noise"],
                    mask=mask,
                    reg_scale=regularize_scale if count > 4 else 0,
                    name=options["ap_name"],
                    active_mask=active_mask,
                    interp_method=interp_method,
                    rad_interp=rad_interp,
                )

            if not np.isfinite(perturbations[0]["loss"]):
                active_mask[i] = False
                continue
            losses = list(p["loss"] for p in perturbations)
            best = _best_finite_index(losses)
            if best > 0:
                ellip = copy(perturbations[best]["ellip"])
                pa = copy(perturbations[best]["pa"])
                count_nochange = 0
            else:
                count_nochange += 1
            stop_threshold = 3 * max(1, len(I) - 1)
            if count_nochange >= stop_threshold:
                activated = _activate_inactive_ellipse_radii(
                    dat,
                    sample_radii,
                    ellip,
                    pa,
                    active_mask,
                    use_center,
                    mask,
                    interp_method=interp_method,
                    rad_interp=rad_interp,
                    ellip_max=ellip_max,
                )
                if len(activated) > 0:
                    count_nochange = 0
                    activated_this_pass = True
                    break
                stop_this_pass = True
                break
        if activated_this_pass:
            continue
        if stop_this_pass:
            break

    logging.info("%s: Completed isohpote fit in %i itterations" % (options["ap_name"], count))
    active_indices = np.flatnonzero(active_mask)
    if len(active_indices) < 6:
        raise Exception(
            "Unable to fit isophotes, too few radii have enough unmasked samples."
        )
    skipped_fit_radii = len(sample_radii) - len(active_indices)
    if skipped_fit_radii > 0:
        logging.info(
            "%s: Excluded %i fit radii with too few unmasked samples"
            % (options["ap_name"], skipped_fit_radii)
        )
    sample_radii = np.asarray(sample_radii)[active_indices]
    ellip = ellip[active_indices]
    pa = pa[active_indices]
    # detect collapsed center
    ######################################################################
    for i in range(5):
        if (_inv_x_to_eps(ellip[i]) - _inv_x_to_eps(ellip[i + 1])) > 0.5:
            ellip[: i + 1] = ellip[i + 1]
            pa[: i + 1] = pa[i + 1]

    # Smooth ellip and pa profile
    ######################################################################
    smooth_ellip = copy(ellip)
    smooth_pa = copy(pa)
    ellip[:3] = min(ellip[:3])
    smooth_ellip = _ellip_smooth(sample_radii, smooth_ellip, 5, ellip_max)
    smooth_pa = _pa_smooth(sample_radii, smooth_pa, 5)

    if "ap_doplot" in options and options["ap_doplot"]:
        ranges = [
            [
                max(0, int(use_center["x"] - sample_radii[-1] * 1.2)),
                min(dat.shape[1], int(use_center["x"] + sample_radii[-1] * 1.2)),
            ],
            [
                max(0, int(use_center["y"] - sample_radii[-1] * 1.2)),
                min(dat.shape[0], int(use_center["y"] + sample_radii[-1] * 1.2)),
            ],
        ]
        LSBImage(
            dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]],
            results["background noise"],
        )
        # plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
        #                    a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(sample_radii)):
            plt.gca().add_patch(
                Ellipse(
                    xy=(use_center["x"] - ranges[0][0], use_center["y"] - ranges[1][0]),
                    width=2 * sample_radii[i],
                    height=2 * sample_radii[i] * (1.0 - ellip[i]),
                    angle=pa[i] * 180 / np.pi,
                    fill=False,
                    linewidth=((i + 1) / len(sample_radii)) ** 2,
                    color="r",
                )
            )
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}fit_ellipse_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

        plt.scatter(sample_radii, ellip, color="r", label="ellip")
        plt.scatter(sample_radii, pa / np.pi, color="b", label="pa/$np.pi$")
        show_ellip = _ellip_smooth(sample_radii, ellip, deg=5, ellip_max=ellip_max)
        show_pa = _pa_smooth(sample_radii, pa, deg=5)
        plt.plot(
            sample_radii,
            show_ellip,
            color="orange",
            linewidth=2,
            linestyle="--",
            label="smooth ellip",
        )
        plt.plot(
            sample_radii,
            show_pa / np.pi,
            color="purple",
            linewidth=2,
            linestyle="--",
            label="smooth pa/$np.pi$",
        )
        # plt.xscale('log')
        plt.legend()
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}phaseprofile_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    # Compute errors
    ######################################################################
    ellip_err = np.zeros(len(ellip))
    ellip_err[:2] = np.sqrt(np.sum((ellip[:4] - smooth_ellip[:4]) ** 2) / 4)
    ellip_err[-1] = np.sqrt(np.sum((ellip[-4:] - smooth_ellip[-4:]) ** 2) / 4)
    pa_err = np.zeros(len(pa))
    pa_err[:2] = np.sqrt(np.sum((pa[:4] - smooth_pa[:4]) ** 2) / 4)
    pa_err[-1] = np.sqrt(np.sum((pa[-4:] - smooth_pa[-4:]) ** 2) / 4)
    for i in range(2, len(pa) - 1):
        ellip_err[i] = np.sqrt(
            np.sum((ellip[i - 2 : i + 2] - smooth_ellip[i - 2 : i + 2]) ** 2) / 4
        )
        pa_err[i] = np.sqrt(np.sum((pa[i - 2 : i + 2] - smooth_pa[i - 2 : i + 2]) ** 2) / 4)

    res = {
        "fit ellip": ellip,
        "fit pa": pa,
        "fit R": sample_radii,
        "fit ellip_err": ellip_err,
        "fit pa_err": pa_err,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % sample_radii[-1],
    }
    if skipped_fit_radii > 0:
        res["auxfile fit masked radii"] = "fit radii excluded for insufficient samples: %i" % skipped_fit_radii
    return IMG, res
