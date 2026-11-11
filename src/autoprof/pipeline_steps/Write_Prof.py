from astropy.io import fits
from astropy.table import Table
import numpy as np
import sys
import os
import logging

from ..autoprofutils.SharedFunctions import PA_shift_convention, _image_celestial_wcs
from datetime import datetime
from time import sleep

__all__ = ("WriteProf", )

def WriteProf(IMG, results, options):
    """Basic method to write SB profile to disk.

    This step writes the results of the AutoProf pipeline analysis to
    a file. There are two files written, a .prof file containing the
    surface brightness profile and acompanying measurements, and a
    .aux file containing global results, messages, and setting used
    for the pipeline. The .prof file looks for specific keywords in
    the results dictionary: prof header, prof units, prof data, and
    prof format. There are the results from the isophotal fitting
    step. prof header gives the column names for the profile, prof
    units is a dictionary which gives the corresponding units for each
    column header key, prof data is a dictionary containing a list of
    values for each header key, and prof format is a dictionary which
    gives the python string format for values under each header key
    (for example '%.4f' gives a number to 4 decimal places). The
    profile is written with comma (or a user specified delimiter)
    separation for each value, where each row corresponds to a given
    isophote at increasing semi-major axis values.

    Center comments record zero-based center_x/center_y in the original input
    image and, with a celestial WCS, center_ra/center_dec in ICRS degrees.
    Measurement comments record background, background_rms, pixscale, zeropoint,
    psf_fwhm, init_radius and fit_limit when available, with their units.

    The .aux file has a less strict format than the .prof file. The
    first line records the date and time that the file was written,
    the second line gives the name of the object as specified by the
    user or the filename. The next lines are taken from the results
    dictionary, any result key with auxfile in the name is taken as a
    message for the .aux file and written (in alphabetical order by
    key) to the file. See the pipeline step output formats for the
    messages that are included in the .aux file. Finally, a record of
    the user specified options is included for reference.

    Parameters
    ---------
    ap_saveto : string, default None
      Directory in which to save profile

    ap_name : string, default None
      Name of the current galaxy, used for making filenames.

    ap_delimiter : string, default ','
      Delimiter to use between entries in the profile.

    ap_profile_format : string, default 'csv'
      Type of file format to use for profile. Can choose from ['csv', 'fits']

    ap_savemask : bool, default False
      Save object mask fits file. This can create large files, depending on the size of the original image.

    Notes
    ----------
    :References:
    - 'prof header'
    - 'prof units'
    - 'prof data'
    - 'mask' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {}

    """

    saveto = options["ap_saveto"] if "ap_saveto" in options else "./"

    crop = options.get("_input_crop")
    sky_center = None
    pixel_center = None
    if "center" in results:
        x, y = results["center"]["x"], results["center"]["y"]
        if crop is not None:
            y0, _, x0, _ = crop["bounds"]
            x, y = x + x0, y + y0
        if np.all(np.isfinite([x, y])):
            pixel_center = (x, y)
        try:
            wcs = _image_celestial_wcs(options)
            if wcs is not None:
                expected_shape = crop["shape"] if crop is not None else IMG.shape
                if wcs.array_shape is not None and tuple(wcs.array_shape) != tuple(expected_shape):
                    raise ValueError("Image shape no longer matches the original WCS")
                if crop is not None:
                    y0, y1, x0, x1 = crop["bounds"]
                    if IMG.shape != (y1 - y0, x1 - x0):
                        raise ValueError("Image shape no longer matches the automatic crop")
                sky = wcs.pixel_to_world(x, y).icrs
                if not np.all(np.isfinite([sky.ra.deg, sky.dec.deg])):
                    raise ValueError("Nonfinite sky center")
                sky_center = (sky.ra.deg, sky.dec.deg)
        except Exception as exc:
            logging.warning("Omitting profile sky center: %s", exc)
    aux_results = results.copy()
    if crop is not None:
        y0, y1, x0, x1 = crop["bounds"]
        aux_results["auxfile crop"] = "input crop: x [%i, %i), y [%i, %i) pix" % (x0, x1, y0, y1)
        if "center" in results:
            aux_results["auxfile center"] = "center x: %.2f pix, y: %.2f pix" % (
                results["center"]["x"] + x0, results["center"]["y"] + y0,
            )

    # Write aux file
    with open(os.path.join(saveto, options["ap_name"] + ".aux"), "w") as f:
        # write profile info
        f.write("written on: %s\n" % str(datetime.now()))
        f.write("name: %s\n" % str(options["ap_name"]))

        for r in sorted(aux_results.keys()):
            if "auxfile" in r:
                f.write(aux_results[r] + "\n")
        for k in sorted(options.keys()):
            if k in ("ap_name", "_input_crop"):
                continue
            value = options[k]
            if crop is not None and k in ("ap_guess_center", "ap_set_center"):
                value = {"x": value["x"] + x0, "y": value["y"] + y0}
            f.write("option %s: %s\n" % (k, str(value)))

    # Write the profile
    delim = options["ap_delimiter"] if "ap_delimiter" in options else ","
    delim = options["ap_delimiter"] if "ap_delimiter" in options else ","
    try:
        results["prof data"]["pa"] = list(
            PA_shift_convention(np.array(results["prof data"]["pa"]), deg=True)
        )
    except:
        pass
    T = Table(data=results["prof data"], names=results["prof header"])
    if (
        "ap_profile_format" in options
        and options["ap_profile_format"].lower() == "fits"
    ):
        T.meta["UNITS"] = delim.join(
            results["prof units"][h] for h in results["prof header"]
        )
        T.write(
            os.path.join(saveto, options["ap_name"] + "_prof.fits"),
            format="fits",
            overwrite=True,
        )
    else:
        pixscale = options.get("ap_pixscale")
        background = results.get("background")
        background = (
            background / pixscale**2
            if background is not None and np.ndim(background) == 0
            and pixscale is not None and pixscale > 0
            else None
        )
        # Forced geometry is not a new initialization or fit of this image.
        forced = options.get("ap_forcing_profile") is not None
        fit_radii = results.get("fit R", [])
        measurements = [
            ("background", background, "flux/arcsec^2"),
            ("background_rms", results.get("background noise"), "flux/pix"),
            ("pixscale", pixscale, "arcsec/pix"),
            ("zeropoint", options.get("ap_zeropoint", 22.5), "mag"),
            ("psf_fwhm", results.get("psf fwhm"), "pix"),
            ("init_radius", None if forced else results.get("init R"), "pix"),
            ("fit_limit", fit_radii[-1] if not forced and len(fit_radii) else None, "pix"),
        ]
        measurements = [
            (key, value, unit) for key, value, unit in measurements
            if value is not None and np.ndim(value) == 0 and np.isfinite(value)
        ]
        metadata_comments = "".join(
            "# %s=%.12g, unit=%s\n" % (key, value, unit)
            for key, value, unit in measurements
        )
        T.write(
            os.path.join(saveto, options["ap_name"] + ".prof"),
            format="ascii.commented_header",
            delimiter=delim,
            overwrite=True,
            comment=(
                "# center_ra=%.10f, center_dec=%.10f\n" % sky_center
                if sky_center is not None else ""
            ) + (
                "# center_x=%.10f, center_y=%.10f\n" % pixel_center
                if pixel_center is not None else ""
            ) + metadata_comments + "# "
            + delim.join(results["prof units"][h] for h in results["prof header"])
            + "\n",
        )
    try:
        results["prof data"]["pa"] = list(
            PA_shift_convention(np.array(results["prof data"]["pa"]), deg=True)
        )
    except:
        pass

    # Write the mask data, if provided
    if (
        "mask" in results
        and (not results["mask"] is None)
        and "ap_savemask" in options
        and options["ap_savemask"]
    ):
        saved_mask = results["mask"].astype(int)
        if crop is not None:
            # Keep saved masks aligned with the input image; unprocessed pixels are masked.
            saved_mask = np.ones(crop["shape"], dtype=int)
            saved_mask[y0:y1, x0:x1] = results["mask"]
        hdul = fits.HDUList([fits.PrimaryHDU(saved_mask)])
        hdul.writeto(saveto + options["ap_name"] + "_mask.fits", overwrite=True)
        sleep(1)
        # Zip the mask file because it can be large and take a lot of memory, but in principle
        # is very easy to compress
        os.system("gzip -fq " + saveto + options["ap_name"] + "_mask.fits")
    return IMG, {}
