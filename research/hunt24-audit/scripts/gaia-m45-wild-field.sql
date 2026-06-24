
SELECT source_id, ra, dec, pmra, pmdec, parallax, parallax_error, phot_g_mean_mag, bp_rp, ruwe, radial_velocity
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', 56.75, 24.12, 17.78)
)
AND phot_g_mean_mag < 21.0
                