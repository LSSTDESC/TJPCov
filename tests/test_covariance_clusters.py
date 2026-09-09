#!/usr/bin/python3
import numpy as np
import pyccl as ccl
import sacc
from tjpcov.covariance_cluster_counts import CovarianceClusterCounts
from tjpcov.covariance_calculator import CovarianceCalculator
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC
from tjpcov.covariance_cluster_mass import ClusterMass
import pytest
import os
import shutil
import jinja2
import yaml
import copy

from conftest import INPUT_YML, OUTDIR

MASSDEF = ccl.halos.MassDef200m
SURVEY_NAME = "NC_mock_redshift_richness"


def teardown_module():
    shutil.rmtree(OUTDIR)


def _load_config(sacc_file=None, sigma_0=None):
    """Load INPUT_YML, optionally overriding the sacc file path/sigma_0."""
    with open(INPUT_YML, "r") as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    if sacc_file is not None:
        config["tjpcov"]["sacc_file"] = sacc_file
    if sigma_0 is not None:
        config["photo-z"]["sigma_0"] = sigma_0
    return config


@pytest.fixture
def mock_covariance_gauss(mock_cosmo, save_cluster_sacc_data):
    cc_cov = ClusterCountsGaussian(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    return cc_cov


@pytest.fixture
def mock_covariance_ssc(mock_cosmo, save_cluster_sacc_data):
    cc_cov = ClusterCountsSSC(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    return cc_cov


@pytest.fixture
def mock_covariance_mass(mock_cosmo, save_cluster_sacc_data):
    cc_cov = ClusterMass(INPUT_YML)
    cc_cov.load_from_cosmology(mock_cosmo)
    return cc_cov


# Tests start
def test_is_not_null(save_cluster_sacc_data):
    cc_cov = ClusterCountsSSC(INPUT_YML)
    assert cc_cov is not None

    cc_cov = ClusterCountsGaussian(INPUT_YML)
    assert cc_cov is not None

    cc_cov = ClusterMass(INPUT_YML)
    assert cc_cov is not None


def test_extract_indices(mock_covariance_gauss: CovarianceClusterCounts):
    cc = mock_covariance_gauss
    bincomb1 = ("mock_survey", "bin_rich_0", "bin_z_1")
    bincomb2 = ("mock_survey", "bin_rich_1", "bin_z_0")
    richness_i, z_i = cc.extract_indices_rich_z(bincomb1)
    richness_j, z_j = cc.extract_indices_rich_z(bincomb2)
    assert richness_i == 0
    assert richness_j == 1
    assert z_i == 1
    assert z_j == 0
    bincomb3 = ("clusters_0_1",)
    bincomb4 = ("clusters_1_0",)
    richness_i, z_i = cc.extract_indices_rich_z(bincomb3)
    richness_j, z_j = cc.extract_indices_rich_z(bincomb4)
    assert richness_i == 0
    assert richness_j == 1
    assert z_i == 1
    assert z_j == 0


def test_load_from_sacc(mock_covariance_gauss: CovarianceClusterCounts):
    assert mock_covariance_gauss.min_halo_ln_mass == np.log(1e13)
    assert mock_covariance_gauss.num_richness_bins == 3
    assert mock_covariance_gauss.num_z_bins == 6
    assert mock_covariance_gauss.min_richness == pytest.approx(
        10**0.8685892469619315
    )
    assert mock_covariance_gauss.max_richness == pytest.approx(
        10**1.8864886182119167
    )
    assert mock_covariance_gauss.z_min == pytest.approx(0.20000240679472392)
    assert mock_covariance_gauss.z_max == pytest.approx(0.6499986281483467)


def test_load_from_cosmology(mock_covariance_gauss: CovarianceClusterCounts):
    cosmo = ccl.CosmologyVanillaLCDM()
    mock_covariance_gauss.load_from_cosmology(cosmo)

    assert mock_covariance_gauss.cosmo == cosmo


def test_load_cluster_parameters(
    mock_covariance_gauss: CovarianceClusterCounts,
    mock_covariance_mass: ClusterMass,
):
    # Test with valid parameters
    mock_covariance_gauss.load_cluster_parameters()

    # Check that the parameters are loaded correctly
    assert mock_covariance_gauss.mass_def == "200m"
    assert mock_covariance_gauss.min_halo_ln_mass == np.log(1.0e13)
    assert mock_covariance_gauss.max_halo_ln_mass == np.log(1.0e16)
    assert mock_covariance_gauss.mass_func is not None
    assert mock_covariance_gauss.hbias is not None
    assert mock_covariance_gauss.sigma_0 == 0.005
    assert mock_covariance_gauss.mor_m_pivot == 14.648719176207223
    assert mock_covariance_gauss.mor_mu_p0 == 3.207
    assert mock_covariance_gauss.mor_mu_p1 == 0.75
    assert mock_covariance_gauss.mor_mu_p2 == 0.0
    assert mock_covariance_gauss.mor_sigma_p0 == 2.68
    assert mock_covariance_gauss.mor_sigma_p1 == 0.54
    assert mock_covariance_gauss.mor_sigma_p2 == 0.0
    assert mock_covariance_gauss.mor_z_pivot == 0.5

    # Test with invalid mass function
    invalid_mass_func = "InvalidMassFunc"
    config_copy = copy.deepcopy(mock_covariance_gauss.config)
    config_copy["mor_parameters"]["mass_func"] = invalid_mass_func
    with pytest.raises(
        ValueError,
        match=f"Invalid mass function: {invalid_mass_func}",
    ):
        ClusterCountsGaussian(config_copy)

    # Test with invalid halo bias
    invalid_halo_bias = "InvalidHaloBias"
    config_copy = copy.deepcopy(mock_covariance_gauss.config)
    config_copy["mor_parameters"]["halo_bias"] = invalid_halo_bias

    with pytest.raises(
        ValueError,
        match=f"Invalid halo bias: {invalid_halo_bias}",
    ):
        ClusterCountsGaussian(config_copy)
    # Test with cluster mass
    mock_covariance_mass.load_cluster_parameters()

    assert mock_covariance_mass.mass_def == "200m"
    assert mock_covariance_mass.min_halo_ln_mass == np.log(1.0e13)
    assert mock_covariance_mass.max_halo_ln_mass == np.log(1.0e16)
    assert mock_covariance_mass.mass_func is not None

    invalid_mass_func = "InvalidMassFunc"
    config_copy = mock_covariance_mass.config.copy()
    config_copy["mor_parameters"]["mass_func"] = invalid_mass_func
    mock_covariance_mass.config = config_copy
    with pytest.raises(
        ValueError,
        match=f"Invalid mass function: {invalid_mass_func}",
    ):
        mock_covariance_mass.load_cluster_parameters()


@pytest.mark.parametrize(
    "z, ref_val",
    [
        (0.3, 1.909297e-05),  # Values obtained from crow
        (0.35, 1.839692e-05),
    ],
)
def test_integral_mass_no_bias(
    mock_covariance_gauss: CovarianceClusterCounts, z, ref_val
):
    test = mock_covariance_gauss.mass_richness_integral(z, 0, remove_bias=True)
    assert test == pytest.approx(ref_val, rel=1e-3)


def test_shot_noise(mock_covariance_gauss: ClusterCountsGaussian):
    ref = 9662.189914386521
    test = mock_covariance_gauss.shot_noise(0, 0)
    assert test == pytest.approx(ref, rel=1e-3)


@pytest.mark.parametrize(
    "z, reference_val",
    [
        (0.5, 3.222714e-05),  # 3.41497484734468e-05),
        (0.55, 3.187768e-05),  # 3.418971366446374e-05),
    ],
)
def test_integral_mass(
    mock_covariance_gauss: CovarianceClusterCounts, z, reference_val
):
    test = mock_covariance_gauss.mass_richness_integral(z, 0)
    assert test == pytest.approx(reference_val, rel=1e-3)


@pytest.mark.parametrize(
    "z, reference_val",
    [
        (0.5, 3.656507e-05),  # 3.8543405453894756e-05),
    ],
)
def test_integral_mass_no_mproxy(
    mock_covariance_gauss: CovarianceClusterCounts, z, reference_val
):
    mock_covariance_gauss.richness_bins = np.linspace(13.5, 14, 4)
    mock_covariance_gauss.has_mproxy = False
    test = mock_covariance_gauss.mass_richness_integral(z, 0)
    assert test == pytest.approx(reference_val, rel=1e-1)


def test_mass_richness(mock_covariance_gauss: CovarianceClusterCounts):
    reference_min = 0.004609851096662065

    test_min = [
        mock_covariance_gauss.mass_richness(
            mock_covariance_gauss.min_halo_ln_mass, 1.0, i
        )
        for i in range(mock_covariance_gauss.num_richness_bins)
    ]
    assert np.sum(test_min) == pytest.approx(reference_min, rel=1e-1)


@pytest.mark.parametrize(
    "z_i, reference_val",
    [
        (0, 379483148.0254884),
        (1, 1793852013.816144),
        (2, 3955019406.7812386),
        (3, 2377993261.4517136),
        (4, 636933571.4428573),
        (5, 68339853.74341723),
    ],
)
def test_calc_dv(
    mock_covariance_gauss: CovarianceClusterCounts, z_i, reference_val
):
    z_true = 0.4
    sigma_0 = 0.05  # use large scatter here to have more non-zero values
    test = (
        mock_covariance_gauss.comoving_volume_element(z_true, z_i, sigma_0)
        / 1e4
    )
    assert test == pytest.approx(reference_val / 1e4)


def test_cov_gaussian_zero_offdiagonal(
    mock_covariance_gauss: ClusterCountsGaussian,
):
    cov_0111_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        (SURVEY_NAME, "bin_rich_1", "bin_z_0"),
        (SURVEY_NAME, "bin_rich_1", "bin_z_1"),
    )
    cov_1011_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        (SURVEY_NAME, "bin_rich_0", "bin_z_1"),
        (SURVEY_NAME, "bin_rich_1", "bin_z_1"),
    )
    cov_1001_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        (SURVEY_NAME, "bin_rich_1", "bin_z_0"),
        (SURVEY_NAME, "bin_rich_0", "bin_z_1"),
    )
    assert cov_0111_gauss == 0
    assert cov_1011_gauss == 0
    assert cov_1001_gauss == 0


def test_cov_nxn(
    mock_covariance_gauss: ClusterCountsGaussian,
    mock_covariance_ssc: ClusterCountsSSC,
):
    ref_sum = COV_REF_GAUSS_005[0, 0] + COV_REF_SSC_005[0, 0]
    # Need to include survey name from mock file here to ensure correct data
    # types are found
    cov_00_gauss = mock_covariance_gauss.get_covariance_block_for_sacc(
        (SURVEY_NAME, "bin_rich_0", "bin_z_0"),
        (SURVEY_NAME, "bin_rich_0", "bin_z_0"),
    )
    cov_00_ssc = mock_covariance_ssc.get_covariance_block_for_sacc(
        (SURVEY_NAME, "bin_rich_0", "bin_z_0"),
        (SURVEY_NAME, "bin_rich_0", "bin_z_0"),
    )
    assert cov_00_gauss + cov_00_ssc == pytest.approx(ref_sum, rel=1e-3)

    cov_00_gauss = mock_covariance_gauss.get_covariance_block(
        ("clusters_0_0",),
        ("clusters_0_0",),
    )
    cov_00_ssc = mock_covariance_ssc.get_covariance_block(
        ("clusters_0_0",),
        ("clusters_0_0",),
    )
    assert cov_00_gauss + cov_00_ssc == pytest.approx(ref_sum, rel=1e-3)


def test_cluster_count_tracer_missing_throws(save_cluster_sacc_data):
    s = copy.deepcopy(save_cluster_sacc_data)
    s.reorder(s.indices(data_type=sacc.standard_types.cluster_mean_log_mass))
    os.makedirs(OUTDIR, exist_ok=True)
    bad_sacc_file = os.path.join(OUTDIR, "test_cl_fails_sacc.fits")
    s.save_fits(bad_sacc_file, overwrite=True)

    config = _load_config(sacc_file=bad_sacc_file)

    with pytest.raises(
        ValueError, match="Cluster count covariance was requested"
    ):
        ClusterCountsGaussian(config)

    os.remove(bad_sacc_file)


def test_cluster_count_defaults_survey_area(save_cluster_sacc_data):
    """Test default survey area."""
    s = copy.deepcopy(save_cluster_sacc_data)
    del s.tracers["NC_mock_redshift_richness"]

    os.makedirs(OUTDIR, exist_ok=True)
    new_sacc_file = os.path.join(OUTDIR, "test_cl_no_survey_sacc.fits")
    s.save_fits(new_sacc_file, overwrite=True)

    config = _load_config(sacc_file=new_sacc_file)

    cc = ClusterCountsGaussian(config)
    assert cc.survey_area == 4 * np.pi

    cc = ClusterMass(config)
    assert cc.survey_area == 4 * np.pi

    os.remove(new_sacc_file)


def test_non_cluster_counts_covmat_zero(save_cluster_sacc_data):
    config = _load_config()

    cc = ClusterCountsGaussian(config)

    trs_cov = cc.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in trs_cov:
        blocks.append(np.array(1))

    cov = cc._build_matrix_from_blocks(blocks, trs_cov)

    # Only upper left should have been populated with values.
    assert np.all(cov[:18, :18] == 1)
    assert np.all(cov[19:, 19:] == 0)
    assert np.all(cov[19:, :18] == 0)
    assert np.all(cov[:18, 19:] == 0)

    cc = ClusterMass(config)

    trs_cov = cc.get_list_of_tracers_for_cov()
    blocks = []
    for trs1, trs2 in trs_cov:
        blocks.append(np.array(1))

    cov = cc._build_matrix_from_blocks(blocks, trs_cov)

    assert np.all(cov[:18, :18] == 0)
    assert np.all(cov[19:, 19:] == 1)
    assert np.all(cov[19:, :18] == 0)
    assert np.all(cov[:18, 19:] == 0)


def test_cluster_mass_tracer_missing_throws(save_cluster_sacc_data):
    # Keep only cluster_counts, so cluster_mean_log_mass is "missing".
    s = copy.deepcopy(save_cluster_sacc_data)
    s.reorder(s.indices(data_type=sacc.standard_types.cluster_counts))
    os.makedirs(OUTDIR, exist_ok=True)

    bad_sacc_file = os.path.join(OUTDIR, "test_cl_mass_fails_sacc.fits")
    s.save_fits(bad_sacc_file, overwrite=True)

    config = _load_config(sacc_file=bad_sacc_file)

    with pytest.raises(
        ValueError, match="Cluster mass covariance was requested"
    ):
        ClusterMass(config)

    os.remove(bad_sacc_file)


# -------------------
# External validation
# -------------------

N_Z_BINS = 6
N_LAMBDA_BINS = 3
LEN_NC = N_Z_BINS * N_LAMBDA_BINS

COV_REF_GAUSS_005 = np.diag(
    [
        9662.191886121263,
        12688.703662905858,
        19265.909425255675,
        20661.845270498427,
        23689.742361218232,
        30009.17124258014,
        4607.998294863565,
        5973.534063222964,
        8945.157180170849,
        9456.975028553872,
        10692.656431207164,
        13336.72847308123,
        2223.2774498819904,
        2842.332919951176,
        4193.715282176797,
        4366.509536275869,
        4864.393924854495,
        5968.457438220038,
    ]
)

COV_REF_GAUSS_05 = np.diag(
    [
        10194.12329803862,
        13048.195639926998,
        19532.821236429125,
        20761.58467107396,
        23667.08902891256,
        29815.699656727247,
        4835.783954606114,
        6113.75973919279,
        9030.402201864066,
        9465.651771980924,
        10644.33710939592,
        13208.94527333313,
        2320.176465888219,
        2894.8758154874317,
        4215.3348554211,
        4353.5471086757925,
        4825.520507128297,
        5893.518040166468,
    ]
)


COV_REF_SSC_005 = np.array(
    [
        [
            2.2298500589882657e04,
            -2.3919248275683431e03,
            -1.9773021661822836e03,
            -6.9163041103537694e02,
            -2.5679390409205558e02,
            -1.7855586323697815e02,
            1.1927001843587461e04,
            -1.2642178214875498e03,
            -1.0315815465248475e03,
            -3.5596651678642695e02,
            -1.3040668069338011e02,
            -8.9320177120171962e01,
            6.4294639760011887e03,
            -6.7249350130072332e02,
            -5.4092271061593351e02,
            -1.8389682175737045e02,
            -6.6393802162330587e01,
            -4.4740822443623941e01,
        ],
        [
            -2.3919248275683431e03,
            2.8104363192106615e04,
            -3.6263467677909534e03,
            -2.1066186092693783e03,
            -1.0090814797367124e03,
            -5.4048039442357970e02,
            -1.2793905901042638e03,
            1.4854161133960104e04,
            -1.8919073022490736e03,
            -1.0842289126883559e03,
            -5.1243804554821588e02,
            -2.7036807240445091e02,
            -6.8967841358495798e02,
            7.9015867836034367e03,
            -9.9204530132800062e02,
            -5.6012642116130837e02,
            -2.6089698806593964e02,
            -1.3542841395844641e02,
        ],
        [
            -1.9773021661822836e03,
            -3.6263467677909534e03,
            4.0412665071406831e04,
            -4.6479481740256142e03,
            -3.7079928171693209e03,
            -1.6529020563970828e03,
            -1.0576175956909792e03,
            -1.9166539675060528e03,
            2.1083757579674504e04,
            -2.3921937140313257e03,
            -1.8830160203047669e03,
            -8.2684209727540781e02,
            -5.7012770862747095e02,
            -1.0195532094884031e03,
            1.1055532486390854e04,
            -1.2358376428484771e03,
            -9.5869776345714638e02,
            -4.1416840691371704e02,
        ],
        [
            -6.9163041103537694e02,
            -2.1066186092693783e03,
            -4.6479481740256142e03,
            4.1644301633457020e04,
            -4.6076403074983791e03,
            -4.5441328187495019e03,
            -3.6993864920419315e02,
            -1.1134232807906126e03,
            -2.4248886424810939e03,
            2.1433379388672885e04,
            -2.3398806153688515e03,
            -2.2731415304442116e03,
            -1.9942205506304194e02,
            -5.9227920046845452e02,
            -1.2715207458405912e03,
            1.1072755900414413e04,
            -1.1913007051038155e03,
            -1.1386253910580722e03,
        ],
        [
            -2.5679390409205558e02,
            -1.0090814797367124e03,
            -3.7079928171693209e03,
            -4.6076403074983791e03,
            4.5653346342149722e04,
            -5.8457554399580140e03,
            -1.3735369134719414e02,
            -5.3333565307446088e02,
            -1.9345029961828936e03,
            -2.3714481675402280e03,
            2.3183966847167823e04,
            -2.9242607990154470e03,
            -7.4042967551176687e01,
            -2.8370487633414859e02,
            -1.0143808872066625e03,
            -1.2251202301553567e03,
            1.1803626163970219e04,
            -1.4647735529182314e03,
        ],
        [
            -1.7855586323697815e02,
            -5.4048039442357970e02,
            -1.6529020563970828e03,
            -4.5441328187495019e03,
            -5.8457554399580140e03,
            5.4685110867378280e04,
            -9.5505798761063502e01,
            -2.8566321939537892e02,
            -8.6233823476983594e02,
            -2.3387623006392209e03,
            -2.9686279577606933e03,
            2.7355493681144562e04,
            -5.1484111488053529e01,
            -1.5195692968320250e02,
            -4.5217785931790456e02,
            -1.2082342963497174e03,
            -1.5114141106356089e03,
            1.3702472667503924e04,
        ],
        [
            1.1927001843587461e04,
            -1.2793905901042638e03,
            -1.0576175956909792e03,
            -3.6993864920419315e02,
            -1.3735369134719414e02,
            -9.5505798761063502e01,
            6.3795039672525036e03,
            -6.7620368583973368e02,
            -5.5177140532916007e02,
            -1.9039904879942711e02,
            -6.9751807516231324e01,
            -4.7775495616293171e01,
            3.4389858809537509e03,
            -3.5970271622001121e02,
            -2.8932825060362126e02,
            -9.8362565827641347e01,
            -3.5512656898203801e01,
            -2.3930930674811361e01,
        ],
        [
            -1.2642178214875498e03,
            1.4854161133960104e04,
            -1.9166539675060528e03,
            -1.1134232807906126e03,
            -5.3333565307446088e02,
            -2.8566321939537892e02,
            -6.7620368583973368e02,
            7.8509554365430886e03,
            -9.9994067561781333e02,
            -5.7305375912927752e02,
            -2.7084183504583530e02,
            -1.4289919631062355e02,
            -3.6451970877184783e02,
            4.1762712250523400e03,
            -5.2433142346569423e02,
            -2.9604684719044928e02,
            -1.3789338949280227e02,
            -7.1578760539943957e01,
        ],
        [
            -1.0315815465248475e03,
            -1.8919073022490736e03,
            2.1083757579674504e04,
            -2.4248886424810939e03,
            -1.9345029961828936e03,
            -8.6233823476983594e02,
            -5.5177140532916007e02,
            -9.9994067561781333e02,
            1.0999641644346706e04,
            -1.2480352944093004e03,
            -9.8239136717659414e02,
            -4.3137314267251014e02,
            -2.9744225917586704e02,
            -5.3191277215819298e02,
            5.7677999321603893e03,
            -6.4475087756806045e02,
            -5.0016377789466492e02,
            -2.1607647684456001e02,
        ],
        [
            -3.5596651678642695e02,
            -1.0842289126883559e03,
            -2.3921937140313257e03,
            2.1433379388672885e04,
            -2.3714481675402280e03,
            -2.3387623006392209e03,
            -1.9039904879942711e02,
            -5.7305375912927752e02,
            -1.2480352944093004e03,
            1.1031275204521959e04,
            -1.2042835870997110e03,
            -1.1699344907975785e03,
            -1.0263801761538078e02,
            -3.0483269762558427e02,
            -6.5442294568999705e02,
            5.6988968186005122e03,
            -6.1313550658681004e02,
            -5.8602471480796896e02,
        ],
        [
            -1.3040668069338011e02,
            -5.1243804554821588e02,
            -1.8830160203047669e03,
            -2.3398806153688515e03,
            2.3183966847167823e04,
            -2.9686279577606933e03,
            -6.9751807516231324e01,
            -2.7084183504583530e02,
            -9.8239136717659414e02,
            -1.2042835870997110e03,
            1.1773426524800661e04,
            -1.4850163426958602e03,
            -3.7600961211194573e01,
            -1.4407277832420624e02,
            -5.1512922367505394e02,
            -6.2214819012057899e02,
            5.9941909977625182e03,
            -7.4385043398475716e02,
        ],
        [
            -8.9320177120171962e01,
            -2.7036807240445091e02,
            -8.2684209727540781e02,
            -2.2731415304442116e03,
            -2.9242607990154470e03,
            2.7355493681144562e04,
            -4.7775495616293171e01,
            -1.4289919631062355e02,
            -4.3137314267251014e02,
            -1.1699344907975785e03,
            -1.4850163426958602e03,
            1.3684219025430248e04,
            -2.5754236649649698e01,
            -7.6014417157096702e01,
            -2.2619591287513165e02,
            -6.0440301089073068e02,
            -7.5606464899295395e02,
            6.8544782761986598e03,
        ],
        [
            6.4294639760011887e03,
            -6.8967841358495798e02,
            -5.7012770862747095e02,
            -1.9942205506304194e02,
            -7.4042967551176687e01,
            -5.1484111488053529e01,
            3.4389858809537509e03,
            -3.6451970877184783e02,
            -2.9744225917586704e02,
            -1.0263801761538078e02,
            -3.7600961211194573e01,
            -2.5754236649649698e01,
            1.8538469370201972e03,
            -1.9390419204552725e02,
            -1.5596757583261171e02,
            -5.3024102944688053e01,
            -1.9143733791057596e01,
            -1.2900396822576647e01,
        ],
        [
            -6.7249350130072332e02,
            7.9015867836034367e03,
            -1.0195532094884031e03,
            -5.9227920046845452e02,
            -2.8370487633414859e02,
            -1.5195692968320250e02,
            -3.5970271622001121e02,
            4.1762712250523400e03,
            -5.3191277215819298e02,
            -3.0483269762558427e02,
            -1.4407277832420624e02,
            -7.6014417157096702e01,
            -1.9390419204552725e02,
            2.2215437963152995e03,
            -2.7891512745290748e02,
            -1.5748044160766969e02,
            -7.3351606606150185e01,
            -3.8075915776628698e01,
        ],
        [
            -5.4092271061593351e02,
            -9.9204530132800062e02,
            1.1055532486390854e04,
            -1.2715207458405912e03,
            -1.0143808872066625e03,
            -4.5217785931790456e02,
            -2.8932825060362126e02,
            -5.2433142346569423e02,
            5.7677999321603893e03,
            -6.5442294568999705e02,
            -5.1512922367505394e02,
            -2.2619591287513165e02,
            -1.5596757583261171e02,
            -2.7891512745290748e02,
            3.0244181704344269e03,
            -3.3808320199310259e02,
            -2.6226714446580223e02,
            -1.1330240827673153e02,
        ],
        [
            -1.8389682175737045e02,
            -5.6012642116130837e02,
            -1.2358376428484771e03,
            1.1072755900414413e04,
            -1.2251202301553567e03,
            -1.2082342963497174e03,
            -9.8362565827641347e01,
            -2.9604684719044928e02,
            -6.4475087756806045e02,
            5.6988968186005122e03,
            -6.2214819012057899e02,
            -6.0440301089073068e02,
            -5.3024102944688053e01,
            -1.5748044160766969e02,
            -3.3808320199310259e02,
            2.9441224470351217e03,
            -3.1675358678625298e02,
            -3.0274780756728791e02,
        ],
        [
            -6.6393802162330587e01,
            -2.6089698806593964e02,
            -9.5869776345714638e02,
            -1.1913007051038155e03,
            1.1803626163970219e04,
            -1.5114141106356089e03,
            -3.5512656898203801e01,
            -1.3789338949280227e02,
            -5.0016377789466492e02,
            -6.1313550658681004e02,
            5.9941909977625182e03,
            -7.5606464899295395e02,
            -1.9143733791057596e01,
            -7.3351606606150185e01,
            -2.6226714446580223e02,
            -3.1675358678625298e02,
            3.0518155136884047e03,
            -3.7871570911669386e02,
        ],
        [
            -4.4740822443623941e01,
            -1.3542841395844641e02,
            -4.1416840691371704e02,
            -1.1386253910580722e03,
            -1.4647735529182314e03,
            1.3702472667503924e04,
            -2.3930930674811361e01,
            -7.1578760539943957e01,
            -2.1607647684456001e02,
            -5.8602471480796896e02,
            -7.4385043398475716e02,
            6.8544782761986598e03,
            -1.2900396822576647e01,
            -3.8075915776628698e01,
            -1.1330240827673153e02,
            -3.0274780756728791e02,
            -3.7871570911669386e02,
            3.4334346996029708e03,
        ],
    ]
)

COV_REF_SSC_05 = np.array(
    [
        [
            6671.154697187735,
            4430.4587987459745,
            1446.654337630236,
            -362.6463990861584,
            -526.3663810779027,
            -404.5134507822957,
            3549.020493295438,
            2329.9382378753508,
            751.1637048219374,
            -185.8004030954049,
            -266.138773571256,
            -201.52666507407434,
            1901.9549766824143,
            1232.776419438187,
            391.92735928274254,
            -95.5393749933282,
            -134.90281891510907,
            -100.5382000102023,
        ],
        [
            4430.4587987459745,
            6650.6648926278385,
            5412.67219136646,
            1301.7464599329828,
            -388.9230417663591,
            -750.3284123488063,
            2356.9816298936985,
            3497.5245554737476,
            2810.486783535102,
            666.9444880551582,
            -196.6453502165861,
            -373.8100237669474,
            1263.1296295397785,
            1850.5493958182483,
            1466.4002750653037,
            342.9457551355062,
            -99.67736649114642,
            -186.4874155560799,
        ],
        [
            1446.654337630236,
            5412.67219136646,
            9861.45412818047,
            6223.741848327425,
            2053.0414852647345,
            -489.97655056665025,
            769.6127767999145,
            2846.475383388926,
            5120.481254692784,
            3188.7087451990437,
            1038.0487102165434,
            -244.10397233821053,
            412.4430539970731,
            1506.0775749802826,
            2671.663558191022,
            1639.6478989103489,
            526.1754809353688,
            -121.77928903454627,
        ],
        [
            -362.6463990861584,
            1301.7464599329828,
            6223.741848327425,
            8229.783168802118,
            5999.037903899393,
            2332.302900869768,
            -192.9260466286491,
            684.5766975364282,
            3231.628221779057,
            4216.495831764555,
            3033.2039578245185,
            1161.9421422103305,
            -103.39096525653031,
            362.2113223006927,
            1686.1351354105816,
            2168.1404868745517,
            1537.4977451206937,
            579.6730246634421,
        ],
        [
            -526.3663810779027,
            -388.9230417663591,
            2053.0414852647345,
            5999.037903899393,
            8410.397495622903,
            6759.366624571984,
            -280.0242474087335,
            -204.53111241181605,
            1066.0253856846132,
            3073.5825959885105,
            4252.423701810409,
            3367.4841002904263,
            -150.06774741282192,
            -108.21794686397917,
            556.209667290538,
            1580.4495325036019,
            2155.5068316344154,
            1679.9801151958309,
        ],
        [
            -404.5134507822957,
            -750.3284123488063,
            -489.97655056665025,
            2332.302900869768,
            6759.366624571984,
            10667.393209772194,
            -215.19910597264672,
            -394.5909302645168,
            -254.4164085543926,
            1194.9458762424672,
            3417.637615643666,
            5314.444240214384,
            -115.32731674993165,
            -208.7790939033031,
            -132.74436786926398,
            614.4463642979239,
            1732.3629405589081,
            2651.2851674955723,
        ],
        [
            3549.020493295438,
            2356.9816298936985,
            769.6127767999145,
            -192.9260466286491,
            -280.0242474087335,
            -215.19910597264672,
            1888.0609180207925,
            1239.515335751115,
            399.6152844958917,
            -98.8448729162469,
            -141.58447890034319,
            -107.2111645971697,
            1011.8304095716253,
            655.8308081330912,
            208.50336907402016,
            -50.82646335737006,
            -71.76761605226682,
            -53.48581293515198,
        ],
        [
            2329.9382378753508,
            3497.5245554737476,
            2846.475383388926,
            684.5766975364282,
            -204.53111241181605,
            -394.5909302645168,
            1239.515335751115,
            1839.3165515979576,
            1478.0095970772147,
            350.74007813812665,
            -103.41401231396122,
            -196.5833128438583,
            664.2684554681033,
            973.1872012023825,
            771.1666507027775,
            180.35207293506917,
            -52.419426110941956,
            -98.0720462877673,
        ],
        [
            751.1637048219374,
            2810.486783535102,
            5120.481254692784,
            3231.628221779057,
            1066.0253856846132,
            -254.4164085543926,
            399.6152844958917,
            1478.0095970772147,
            2658.768974520181,
            1655.7115354629018,
            538.998497892172,
            -126.74903703927451,
            214.1577600188993,
            782.0187459739886,
            1387.2399537376418,
            851.3740693347372,
            273.21241388825837,
            -63.232922711598405,
        ],
        [
            -185.8004030954049,
            666.9444880551582,
            3188.7087451990437,
            4216.495831764555,
            3073.5825959885105,
            1194.9458762424672,
            -98.8448729162469,
            350.74007813812665,
            1655.7115354629018,
            2160.3044375075146,
            1554.0496733306281,
            595.3163162249577,
            -52.971939248519476,
            185.5774932793398,
            863.8844577584797,
            1110.8379331599297,
            787.7306972344469,
            296.99310931365864,
        ],
        [
            -266.138773571256,
            -196.6453502165861,
            1038.0487102165434,
            3033.2039578245185,
            4252.423701810409,
            3417.637615643666,
            -141.58447890034319,
            -103.41401231396122,
            538.998497892172,
            1554.0496733306281,
            2150.089499233548,
            1702.6507024189254,
            -75.87651431548868,
            -54.71662456443281,
            281.2279887595476,
            799.0990978763168,
            1089.8567332907692,
            849.4232602141502,
        ],
        [
            -201.52666507407434,
            -373.8100237669474,
            -244.10397233821053,
            1161.9421422103305,
            3367.4841002904263,
            5314.444240214384,
            -107.2111645971697,
            -196.5833128438583,
            -126.74903703927451,
            595.3163162249577,
            1702.6507024189254,
            2647.6306841745254,
            -57.45551721854479,
            -104.01274531205566,
            -66.13260872369911,
            306.1142377945122,
            863.0549254507841,
            1320.8575807119398,
        ],
        [
            1901.9549766824143,
            1263.1296295397785,
            412.4430539970731,
            -103.39096525653031,
            -150.06774741282192,
            -115.32731674993165,
            1011.8304095716253,
            664.2684554681033,
            214.1577600188993,
            -52.971939248519476,
            -75.87651431548868,
            -57.45551721854479,
            542.2498649074881,
            351.4661782727964,
            111.73900551280104,
            -27.238401444099267,
            -38.46097106880772,
            -28.66357302982433,
        ],
        [
            1232.776419438187,
            1850.5493958182483,
            1506.0775749802826,
            362.2113223006927,
            -108.21794686397917,
            -208.7790939033031,
            655.8308081330912,
            973.1872012023825,
            782.0187459739886,
            185.5774932793398,
            -54.71662456443281,
            -104.01274531205566,
            351.4661782727964,
            514.915895124911,
            408.02629313917856,
            95.42475379686178,
            -27.735255544361266,
            -51.89017635929197,
        ],
        [
            391.92735928274254,
            1466.4002750653037,
            2671.663558191022,
            1686.1351354105816,
            556.209667290538,
            -132.74436786926398,
            208.50336907402016,
            771.1666507027775,
            1387.2399537376418,
            863.8844577584797,
            281.2279887595476,
            -66.13260872369911,
            111.73900551280104,
            408.02629313917856,
            723.8066592804707,
            444.21314370515813,
            142.55137623279055,
            -32.99242529823941,
        ],
        [
            -95.5393749933282,
            342.9457551355062,
            1639.6478989103489,
            2168.1404868745517,
            1580.4495325036019,
            614.4463642979239,
            -50.82646335737006,
            180.35207293506917,
            851.3740693347372,
            1110.8379331599297,
            799.0990978763168,
            306.1142377945122,
            -27.238401444099267,
            95.42475379686178,
            444.21314370515813,
            571.197694326234,
            405.05454898390894,
            152.71514791376626,
        ],
        [
            -134.90281891510907,
            -99.67736649114642,
            526.1754809353688,
            1537.4977451206937,
            2155.5068316344154,
            1732.3629405589081,
            -71.76761605226682,
            -52.419426110941956,
            273.21241388825837,
            787.7306972344469,
            1089.8567332907692,
            863.0549254507841,
            -38.46097106880772,
            -27.735255544361266,
            142.55137623279055,
            405.05454898390894,
            552.4363983558094,
            430.5633137077293,
        ],
        [
            -100.5382000102023,
            -186.4874155560799,
            -121.77928903454627,
            579.6730246634421,
            1679.9801151958309,
            2651.2851674955723,
            -53.48581293515198,
            -98.0720462877673,
            -63.232922711598405,
            296.99310931365864,
            849.4232602141502,
            1320.8575807119398,
            -28.66357302982433,
            -51.89017635929197,
            -32.99242529823941,
            152.71514791376626,
            430.5633137077293,
            658.9532138875885,
        ],
    ]
)


@pytest.mark.parametrize(
    "sigma_0, ref_gauss, ref_ssc",
    [
        (0.005, COV_REF_GAUSS_005, COV_REF_SSC_005),
        (0.05, COV_REF_GAUSS_05, COV_REF_SSC_05),
    ],
)
def test_cluster_count_covariance_matches_reference(
    save_cluster_sacc_data, sigma_0, ref_gauss, ref_ssc
):
    """Gaussian  SSC cluster-count covariance vs. an external reference."""
    config = _load_config(sigma_0=sigma_0)

    cc = CovarianceCalculator(config)
    cov_terms = cc.get_covariance_terms()

    cov_gauss = cov_terms["gauss"][:LEN_NC, :LEN_NC]
    cov_ssc = cov_terms["SSC"][:LEN_NC, :LEN_NC]

    np.testing.assert_allclose(cov_gauss, ref_gauss, rtol=1e-3)
    np.testing.assert_allclose(cov_ssc, ref_ssc, rtol=5e-3)
