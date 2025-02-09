"""LDDMM functions.

Set of functions that wrap around [deformetrica](
https://gitlab.com/icm-institute/aramislab/deformetrica) to perform registration, parallel
transport, geodesic and spline regression with the Large Deformations Diffeomorphic Metric
Mapping (LDDMM) framework.

For a brief introduction to LDDMM see [NG22](https://hal.science/tel-03563980v1) chapter 5.
"""

import numpy as np
import pyvista as pv
import time
import torch
from os.path import join
from api.deformetrica import Deformetrica
from in_out.array_readers_and_writers import read_2D_array, read_3D_array
from launch.compute_parallel_transport import compute_pole_ladder
from launch.compute_shooting import compute_shooting
from support.kernels.torch_kernel import TorchKernel

import herbrain.strings as strings


def registration(
        source, target, output_dir, kernel_width=20.0, regularisation=1.0,
        number_of_time_steps=11, metric='landmark', kernel_type='torch',
        kernel_device='cuda', tol=1e-5,
        use_svf=False, initial_control_points=None, max_iter=200,
        freeze_control_points=False, use_rk2_for_shoot=False, use_rk2_for_flow=False,
        dimension=3, use_rk4_for_shoot=False, preserve_volume=False, print_every=20,
        filter_cp=False, threshold=1., attachment_kernel_width=4.):
    r"""Registration

    Estimates the best possible deformation between two shapes, i.e. solves the following
    optimization problem:

    ..math::
         C(c, \mu) = \frac{1}{\alpha^2} d(q, \phi_1^{c,\mu}(\bar{q}))^2 + \| v_0^{c,
         \mu} \|_K^2.

    where $c, \mu$ are the control points and momenta that parametrize the deformation, $v_0^{c,
    \mu}$ is the associated velocity field defined by the convolution $v_t(x) = \sum_{k=1}^{N_c}
    K(x, c^{(t)}_k) \mu^{(t)}_K$, K is the Gaussian kernel, $\phi_1^{c,\mu}$ is the flow of $v_t$
    at time 1, $\bar{q}$ is the source shape being deformed, $q$ is the target shape,
    and $\alpha$ is a regularization term that controls the tradeoff between exact matching and
    smoothness of the deformation. $d$ is a distance function on shapes (point-to-point L2,
    varifold, metric, etc).

    Control points can be passed as parameters or are initialized on a grid that contains the
    source shapes. They are optimized if `freeze_control_points` is set to false.

    Resulting control points and momenta are saved in the ouput dir as txt files. Control points
    are also saved with attached momenta as a vtk file to allow visualization with paraview.


    Parameters
    ----------
    source: str or pathlib.Path
        Path to the vtk file that contains the source mesh.
    target: str or pathlib.Path
        Path to the vtk file that contains the target mesh.
    output_dir: str or pathlib.Path
        Path a directory where results will be saved.
    kernel_width: float
        Width of the Gaussian kernel. Controls the spatial smoothness of the deformation and
        influences the number of parameters required to represent the deformation.
        Optional, default: 20.
    regularisation: float
        $\alpha$ in the above equation. Smaller values will yeild larger deformations to reduce
        the data attachment term, while larger values will allow attachment errors for a smoother
        deformation.
        Optional, default: 1.
    number_of_time_steps: int
        Number used in the discretization of the flow equation.
        Optional, default: 11.
    metric: str, {landmark, varifold, current}
        Metric to use to measure attachment between meshes. Landmark refers to L2.
    attachment_kernel_width: float,
        If using varifold or currents, width of the kernel used in the attachment metric. Defines
        the scale at which differences must be taken into account.
    dimension: int {2, 3}
        Dimension of the shape embedding space.
    kernel_type: str, {torch, keops}
        Package to use for convolutions of velocity fields and loss functions.
    kernel_device: str, {cuda, cpu}
    use_svf: bool
        Whether to use stationnary velocity fields insteads of time evolving velocity. The
        deformation is no longer a geodesic but there is more symmetry wrt source / target.
        Optional, default: False
    initial_control_points: str or pathlib.Path
        Path to the txt file that contains the initial control points.
        Optional
    freeze_control_points: bool
        Wether to optimize control points jointly with momenta.
        Optional, default: False
    preserve_volume: bool
        Whether to use volume preserving deformation. This modifies the metric on deformations.
        Optional, default: False
    use_rk2_for_flow: bool
        Wether to use Runge-Kutta order 2 steps in the integration of the flow equation, i.e. when
        warping the shape. If False, a Euler step is used.
        Optional, default: False
    use_rk2_for_shoot: bool
        Wether to use Runge-Kutta order 2 steps in the integration of the Hamiltonian equation that
        governs the time evolution of control points and momenta. If False, a Euler step is used.
        Optional, default: False
    use_rk4_for_shoot: bool
        Wether to use Runge-Kutta order 4 steps in the integration of the Hamiltonian equation that
        governs the time evolution of control points and momenta. Overrides use_rk2_for_shoot.
        RK4 steps are required when estimating a geodesic that will be used for parallel transport.
        Optional, default: False
    print_every: int
        Sets the verbosity level of the optimization scheme.
    filter_cp: bool
        Whether to filter control points saved in the vtk file to exclude those whose momenum
        vector is not significative and does not contribute to the deformation.
        Optional, default: False
    threshold: float
        Threshold to use on momenta norm when filtering.
    """
    optimization_parameters = {
        'max_iterations': max_iter,
        'freeze_template': False, 'freeze_control_points': freeze_control_points,
        'freeze_momenta': False, 'use_sobolev_gradient': True,
        'sobolev_kernel_width_ratio': 1, 'max_line_search_iterations': 50,
        'initial_control_points': initial_control_points,
        'initial_cp_spacing': None, 'initial_momenta': None,
        'dense_mode': False, 'number_of_threads': 1, 'print_every_n_iters': print_every,
        'downsampling_factor': 1, 'dimension': dimension,
        'optimization_method_type': 'ScipyLBFGS', 'convergence_tolerance': tol}

    # register source on target
    deformetrica = Deformetrica(output_dir, verbosity='DEBUG')

    model_options = {
        'deformation_kernel_type': kernel_type,
        'deformation_kernel_width': kernel_width,
        'deformation_kernel_device': kernel_device,
        'use_svf': use_svf, 'preserve_volume': preserve_volume,
        'number_of_time_points': number_of_time_steps,
        'use_rk2_for_shoot': use_rk2_for_shoot,
        'use_rk4_for_shoot': use_rk4_for_shoot,
        'use_rk2_for_flow': use_rk2_for_flow,
        'freeze_template': False,
        'freeze_control_points': freeze_control_points,
        'initial_control_points': initial_control_points,
        'dimension': dimension,
        'output_dir': output_dir
    }

    template = {
        'shape': {
            'deformable_object_type': 'SurfaceMesh', 'kernel_type': kernel_type,
            'kernel_width': attachment_kernel_width, 'kernel_device': kernel_device,
            'noise_std': regularisation, 'filename': source,
            'noise_variance_prior_scale_std': None,
            'noise_variance_prior_normalized_dof': 0.01, 'attachment_type': metric}}

    data_set = {'visit_ages': [[]], 'dataset_filenames': [[{'shape': target}]],
                'subject_ids': ['ventricle']}

    deformetrica.estimate_registration(
        template_specifications=template, dataset_specifications=data_set,
        model_options=model_options, estimator_options=optimization_parameters)

    path_cp = join(output_dir, strings.cp_str)
    cp = read_2D_array(path_cp)

    path_momenta = join(output_dir, strings.momenta_str)
    momenta = read_3D_array(path_momenta)
    poly_cp = momenta_to_vtk(cp, momenta, kernel_width, filter_cp, threshold)
    poly_cp.save(join(output_dir, 'initial_control_points.vtk'))
    pv.read(target).save(join(output_dir, 'target_shape.vtk'))
    return time.gmtime()


def spline_regression(
        source, target, output_dir, times, subject_id='patient', t0=0,
        max_iter=200, kernel_width=15.0, regularisation=1.0, number_of_time_steps=11,
        initial_step_size=1e-4, kernel_type='torch', kernel_device='cuda',
        initial_control_points=None, tol=1e-5, freeze_control_points=False,
        use_rk2_for_flow=False, dimension=3, freeze_external_forces=False,
        target_weights=None, geodesic_weight=0.1, metric='landmark',
        filter_cp=False, threshold=1., attachment_kernel_width=15.):
    """
    :param geodesic_weight:
    :param target_weights:
    :param max_iter:
    :param initial_step_size:
    :param freeze_external_forces:
    :param t0:
    :param source: path to template
    :param target: list of dict with 'shape' as keys and path to file as value
    :param output_dir:
    :param times: list of observed times
    :param subject_id:
    :param kernel_width:
    :param regularisation:
    :param number_of_time_steps:
    :param kernel_type:
    :param kernel_device:
    :param use_svf:
    :param initial_control_points:
    :param freeze_control_points:
    :param use_rk2_for_shoot:
    :param use_rk2_for_flow:
    :param dimension:
    :param metric
    :param tol
    :return:
    """
    template = {
        'shape': {
            'deformable_object_type': 'SurfaceMesh', 'kernel_type': kernel_type,
            'kernel_width': attachment_kernel_width, 'kernel_device': kernel_device,
            'noise_std': regularisation, 'filename': source,
            'noise_variance_prior_scale_std': None,
            'noise_variance_prior_normalized_dof': 0.01,
            'attachment_type': metric}}

    data_set = {
        'visit_ages': [times], 'dataset_filenames': [target], 'subject_ids': subject_id}

    model = {'deformation_kernel_type': kernel_type,
             'deformation_kernel_width': kernel_width,
             'deformation_kernel_device': kernel_device,
             'number_of_time_points': number_of_time_steps,
             'concentration_of_time_points': number_of_time_steps - 1,
             'use_rk2_for_flow': use_rk2_for_flow, 'freeze_template': True,
             'freeze_control_points': freeze_control_points,
             'freeze_external_forces': freeze_external_forces,
             'freeze_momenta': False, 'freeze_noise_variance': False,
             'use_sobolev_gradient': True,
             'sobolev_kernel_width_ratio': 1,
             'initial_control_points': initial_control_points,
             'initial_cp_spacing': None, 'initial_momenta': None, 'dense_mode': False,
             'number_of_processes': 1, 'dimension': dimension,
             'random_seed': None, 't0': t0, 'tmin': min(times), 'tmax': max(times),
             'target_weights': target_weights,
             'geodesic_weight': geodesic_weight}

    optimization_parameters = {
        'initial_step_size': initial_step_size, 'scale_initial_step_size': True,
        'line_search_shrink': 0.5, 'line_search_expand': 1.5,
        'max_line_search_iterations': 30, 'optimized_log_likelihood': 'complete',
        'optimization_method_type': 'ScipyLBFGS', 'max_iterations': max_iter,
        'convergence_tolerance': tol, 'print_every_n_iters': 20,
        'save_every_n_iters': 100, 'state_file': None, 'load_state_file': False}

    if subject_id != 'patient':
        patient_output_dir = join(output_dir, subject_id[0])
    else:
        patient_output_dir = output_dir

    deformetrica = Deformetrica(patient_output_dir, verbosity='DEBUG')
    deformetrica.estimate_spline_regression(
        template_specifications=template, dataset_specifications=data_set,
        model_options=model, estimator_options=optimization_parameters)

    # agregate results in vtk file for paraview
    path_cp = join(output_dir, strings.cp_str_spline)
    cp = read_2D_array(path_cp)
    path_momenta = join(output_dir, strings.mom_str_spline)
    momenta = read_3D_array(path_momenta)
    poly_cp = momenta_to_vtk(
        cp, momenta, kernel_width, filter_cp, threshold)
    poly_cp.save(join(output_dir, 'initial_control_points.vtk'))

    if not freeze_external_forces:
        forces = read_3D_array(join(output_dir, strings.ext_forces_str))
        external_forces_to_vtk(cp, forces, output_dir, filter_cp, threshold)

    return time.gmtime()


def transport(
        control_points, momenta, control_points_to_transport,
        momenta_to_transport, output_dir,
        kernel_type='torch', kernel_width=15, kernel_device='cuda', n_rungs=10):

    Deformetrica(output_dir, verbosity='INFO')

    deformation_parameters = {
        'deformation_kernel_type': kernel_type,
        'deformation_kernel_width': kernel_width,
        'deformation_kernel_device': kernel_device,
        'concentration_of_time_points': n_rungs,
        'number_of_time_points': n_rungs + 1,
        'tmin': 0,
        'tmax': 1,
        'output_dir': output_dir}

    _, _ = compute_pole_ladder(
        initial_control_points=control_points,
        initial_momenta=momenta,
        initial_momenta_to_transport=momenta_to_transport,
        initial_control_points_to_transport=control_points_to_transport,
        **deformation_parameters)


def shoot(source, control_points, momenta, output_dir, kernel_width=20.0,
          regularisation=1.0, number_of_time_steps=11,
          kernel_type='torch', kernel_device='cuda', write_params=True,
          deformation='geodesic', external_forces=None, use_rk2_for_flow=False):
    """
    Wrapper to deformetrica compute_shooting
    :param external_forces:
    :param deformation:
    :param source:
    :param control_points:
    :param momenta:
    :param output_dir:
    :param kernel_width:
    :param regularisation:
    :param number_of_time_steps:
    :param kernel_type:
    :param kernel_device:
    :param write_params:
    :return:
    """
    deformation_parameters = {
        'deformation_model': deformation,
        'deformation_kernel_type': kernel_type,
        'deformation_kernel_width': kernel_width,
        'deformation_kernel_device': kernel_device,
        'concentration_of_time_points': number_of_time_steps - 1,
        'number_of_time_points': number_of_time_steps,
        'use_rk2_for_flow': use_rk2_for_flow,
        'output_dir': output_dir,
        'write_adjoint_parameters': write_params}

    template_specifications = {
        'shape': {
            'deformable_object_type': 'landmark',
            'kernel_type': kernel_type, 'kernel_width': kernel_width,
            'kernel_device': kernel_device, 'noise_std': regularisation,
            'filename': source,
            'noise_variance_prior_scale_std': None,
            'noise_variance_prior_normalized_dof': 0.01}}

    Deformetrica(output_dir, verbosity='INFO')
    compute_shooting(
        template_specifications,
        initial_control_points=control_points,
        external_forces=external_forces,
        initial_momenta=momenta, **deformation_parameters)

    return time.gmtime()


def deterministic_atlas(
        source, targets, subject_id, output_dir, t0=0, max_iter=200,
        kernel_width=15.0, regularisation=1.0, number_of_time_steps=11,
        metric='landmark', kernel_type='torch',
        kernel_device='auto', initial_control_points=None, tol=1e-5,
        freeze_control_points=False, use_rk2_for_flow=False, dimension=3,
        print_every=20, filter_cp=False, threshold=1., attachment_kernel_width=4.,
        initial_step_size=1e-4):
    template = {
        'shape': {
            'deformable_object_type': 'SurfaceMesh', 'kernel_type': kernel_type,
            'kernel_width': attachment_kernel_width, 'kernel_device': kernel_device,
            'noise_std': regularisation, 'filename': source,
            'noise_variance_prior_scale_std': None,
            'noise_variance_prior_normalized_dof': 0.01,
            'attachment_type': metric}}

    data_set = {'dataset_filenames': [[k] for k in targets],
                'visit_ages': None, #[[1.]] * len(targets),
                'subject_ids': [subject_id] * len(targets)}

    model = {
        'deformation_kernel_type': kernel_type,
        'deformation_kernel_width': kernel_width,
        'deformation_kernel_device': kernel_device,
        'number_of_time_points': number_of_time_steps,
        'concentration_of_time_points': number_of_time_steps - 1,
        'use_rk2_for_flow': use_rk2_for_flow, 'freeze_template': False,
        'freeze_control_points': freeze_control_points,
        'freeze_momenta': False, 'freeze_noise_variance': False,
        'use_sobolev_gradient': True,
        'sobolev_kernel_width_ratio': 1,
        'initial_control_points': initial_control_points,
        'initial_cp_spacing': None, 'initial_momenta': None, 'dense_mode': False,
        'number_of_processes': 1, 'dimension': dimension,
        'random_seed': None, 't0': t0, 'tmin': t0, 'tmax': 1.}

    optimization_parameters = {
        'max_iterations': max_iter,
        'freeze_template': False, 'freeze_control_points': freeze_control_points,
        'freeze_momenta': False, 'use_sobolev_gradient': True,
        'sobolev_kernel_width_ratio': 1, 'max_line_search_iterations': 50,
        'initial_control_points': initial_control_points,
        'initial_cp_spacing': None, 'initial_momenta': None,
        'dense_mode': False, 'number_of_threads': 1, 'print_every_n_iters': print_every,
        'downsampling_factor': 1, 'dimension': dimension,
        'optimization_method_type': 'ScipyLBFGS', 'convergence_tolerance': tol,
        'initial_step_size': initial_step_size}

    deformetrica = Deformetrica(output_dir, verbosity='DEBUG')
    deformetrica.estimate_deterministic_atlas(
        template_specifications=template, dataset_specifications=data_set,
        model_options=model, estimator_options=optimization_parameters)
    path_cp = join(output_dir, strings.cp_str)
    cp = read_2D_array(path_cp)

    # path_momenta = join(output_dir, strings.momenta_str)
    # momenta = read_3D_array(path_momenta)
    # poly_cp = momenta_to_vtk(cp, momenta, kernel_width, filter_cp, threshold)
    # poly_cp.save(join(output_dir, 'initial_control_points.vtk'))
    return time.gmtime()


def momenta_to_vtk(cp, momenta, kernel_width=5., filter_cp=True, threshold=1.):
    kernel = TorchKernel(kernel_width=kernel_width)
    velocity = kernel.convolve(cp, cp, momenta).cpu()

    if filter_cp:
        vel_thresholded = np.linalg.norm(velocity, axis=-1) > threshold
        cp = cp[vel_thresholded, :]
        momenta = momenta[vel_thresholded, :]
        velocity = velocity[vel_thresholded, :]

    poly = pv.PolyData(cp)
    poly['Momentum'] = momenta
    poly['Velocity'] = velocity
    return poly


def external_forces_to_vtk(cp, forces, output_dir, filter_cp=True, threshold=1.):
    mask = np.linalg.norm(forces, axis=-1) > threshold
    for i, f in enumerate(forces[:-1]):
        filename = join(output_dir, f'cp_with_external_forces_{i}.vtk')
        if filter_cp:
            cp_filtered = cp[mask[i], :]
            f = f[mask[i]]
            poly_cp = pv.PolyData(cp_filtered)
        else:
            poly_cp = pv.PolyData(cp)
        poly_cp['external_force'] = f
        poly_cp.save(filename)


def deformation_norm(atlas_dir, kernel_width):
    momenta = torch.from_numpy(read_3D_array(atlas_dir / strings.momenta_str))
    cp = torch.from_numpy(read_2D_array(atlas_dir / strings.cp_str))
    kernel = TorchKernel(kernel_width=kernel_width)
    kernel_matrix = kernel.get_kernel_matrix(cp, cp)
    ssd = (torch.einsum('...ij,...kj->...ik', momenta, momenta) * kernel_matrix).sum()
    return ssd
